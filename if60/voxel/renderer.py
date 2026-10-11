"""복셀 장면 렌더러 (moderngl, 화면 없이 오프스크린 렌더링).

프레임 하나 = 그림자 깊이 패스 → 하늘 → 정적 세계 → 군중(인스턴싱) → MSAA 해상 → RGB 읽기
GPU가 있으면 1080x1920도 빠르고, GPU가 없는 서버에서도 소프트웨어 렌더러(llvmpipe)로 동작한다.
"""
from __future__ import annotations

import math
import platform
from dataclasses import dataclass, field

import numpy as np

from . import models
from .camera import camera_at, gl_bytes, look_at, ortho, perspective
from .geometry import rgb
from .shaders import COMMON_FS, DEPTH_FS, INSTANCE_VS, SKY_FS, SKY_VS, WORLD_VS
from .sky import Lighting, parse_time, preset

INSTANCE_FMT = "3f 1f 3f 3f 1f 3f 1f/i"   # off yaw c1 c2 phase vel scale
INSTANCE_ATTRS = ("i_off", "i_yaw", "i_c1", "i_c2", "i_phase", "i_vel", "i_scale")


def create_context():
    import moderngl
    if platform.system() == "Linux":
        try:
            return moderngl.create_standalone_context(backend="egl")
        except Exception:
            pass
    return moderngl.create_standalone_context(require=330)


# ------------------------------------------------------------------ crowd
@dataclass
class Crowd:
    model: str
    instances: np.ndarray                  # (N, 15) float32
    bob: float = 0.0
    bob_freq: float = 6.0
    wrap_min: tuple = (0.0, 0.0, 0.0)
    wrap_size: tuple = (0.0, 0.0, 0.0)
    spin: float = 0.0


def make_crowd(spec: dict, areas: dict, rng: np.random.Generator, avoid: np.ndarray | None = None) -> Crowd:
    """장면 YAML의 crowds 항목 → 인스턴스 배열.

    agent/model: person|cat|dog|car|train|astronaut|tree   count: 개수
    area: 템플릿이 정의한 구역 이름 또는 [x0, z0, x1, z1]
    layout: random | grid | queue(한 줄로 이어짐)
    facing: random | 각도(도) | [x, z](그 지점을 바라봄) | velocity
    velocity: [vx, vz] (m/s) — 구역 안에서 순환 이동
    bob: 들썩임 높이(m), bob_freq: 빠르기, scale: [최소, 최대]
    colors / colors2: 색 목록으로 팔레트 교체
    """
    name = spec.get("agent", spec.get("model", "person"))
    mdl = models.get(name)
    area = spec.get("area", [-10, -10, 10, 10])
    x0, z0, x1, z1 = areas[area] if isinstance(area, str) else area
    n = int(spec.get("count", 100))
    layout = spec.get("layout", "random")
    y = float(spec.get("y", 0.0))
    if layout == "grid":
        cols = max(1, int(round(math.sqrt(n * (x1 - x0) / max(1e-6, z1 - z0)))))
        rows = math.ceil(n / cols)
        gx = (np.arange(n) % cols + 0.5) / cols
        gz = (np.arange(n) // cols + 0.5) / rows
        jit = float(spec.get("jitter", 0.15))
        xs = x0 + gx * (x1 - x0) + rng.normal(0, jit, n)
        zs = z0 + gz * (z1 - z0) + rng.normal(0, jit, n)
    elif layout == "queue":   # 한 줄로 이어 세우기 (열차, 차량 대기열, 줄 선 사람)
        gap = float(spec.get("gap", 6.0))
        jit = rng.normal(0, float(spec.get("jitter", 0.0)), n)
        if spec.get("axis", "z") == "x":
            xs = x0 + np.arange(n) * gap
            zs = np.full(n, (z0 + z1) / 2) + jit
        else:
            xs = np.full(n, (x0 + x1) / 2) + jit
            zs = z0 + np.arange(n) * gap
    else:
        xs = rng.uniform(x0, x1, n)
        zs = rng.uniform(z0, z1, n)
    vel = np.zeros((n, 3), np.float32)
    y_range = spec.get("y_range")
    if "wander" in spec:     # 개체마다 제각각 떠다니는 방향 (기체 분자, 먼지, 반딧불 등)
        d = rng.normal(0, 1, (n, 3))
        if not y_range:
            d[:, 1] = 0
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        vel = (d * float(spec["wander"]) * rng.uniform(0.6, 1.4, (n, 1))).astype(np.float32)
    if "velocity" in spec:   # [vx, vz] 또는 위아래 포함 [vx, vy, vz]
        v = list(spec["velocity"])
        vx, vy, vz = (v[0], 0.0, v[1]) if len(v) == 2 else v
        spread = float(spec.get("speed_jitter", 0.15))
        k = 1 + rng.normal(0, spread, n)
        vel[:, 0], vel[:, 1], vel[:, 2] = vx * k, vy * k, vz * k
    facing = spec.get("facing", "velocity" if "velocity" in spec else "random")
    if facing == "random":
        yaw = rng.uniform(0, 2 * math.pi, n)
    elif facing == "velocity":
        yaw = np.arctan2(vel[:, 0], vel[:, 2]) if np.any(vel[:, [0, 2]]) else np.zeros(n)
    elif isinstance(facing, (list, tuple)):
        yaw = np.arctan2(facing[0] - xs, facing[1] - zs)
    else:
        yaw = np.full(n, math.radians(float(facing))) + rng.normal(0, math.radians(float(spec.get("yaw_jitter", 8))), n)
    p1 = [rgb(c) for c in spec.get("colors", mdl.palette1)]
    p2 = [rgb(c) for c in spec.get("colors2", mdl.palette2)]
    c1 = np.stack([p1[i] for i in rng.integers(0, len(p1), n)])
    c2 = np.stack([p2[i] for i in rng.integers(0, len(p2), n)])
    smin, smax = spec.get("scale", [0.92, 1.08]) if isinstance(spec.get("scale"), list) else (spec.get("scale", 1.0),) * 2
    inst = np.zeros((n, 15), np.float32)
    if "y" not in spec and isinstance(area, str) and f"__y_{area}" in areas:   # 탁자 위 같은 구역의 기본 높이
        y = float(areas[f"__y_{area}"])
    ys = np.full(n, y, np.float32)
    if "__height__" in areas and "y" not in spec:   # 울퉁불퉁한 지형 위에 세우기
        ys = areas["__height__"](xs, zs).astype(np.float32)
    if y_range:
        ys = rng.uniform(float(y_range[0]), float(y_range[1]), n).astype(np.float32)
    inst[:, 0], inst[:, 1], inst[:, 2] = xs, ys, zs
    inst[:, 3] = yaw
    inst[:, 4:7], inst[:, 7:10] = c1, c2
    inst[:, 10] = rng.uniform(0, 2 * math.pi, n)
    inst[:, 11:14] = vel
    inst[:, 14] = rng.uniform(smin, smax, n)
    # 소수의 주인공 개체(파리, 우주비행사 등)는 일부러 카메라 가까이 두므로 비우지 않음
    if avoid is not None and len(avoid) and not np.any(vel) and spec.get("avoid_camera", n > 20):
        # 카메라가 지나가는 길 주변(반경 clearance m)에 서 있는 개체는 빼서 렌즈를 가리지 않게
        clr = float(spec.get("clearance", 3.0))
        d2 = (inst[:, None, 0] - avoid[None, :, 0]) ** 2 + (inst[:, None, 2] - avoid[None, :, 1]) ** 2
        inst = inst[d2.min(1) > clr * clr]
    wrap = spec.get("wrap", "velocity" in spec or "wander" in spec)
    if len(inst) == 0:
        return Crowd(name, inst)
    v = inst[:, 11:14]
    ylo, yhi = (float(y_range[0]), float(y_range[1])) if y_range else (0.0, 0.0)
    wsize = (x1 - x0 if (wrap and abs(v[:, 0]).max() > 0) else 0,
             yhi - ylo if (wrap and y_range and abs(v[:, 1]).max() > 0) else 0,
             z1 - z0 if (wrap and abs(v[:, 2]).max() > 0) else 0)
    return Crowd(name, inst, bob=float(spec.get("bob", 0.0)), bob_freq=float(spec.get("bob_freq", 6.0)),
                 wrap_min=(x0, ylo, z0), wrap_size=wsize, spin=float(spec.get("spin", 0.0)))


def add_props(builder, props: list, rng) -> None:
    """장면 YAML의 props → 정적 세계에 박아 넣기.
      - {model: snack_bag, at: [x,y,z], yaw: 도, scale: 배율, color: "#hex", color2: "#hex"}
      - {shape: box, size: [w,h,d], at: [x,y,z](밑면 중심), color, mat, yaw}
      - {shape: sphere, r: 반지름 또는 [rx,ry,rz], at: [x,y,z](중심), color, mat}
      - {shape: cylinder, r, h, at: [x,y,z](밑면 중심), color, mat, axis: y|x|z}
    """
    import math as _m
    from .geometry import Model, rgb as _rgb
    for pr in props or []:
        at = np.asarray(pr.get("at", [0, 0, 0]), np.float32)
        yaw = _m.radians(float(pr.get("yaw", 0)))
        scale = float(pr.get("scale", 1.0))
        mat = pr.get("mat", "flat")
        if "model" in pr:
            mdl = models.get(pr["model"])
            c1 = pr.get("color", mdl.palette1[rng.integers(len(mdl.palette1))])
            c2 = pr.get("color2", mdl.palette2[rng.integers(len(mdl.palette2))])
            builder.stamp(mdl, at, yaw, c1=c1, c2=c2, scale=scale)
            continue
        shape = pr.get("shape", "box")
        color = pr.get("color", "#cccccc")
        m = Model("prop", [color], [color])
        if shape == "box":
            w, h, d = (np.asarray(pr.get("size", [1, 1, 1]), np.float32) * scale)
            m.part((-w / 2, 0, -d / 2), (w / 2, h, d / 2), color, mat)
        elif shape == "sphere":
            r = np.asarray(pr.get("r", 0.5), np.float32) * scale
            m.sphere((0, 0, 0), r if r.size > 1 else float(r), color, mat=mat)
        elif shape == "cylinder":
            m.cylinder((0, 0, 0), float(pr.get("r", 0.5)) * scale, float(pr.get("h", 1.0)) * scale, color,
                       mat=mat, axis=pr.get("axis", "y"))
        else:
            raise ValueError(f"알 수 없는 소품 shape: {shape} (box/sphere/cylinder 또는 model)")
        builder.stamp(m, at, yaw)


# ------------------------------------------------------------------ renderer
@dataclass
class Prepared:
    world_vao: object
    world_depth_vao: object
    world_vbo: object
    crowds: list = field(default_factory=list)   # (Crowd, vao, depth_vao, vbo_model, vbo_inst)
    shots: dict = field(default_factory=dict)
    shadow_radius: float = 90.0
    sky: str = "auto"
    sun_azimuth: float = 35.0
    lighting_override: dict = field(default_factory=dict)
    texel: float = 8.0


class VoxelRenderer:
    def __init__(self, width: int, height: int, samples: int = 4, shadow_size: int = 2048):
        self.ctx = create_context()
        self.w, self.h = width, height
        ctx = self.ctx
        self.world_prog = ctx.program(vertex_shader=WORLD_VS, fragment_shader=COMMON_FS)
        self.inst_prog = ctx.program(vertex_shader=INSTANCE_VS, fragment_shader=COMMON_FS)
        self.world_depth = ctx.program(vertex_shader=WORLD_VS, fragment_shader=DEPTH_FS)
        self.inst_depth = ctx.program(vertex_shader=INSTANCE_VS, fragment_shader=DEPTH_FS)
        self.sky_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        tri = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.sky_vao = ctx.vertex_array(self.sky_prog, [(ctx.buffer(tri.tobytes()), "2f", "in_pos")])

        self.shadow_tex = ctx.depth_texture((shadow_size, shadow_size))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)

        try:
            self.ms_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((width, height), samples=samples)],
                                          depth_attachment=ctx.depth_renderbuffer((width, height), samples=samples))
        except Exception:
            self.ms_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((width, height))],
                                          depth_attachment=ctx.depth_renderbuffer((width, height)))
        self.out_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((width, height))])

    # -------------------------------------------------------------- scene setup
    def prepare(self, world: dict, crowds: list[dict], seed: int = 0, shot=None, props=None) -> Prepared:
        from .templates import build_template
        rng = np.random.default_rng(seed)
        built = build_template(world, rng)
        add_props(built.builder, list(built.props) + list(props or []), rng)
        ctx = self.ctx
        data = built.builder.build()
        vbo = ctx.buffer(data.tobytes())
        layout = (vbo, "3f 3f 3f 1f", "in_pos", "in_norm", "in_col", "in_mat")
        prep = Prepared(
            world_vao=ctx.vertex_array(self.world_prog, [layout]),
            world_depth_vao=ctx.vertex_array(self.world_depth, [(vbo, "3f 12x 12x 4x", "in_pos")]),
            world_vbo=vbo, shots=built.shots, shadow_radius=built.shadow_radius,
            sky=world.get("sky", built.sky), sun_azimuth=built.sun_azimuth,
            lighting_override=built.lighting, texel=built.texel)
        cam = shot if isinstance(shot, dict) else built.shots.get(shot or next(iter(built.shots)))
        avoid = None
        if cam is not None and cam.get("type", "dolly") == "dolly":
            ps = np.linspace(0, 1, 24)[:, None]
            eye = np.asarray(cam["from"], np.float32) * (1 - ps) + np.asarray(cam.get("to", cam["from"]), np.float32) * ps
            if eye[:, 1].min() < 6:          # 땅 가까이 있는 카메라만
                avoid = eye[:, [0, 2]]
        for spec in list(built.crowds) + list(crowds or []):
            cr = make_crowd(spec, built.areas, rng, avoid)
            if len(cr.instances) == 0:
                continue
            mdata = models.get(cr.model).data
            mvbo = ctx.buffer(mdata.tobytes())
            ivbo = ctx.buffer(cr.instances.tobytes())
            vao = ctx.vertex_array(self.inst_prog, [
                (mvbo, "3f 3f 3f 1f 1f", "in_pos", "in_norm", "in_col", "in_slot", "in_mat"),
                (ivbo, INSTANCE_FMT, *INSTANCE_ATTRS)])
            dvao = ctx.vertex_array(self.inst_depth, [
                (mvbo, "3f 12x 12x 4x 4x", "in_pos"),
                (ivbo, "3f 1f 12x 12x 1f 3f 1f/i", "i_off", "i_yaw", "i_phase", "i_vel", "i_scale")])
            prep.crowds.append((cr, vao, dvao, mvbo, ivbo))
        return prep

    def release(self, prep: Prepared) -> None:
        for obj in (prep.world_vao, prep.world_depth_vao, prep.world_vbo):
            obj.release()
        for cr, vao, dvao, mvbo, ivbo in prep.crowds:
            for obj in (vao, dvao, mvbo, ivbo):
                obj.release()

    # -------------------------------------------------------------- frame
    def lighting(self, prep: Prepared, hours: float) -> Lighting:
        from .sky import time_of_day
        light = preset(prep.sky, hours) if prep.sky in ("space", "mars") else time_of_day(hours, prep.sun_azimuth)
        for k, v in prep.lighting_override.items():
            setattr(light, k, np.asarray(v, np.float32) if isinstance(v, (list, tuple)) else v)
        return light

    def render(self, prep: Prepared, cam: dict, t: float, duration: float, hours: float,
               shake: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        ctx = self.ctx
        eye, target, fov = camera_at(cam, t, duration)
        if shake != (0.0, 0.0):
            eye = eye + np.array([shake[0], shake[1], 0], np.float32)
            target = target + np.array([shake[0], shake[1], 0], np.float32)
        proj = perspective(fov, self.w / self.h, 0.1, 2500.0)
        view = look_at(eye, target)
        vp = proj @ view
        self.last_vp = vp
        light = self.lighting(prep, hours)

        # 그림자 상자는 카메라가 보는 곳 주변으로
        fwd = (target - eye)
        fwd[1] = 0
        fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
        center = np.asarray(target, np.float32).copy() if cam.get("type") == "topdown" else eye + fwd * prep.shadow_radius * 0.6
        center[1] = 0
        R = prep.shadow_radius
        sun = light.sun_dir / np.linalg.norm(light.sun_dir)
        light_vp = ortho(-R, R, -R, R, 1.0, R * 6) @ look_at(center + sun * R * 3, center)

        shadow_on = 1.0 if sun[1] > 0.05 else 0.0
        if shadow_on:
            self.shadow_fbo.use()
            ctx.enable(ctx.DEPTH_TEST)
            ctx.clear(depth=1.0)
            self.world_depth["u_vp"].write(gl_bytes(light_vp))
            prep.world_depth_vao.render()
            self.inst_depth["u_vp"].write(gl_bytes(light_vp))
            for cr, _, dvao, _, _ in prep.crowds:
                self._crowd_uniforms(self.inst_depth, cr, t)
                dvao.render(instances=len(cr.instances))

        self.ms_fbo.use()
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.clear(*light.horizon, 1.0, depth=1.0)
        # 하늘
        ctx.disable(ctx.DEPTH_TEST)
        sp = self.sky_prog
        sp["u_inv_vp"].write(gl_bytes(np.linalg.inv(vp)))
        self._set(sp, u_cam=eye, u_zenith=light.zenith, u_horizon=light.horizon, u_sun_dir=sun,
                  u_sun_col=np.minimum(light.sun_col * 1.25, 1.0), u_sun_size=light.sun_size,
                  u_night=light.night,
                  u_planet_dir=light.planet_dir if light.planet_dir is not None else np.array([0, 1, 0], np.float32),
                  u_planet_col=light.planet_col if light.planet_col is not None else np.zeros(3, np.float32),
                  u_planet_size=light.planet_size)
        self.sky_vao.render()
        ctx.enable(ctx.DEPTH_TEST)

        self.shadow_tex.use(location=0)
        for prog in (self.world_prog, self.inst_prog):
            prog["u_vp"].write(gl_bytes(vp))
            prog["u_light_vp"].write(gl_bytes(light_vp))
            self._set(prog, u_cam=eye, u_sun_dir=sun, u_sun_col=light.sun_col, u_amb_sky=light.amb_sky,
                      u_amb_ground=light.amb_ground, u_fog_col=light.horizon, u_fog_density=light.fog_density,
                      u_time=t, u_night=light.night, u_texel=prep.texel, u_shadow_on=shadow_on)
            if prog.get("u_shadow", None) is not None:
                prog["u_shadow"].value = 0
        prep.world_vao.render()
        for cr, vao, _, _, _ in prep.crowds:
            self._crowd_uniforms(self.inst_prog, cr, t)
            vao.render(instances=len(cr.instances))

        ctx.copy_framebuffer(self.out_fbo, self.ms_fbo)
        raw = self.out_fbo.read(components=3)
        img = np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)
        return img[::-1]

    @staticmethod
    def _set(prog, **kw):
        for k, v in kw.items():
            u = prog.get(k, None)
            if u is None:
                continue
            if isinstance(v, np.ndarray):
                u.value = tuple(float(x) for x in v)
            else:
                u.value = float(v)

    def _crowd_uniforms(self, prog, cr: Crowd, t: float):
        self._set(prog, u_time=t, u_bob=cr.bob, u_bob_freq=cr.bob_freq, u_spin=cr.spin,
                  u_wrap_min=np.asarray(cr.wrap_min, np.float32), u_wrap_size=np.asarray(cr.wrap_size, np.float32))

    def project(self, points) -> list:
        """마지막으로 그린 프레임 기준, 3D 점 → 화면 픽셀 좌표 (카메라 뒤면 None)"""
        out = []
        for p in points:
            c = self.last_vp @ np.array([p[0], p[1], p[2], 1.0], np.float32)
            if c[3] <= 1e-4:
                out.append(None)
                continue
            nx, ny = c[0] / c[3], c[1] / c[3]
            out.append(((nx * 0.5 + 0.5) * self.w, (1 - (ny * 0.5 + 0.5)) * self.h))
        return out

    def close(self):
        self.ctx.release()


def scene_hours(scene_time, hud, t: float, duration: float) -> float:
    """조명 시각: scene time(고정/구간) > HUD 시계 > 정오"""
    def interp(a, b):
        ha, hb = parse_time(a), parse_time(b)
        if hb < ha:
            hb += 24
        p = min(max(t / max(duration, 1e-6), 0), 1)
        return ha + (hb - ha) * p
    if isinstance(scene_time, dict):
        return interp(scene_time["from"], scene_time.get("to", scene_time["from"]))
    if scene_time is not None:
        return parse_time(scene_time)
    if isinstance(hud, dict) and isinstance(hud.get("from"), str) and ":" in hud["from"]:
        return interp(hud["from"], hud.get("to", hud["from"]))
    return 12.0
