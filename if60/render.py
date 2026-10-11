"""장면 YAML → 9:16 도트 쇼츠 mp4.

한 프레임이 만들어지는 과정
  레이어(절차적 / 이미지 / AI 이미지 / AI 영상) 각각 저해상도로 준비
  → 카메라 이동량 × 레이어 speed 만큼 정수 픽셀 단위로 이동 (패럴랙스)
  → 흔들림·플래시·디더 페이드 효과
  → (선택) 전체 팔레트 재양자화
  → 정수배 확대 + 도트 자막 합성 → ffmpeg로 인코딩
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import chiptune, procedural, text
from .media import Encoder, read_frames, require_ffmpeg
from .pixelate import (TemporalPixelator, auto_palette, downscale, indices_to_rgb, palette_array,
                       quantize)
from .providers import ImageProvider, get_image_provider

ROOT = Path(__file__).resolve().parent.parent
MARGIN = 48  # 패럴랙스 이동을 위해 레이어를 화면보다 크게 만드는 여백(px, 저해상도 기준)

_BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


# ------------------------------------------------------------------ config
def load_styles(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config" / "styles.yaml").read_text(encoding="utf-8"))


@dataclass
class Style:
    name: str
    resolution: tuple[int, int]
    scale: int
    fps: int
    palette: np.ndarray | None
    n_colors: int
    dither: float
    hold: float
    final_quantize: bool
    text_canvas: tuple[int, int]
    caption_size: int
    title_size: int
    font: str | None = None

    @classmethod
    def from_preset(cls, name: str, overrides: dict | None = None) -> "Style":
        presets = load_styles()
        if name not in presets:
            raise ValueError(f"스타일 '{name}' 없음 (가능: {', '.join(presets)})")
        d = {**presets[name], **(overrides or {})}
        pal = d.get("palette", "auto")
        return cls(
            name=name, resolution=tuple(d["resolution"]), scale=int(d["scale"]), fps=int(d["fps"]),
            palette=None if pal == "auto" else palette_array(pal), n_colors=int(d.get("n_colors", 24)),
            dither=float(d.get("dither", 0)), hold=float(d.get("hold", 20)),
            final_quantize=bool(d.get("final_quantize", False)),
            text_canvas=tuple(d.get("text_canvas", [360, 640])),
            caption_size=int(d.get("caption_size", 22)), title_size=int(d.get("title_size", 30)),
            font=d.get("font"))

    @property
    def out_size(self) -> tuple[int, int]:
        return self.resolution[0] * self.scale, self.resolution[1] * self.scale


# ------------------------------------------------------------------ layers
def _image_layer(img: Image.Image, style: Style, size: tuple[int, int], speed: float,
                 fit: str = "cover") -> procedural.Layer:
    """일반 이미지/AI 이미지를 도트 레이어로. PNG 투명도는 0/255로 잘라 유지."""
    rgba = img.convert("RGBA")
    rgb = downscale(rgba, size)
    alpha = downscale(rgba.getchannel("A").convert("RGB"), size)[..., 0]
    pal = style.palette if style.palette is not None else auto_palette(
        Image.fromarray(rgb.astype(np.uint8)), style.n_colors)
    out = np.zeros((size[1], size[0], 4), np.uint8)
    out[..., :3] = indices_to_rgb(quantize(rgb, pal, style.dither), pal)
    out[..., 3] = np.where(alpha > 127, 255, 0)
    return procedural.Layer(out, speed=speed)


def _video_layer(path: str, style: Style, size: tuple[int, int], loop: bool,
                 speed: float) -> procedural.Layer:
    """AI 영상 클립 → 시간축 안정화된 도트 프레임 시퀀스."""
    tp = TemporalPixelator(size, style.palette, style.n_colors, style.dither, style.hold)
    frames = []
    for f in read_frames(path, style.fps):
        rgb = tp(Image.fromarray(f))
        frames.append(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]))
    if not frames:
        raise RuntimeError(f"영상에서 프레임을 읽지 못했어요: {path}")
    fps = style.fps

    def animate(_: np.ndarray, t: float) -> np.ndarray:
        i = int(t * fps)
        return frames[i % len(frames)] if loop else frames[min(i, len(frames) - 1)]

    return procedural.Layer(frames[0], speed=speed, animator=animate)


def build_layer(spec: dict, style: Style, provider: ImageProvider, base: Path) -> procedural.Layer:
    w, h = style.resolution
    size = (w + MARGIN * 2, h + MARGIN * 2)
    src = spec.get("source", "procedural")
    speed = float(spec.get("speed", 1.0))
    if src == "procedural":
        params = dict(spec.get("params", {}))
        if "speed" in spec:
            params["speed"] = speed
        return procedural.make(spec["kind"], size[0], size[1], **params)
    if src == "image":
        return _image_layer(Image.open(base / spec["path"]), style, size, speed)
    if src == "ai":
        prompt = f"{spec['prompt']}, {spec.get('style_suffix', 'pixel art, limited palette, crisp edges, vertical 9:16 composition')}"
        return _image_layer(provider.cached(prompt, size), style, size, speed)
    if src == "video":
        # 영상은 패럴랙스 여백 없이 화면 크기로 (카메라 이동은 보통 영상 자체에 들어있음)
        layer = _video_layer(str(base / spec["path"]), style, size, bool(spec.get("loop", True)), speed)
        return layer
    raise ValueError(f"알 수 없는 레이어 source: {src}")


# ------------------------------------------------------------------ scene
@dataclass
class Scene:
    duration: float
    layers: list[procedural.Layer]
    caption: str = ""
    title: str = ""
    camera: dict = field(default_factory=dict)
    effects: list = field(default_factory=list)
    typewriter: bool = False
    caption_mode: str = "chunk"           # chunk | full | typewriter
    captions: list[str] | None = None     # 직접 나눈 자막 덩어리 (있으면 caption보다 우선)
    hud: object = None                    # "+1년" 같은 고정 문구 또는 {from, to, format, step}
    # --- 3D 복셀 장면 (world가 있으면 layers 대신 3D로 렌더링)
    world: dict | None = None             # {template: plaza, seed: 3, params: {...}, sky: auto|mars|space}
    crowds: list = field(default_factory=list)
    shot: object = None                   # 템플릿 샷 이름 또는 카메라 dict
    time: object = None                   # 조명 시각 "05:00" 또는 {from, to} (없으면 HUD 시계를 따름)
    seed: int = 0


# ------------------------------------------------------------------ HUD
def _parse_clock(s) -> int | None:
    if isinstance(s, str) and ":" in s:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    return None


def hud_label(hud, t: float, duration: float) -> str | None:
    """시계('05:00'→'06:30')나 숫자(1670→0) 카운터를 장면 진행에 맞춰 보간한다."""
    if hud is None or hud == "":
        return None
    if not isinstance(hud, dict):
        return str(hud)
    a, b = hud.get("from"), hud.get("to", hud.get("from"))
    p = min(1.0, max(0.0, t / max(duration, 1e-6)))
    if hud.get("ease", True):
        p = _ease(p)
    ca, cb = _parse_clock(a), _parse_clock(b)
    if ca is not None and cb is not None:
        if cb < ca:  # 자정을 넘어가는 경우 (23:30 → 00:30)
            cb += 24 * 60
        step = int(hud.get("step", 10))  # 시계는 10분 단위로 '딸깍' 넘어가게
        m = int(round((ca + (cb - ca) * p) / step) * step) % (24 * 60)
        return hud.get("format", "{}").format(f"{m // 60:02d}:{m % 60:02d}")
    v = float(a) + (float(b) - float(a)) * p
    if "step" in hud:
        v = round(v / float(hud["step"])) * float(hud["step"])
    return hud.get("format", "{:,.0f}").format(v)


def caption_at(scene: "Scene", t: float) -> str:
    """현재 시각에 보여줄 자막. chunk 모드는 글자 수에 비례해 덩어리를 순서대로 보여준다."""
    if scene.captions:
        chunks = scene.captions
    elif scene.caption_mode == "chunk" and scene.caption and not scene.typewriter:
        chunks = text.chunk_caption(scene.caption)
    else:
        return scene.caption
    lead, tail = 0.15, 0.25
    span = max(0.1, scene.duration - lead - tail)
    weights = np.array([len(c) + 3 for c in chunks], float)
    edges = lead + np.cumsum(weights) / weights.sum() * span
    i = int(np.searchsorted(edges, t, side="right"))
    return chunks[min(i, len(chunks) - 1)]


def _ease(x: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))


def _effect_params(e) -> tuple[str, dict]:
    if isinstance(e, str):
        return e, {}
    (name, params), = e.items()
    return name, params or {}


def compose_frame(scene: Scene, t: float, style: Style) -> tuple[np.ndarray, list[dict]]:
    """저해상도 RGB 프레임 + 이 프레임에서 발생한 효과음 트리거."""
    w, h = style.resolution
    canvas = np.zeros((h + MARGIN * 2, w + MARGIN * 2, 4), np.uint8)
    p = _ease(t / scene.duration)
    pan = scene.camera.get("pan", [0, 0])
    cam_x, cam_y = pan[0] * p, pan[1] * p

    shake_x = shake_y = 0
    for e in scene.effects:
        name, prm = _effect_params(e)
        if name == "shake":
            at, length, amp = prm.get("at", 0.0), prm.get("length", 1.2), prm.get("amp", 4)
            if at <= t < at + length:
                k = 1 - (t - at) / length
                shake_x = int(round(math.sin(t * 61) * amp * k))
                shake_y = int(round(math.cos(t * 47) * amp * k))

    for layer in scene.layers:
        img = layer.frame(t)
        ox = int(round(-cam_x * layer.speed)) + shake_x
        oy = int(round(-cam_y * layer.speed)) + shake_y
        procedural.paste(canvas, img, ox, oy)

    frame = canvas[MARGIN:MARGIN + h, MARGIN:MARGIN + w, :3].copy()

    for e in scene.effects:
        name, prm = _effect_params(e)
        if name == "flash":
            at = prm.get("at", 0.0)
            if at <= t < at + prm.get("length", 0.25):
                frame[:] = 255
        if name == "tint":  # 예: 붉은 경보 — 저해상도에서 블렌딩 후 아래에서 팔레트 정리
            at, length = prm.get("at", 0.0), prm.get("length", scene.duration)
            if at <= t < at + length and int(t * 4) % 2 == 0:
                c = np.array(text_color(prm.get("color", "#c8384a")), np.float32)
                frame = (frame * 0.6 + c * 0.4).astype(np.uint8)

    # 디더 페이드 인/아웃 (팔레트 밖 색을 만들지 않는 레트로 방식)
    fade_in = scene.camera.get("fade_in", 0.35)
    fade_out = scene.camera.get("fade_out", 0.35)
    level = 1.0
    if fade_in and t < fade_in:
        level = t / fade_in
    if fade_out and t > scene.duration - fade_out:
        level = min(level, (scene.duration - t) / fade_out)
    if level < 1.0:
        mask = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w] >= level
        frame[mask] = 0
    return frame, []


def text_color(hexstr: str) -> tuple[int, int, int]:
    from .pixelate import hex_to_rgb
    return hex_to_rgb(hexstr)


# ------------------------------------------------------------------ project
def load_project(path: str, style_override: str | None = None,
                 length: float | None = None, provider_name: str | None = None):
    path_p = Path(path)
    spec = yaml.safe_load(path_p.read_text(encoding="utf-8"))
    style = Style.from_preset(style_override or spec.get("style", "cinematic_pixel"),
                              spec.get("style_overrides"))
    provider = get_image_provider(provider_name or spec.get("provider"))
    scenes_spec = spec["scenes"]
    total = sum(float(s["duration"]) for s in scenes_spec)
    ratio = (length / total) if length else 1.0  # 30/45/60초로 맞출 때 장면 길이를 비례 조정
    scenes = []
    for i, s in enumerate(scenes_spec):
        print(f"  · 장면 {i + 1}/{len(scenes_spec)} 레이어 준비")
        layers = [] if s.get("world") else [build_layer(l, style, provider, path_p.parent)
                                             for l in s.get("layers", [])]
        scenes.append(Scene(
            duration=float(s["duration"]) * ratio, layers=layers, caption=s.get("caption", ""),
            title=s.get("title", ""), camera=s.get("camera", {}) or {},
            effects=s.get("effects", []) or [], typewriter=bool(s.get("typewriter", False)),
            caption_mode=s.get("caption_mode", spec.get("caption_mode", "chunk")),
            captions=s.get("captions"), hud=s.get("hud"),
            world=s.get("world"), crowds=s.get("crowds", []) or [], shot=s.get("shot"),
            time=s.get("time"), seed=int(s.get("seed", (s.get("world") or {}).get("seed", i)))))
    return spec, style, scenes


def _text_layer(scene: Scene, t: float, style: Style, font: str) -> np.ndarray | None:
    tw, th = style.text_canvas
    out = None
    if scene.title:
        out = text.render_text((tw, th), scene.title, font_path=font, size=style.title_size,
                               y=0.2, color=(255, 236, 160), box=(16, 12, 40))
    def merge(layer: np.ndarray) -> None:
        nonlocal out
        if out is None:
            out = layer
        else:
            m = layer[..., 3] == 255
            out[m] = layer[m]

    cap_text = caption_at(scene, t) if (scene.caption or scene.captions) else ""
    if cap_text:
        vis = None
        if scene.typewriter or scene.caption_mode == "typewriter":
            vis = int(len(cap_text) * min(1.0, t / max(0.6, scene.duration * 0.45))) + 1
        size = style.caption_size if (scene.typewriter or scene.caption_mode != "chunk") \
            else int(style.caption_size * 1.25)  # 덩어리 자막은 짧으니 더 크게
        merge(text.render_text((tw, th), cap_text, font_path=font, size=size, y=0.74,
                               visible_chars=vis))
    label = hud_label(scene.hud, t, scene.duration)
    if label:
        merge(text.render_badge((tw, th), label, font_path=font,
                                size=int(style.caption_size * 0.8)))
    return out


def render(project_path: str, out_path: str, *, style_override: str | None = None,
           length: float | None = None, provider: str | None = None, preview: bool = False,
           stills_dir: str | None = None, fps: int | None = None) -> str:
    require_ffmpeg()
    print("▶ 프로젝트 불러오는 중")
    spec, style, scenes = load_project(project_path, style_override, length, provider)
    has_3d = any(sc.world for sc in scenes)
    if fps:
        style.fps = int(fps)
    elif preview and has_3d:
        style.fps = min(style.fps, 15)   # 3D 미리보기는 프레임 수를 줄여 빠르게
    font = text.find_font(style.font)
    w, h = style.resolution
    out_scale = max(1, style.scale // 3) if preview else style.scale
    ow, oh = w * out_scale, h * out_scale
    tw, th = style.text_canvas
    text_scale = ow // tw if ow % tw == 0 else None
    total = sum(s.duration for s in scenes)

    # ---- 오디오: 칩튠 BGM + 장면 효과음
    music = spec.get("music", {}) or {}
    audio = chiptune.bgm(total, mood=music.get("mood", "wonder"), bpm=int(music.get("bpm", 112)),
                         seed=int(music.get("seed", 0)), volume=float(music.get("volume", 0.22)))
    t0 = 0.0
    for sc in scenes:
        for e in sc.effects:
            name, prm = _effect_params(e)
            if name == "shake":
                chiptune.mix(audio, chiptune.sfx_boom(prm.get("length", 1.2)), t0 + prm.get("at", 0))
            if name == "flash":
                chiptune.mix(audio, chiptune.sfx_blip(1320, 0.12), t0 + prm.get("at", 0))
        t0 += sc.duration

    tmp = tempfile.mkdtemp(prefix="if60_")
    wav = str(Path(tmp) / "bgm.wav")
    chiptune.write_wav(wav, audio)

    # ---- 영상
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(out_path, ow, oh, style.fps, audio=wav)
    vr = None
    if has_3d:
        from .voxel.renderer import VoxelRenderer
        vr = VoxelRenderer(ow, oh, samples=2 if preview else 4)
    if stills_dir:
        Path(stills_dir).mkdir(parents=True, exist_ok=True)
    final_pal = style.palette if (style.final_quantize and style.palette is not None) else None
    n_frames_total = 0
    for si, sc in enumerate(scenes):
        n = max(1, int(round(sc.duration * style.fps)))
        print(f"▶ 장면 {si + 1}/{len(scenes)} 렌더링 ({n}프레임{', 3D' if sc.world else ''})")
        prep = vr.prepare(sc.world, sc.crowds, seed=sc.seed, shot=sc.shot) if sc.world else None
        cam = None
        if prep is not None:
            cam = sc.shot if isinstance(sc.shot, dict) else prep.shots.get(sc.shot or next(iter(prep.shots)))
            if cam is None:
                raise ValueError(f"장면 {si + 1}: 샷 '{sc.shot}' 없음 (가능: {', '.join(prep.shots)})")
        for fi in range(n):
            t = fi / style.fps
            if prep is not None:
                big = render_3d_frame(vr, prep, cam, sc, t)
            else:
                frame, _ = compose_frame(sc, t, style)
                if final_pal is not None:
                    frame = indices_to_rgb(quantize(frame.astype(np.float32), final_pal), final_pal)
                big = frame.repeat(out_scale, 0).repeat(out_scale, 1)
            tl = _text_layer(sc, t, style, font)
            if tl is not None:
                if text_scale:
                    tl_big = tl.repeat(text_scale, 0).repeat(text_scale, 1)
                else:
                    tl_big = np.asarray(Image.fromarray(tl).resize((ow, oh), Image.Resampling.NEAREST))
                m = tl_big[..., 3] == 255
                big[m] = tl_big[..., :3][m]
            enc.write(big)
            if stills_dir and fi == n // 2:
                Image.fromarray(big).save(Path(stills_dir) / f"scene_{si + 1:02d}.png")
            n_frames_total += 1
        if prep is not None:
            vr.release(prep)
    enc.close()
    if vr is not None:
        vr.close()
    print(f"✔ 완료: {out_path}  ({total:.1f}초, {n_frames_total}프레임, {ow}x{oh})")
    return out_path


def render_3d_frame(vr, prep, cam: dict, sc: Scene, t: float) -> np.ndarray:
    """3D 장면 한 프레임. 흔들림은 카메라를 실제로 흔들고, 플래시·틴트·페이드는 화면에 입힌다."""
    from .voxel.renderer import scene_hours
    shake = (0.0, 0.0)
    for e in sc.effects:
        name, prm = _effect_params(e)
        if name == "shake":
            at, length, amp = prm.get("at", 0.0), prm.get("length", 1.2), prm.get("amp", 4)
            if at <= t < at + length:
                k = (1 - (t - at) / length) * amp * 0.06
                shake = (math.sin(t * 61) * k, math.cos(t * 47) * k)
    hours = scene_hours(sc.time, sc.hud, t, sc.duration)
    img = vr.render(prep, cam, t, sc.duration, hours, shake).astype(np.float32)
    for e in sc.effects:
        name, prm = _effect_params(e)
        at, length = prm.get("at", 0.0), prm.get("length", 0.25 if name == "flash" else sc.duration)
        if name == "flash" and at <= t < at + length:
            img[:] = 255
        if name == "tint" and at <= t < at + length and int(t * 4) % 2 == 0:
            img = img * 0.6 + np.array(text_color(prm.get("color", "#c8384a")), np.float32) * 0.4
    fade_in, fade_out = sc.camera.get("fade_in", 0.0), sc.camera.get("fade_out", 0.0)
    level = 1.0
    if fade_in and t < fade_in:
        level = t / fade_in
    if fade_out and t > sc.duration - fade_out:
        level = min(level, (sc.duration - t) / fade_out)
    if level < 1.0:
        img *= level
    return np.clip(img, 0, 255).astype(np.uint8)


def pixelate_file(src: str, out: str, style_name: str = "cinematic_pixel",
                  preview: bool = False) -> str:
    """AI 이미지/영상 한 개를 바로 도트로 변환 (자막·패럴랙스 없이 후처리만)."""
    style = Style.from_preset(style_name)
    ext = Path(src).suffix.lower()
    w, h = style.resolution
    scale = max(1, style.scale // 3) if preview else style.scale
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        layer = _image_layer(Image.open(src), style, (w, h), 0)
        Image.fromarray(layer.img[..., :3].repeat(scale, 0).repeat(scale, 1)).save(out)
        print(f"✔ 이미지 변환 완료: {out}")
        return out
    require_ffmpeg()
    tp = TemporalPixelator((w, h), style.palette, style.n_colors, style.dither, style.hold)
    tmp = tempfile.mkdtemp(prefix="if60_")
    audio = str(Path(tmp) / "a.wav")
    import subprocess
    streams = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                              "stream=index", "-of", "csv=p=0", src],
                             capture_output=True, text=True).stdout.strip()
    has_audio = bool(streams) and subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-vn", audio]).returncode == 0
    enc = Encoder(out, w * scale, h * scale, style.fps, audio=audio if has_audio else None)
    n = 0
    for f in read_frames(src, style.fps):
        enc.write(tp(Image.fromarray(f)).repeat(scale, 0).repeat(scale, 1))
        n += 1
    enc.close()
    print(f"✔ 영상 변환 완료: {out} ({n}프레임 @ {style.fps}fps)")
    return out
