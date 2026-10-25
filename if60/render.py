"""장면 YAML → 9:16 도트 쇼츠 mp4.

한 프레임이 만들어지는 과정
  레이어(절차적 / 이미지 / AI 이미지 / AI 영상) 각각 저해상도로 준비
  → 카메라 이동량 × 레이어 speed 만큼 정수 픽셀 단위로 이동 (패럴랙스)
  → 흔들림·플래시·디더 페이드 효과
  → (선택) 전체 팔레트 재양자화
  → 정수배 확대 + 도트 자막 합성 → ffmpeg로 인코딩
"""
from __future__ import annotations

import re

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import overlay, procedural, text
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
    props: list = field(default_factory=list)    # 장면에 놓는 사물 (model 또는 box/sphere/cylinder)
    labels: list = field(default_factory=list)   # 3D 위치에 붙는 화면 라벨 {text, at: [x,y,z], from, to}
    # --- 방송형 연출
    vfx: list = field(default_factory=list)      # 충격파·불덩이·버섯구름·불티 {type, at, from, to, radius...}
    panel: dict | None = None                    # 데이터 대시보드 {title, value, stats, bars, status}
    big_text: list = field(default_factory=list) # 빛나는 큰 글자 {text, from, to, pos | at, color, size}
    markers: list = field(default_factory=list)  # 빛나는 표적 링 {at: [x,y,z], from, to, text, color}
    card: dict | None = None                     # 마무리 카드 {question, options, cta}
    chyron: list = field(default_factory=list)   # 뉴스 자막띠 {tag: 속보, text, from, to}
    beat: float = 0.0                            # 대사 뒤 '정적'(초) — 펀치라인 뒤 웃음 틈
    weather: object = None                       # rain | snow | {type, color, density, wind, puddles}
    alert: dict | None = None                    # 휴대폰 재난문자 팝업 {title, text, from, to}
    credits: dict | None = None                  # 엔딩 크레딧 {title, heading: 카메오 출연, cast: [{name, color, color2}]}
    look: str = "voxel"                          # voxel | smooth | cinematic
    readout: dict | None = None                  # 시네마틱 미니멀 수치 {status, label, value, note, right: [...]}
    caption_style: str = "pixel"                 # pixel(도트 자막) | cinema(명조 영화 자막)
    # --- 내레이션 (TTS)
    narration: str | None = None                 # 읽을 문장 (없으면 caption). 숫자·기호를 읽는 그대로 쓸 때
    audio_start: float | None = None             # 이 장면 안에서 목소리가 시작하는 시각
    audio_len: float | None = None               # 목소리 길이 — 자막 덩어리가 이 구간에 맞춰 나온다
    bloom: float = 0.45                          # 빛 번짐 세기 (0이면 끔)


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
    elif scene.caption_style == "cinema" and scene.caption:    # 영화 자막: 문장 단위로 1~2줄
        chunks = [c.strip() for c in re.split(r"(?<=[.?!])\s+", scene.caption) if c.strip()]
    elif scene.caption_mode == "chunk" and scene.caption and not scene.typewriter:
        chunks = text.chunk_caption(scene.caption)
    else:
        return scene.caption
    if scene.audio_len:                     # 목소리가 있으면 말하는 구간에 맞춰 자막을 넘긴다
        lead, span = scene.audio_start or 0.0, scene.audio_len
    else:
        lead, tail = 0.15, 0.25
        span = max(0.1, scene.duration - lead - tail)
    weights = np.array([len(c) + 3 for c in chunks], float)
    edges = lead + np.cumsum(weights) / weights.sum() * span
    i = int(np.searchsorted(edges, t, side="right"))
    return chunks[min(i, len(chunks) - 1)]


def _ease(x: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(x, 0.0), 1.0))


EFFECTS = ("shake", "flash", "tint", "whiteout", "rewind", "punch")


def normalize_effects(raw) -> list[tuple[str, dict]]:
    """효과 목록을 (이름, 옵션) 쌍으로 통일한다. AI가 쓰는 여러 형식을 모두 받아준다.
      - "flash"
      - {shake: {at: 1, amp: 4}}
      - {shake: {...}, flash: {...}}           (한 항목에 여러 효과)
      - {type: shake, at: 1, amp: 4}           (type/name/effect 키)
    알 수 없는 효과는 경고만 하고 건너뛴다."""
    out: list[tuple[str, dict]] = []
    items = raw if isinstance(raw, list) else ([raw] if raw else [])
    for e in items:
        if isinstance(e, (tuple, list)) and len(e) == 2 and isinstance(e[0], str):
            out.append((e[0], dict(e[1] or {})))
        elif isinstance(e, str):
            out.append((e, {}))
        elif isinstance(e, dict):
            kind = e.get("type") or e.get("name") or e.get("effect")
            if isinstance(kind, str):
                out.append((kind, {k: v for k, v in e.items() if k not in ("type", "name", "effect")}))
                continue
            for k, v in e.items():
                out.append((str(k), dict(v) if isinstance(v, dict) else {}))
    known = [(n, p) for n, p in out if n in EFFECTS]
    for n, _ in out:
        if n not in EFFECTS:
            print(f"  ⚠ 알 수 없는 효과 '{n}'는 건너뜀 (가능: {', '.join(EFFECTS)})")
    return known


def _effect_params(e) -> tuple[str, dict]:
    if isinstance(e, tuple):
        return e
    norm = normalize_effects([e])
    return norm[0] if norm else ("none", {})


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
                 length: float | None = None, provider_name: str | None = None, quiet: bool = False):
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
        if not quiet and not s.get("world"):
            print(f"  · 장면 {i + 1}/{len(scenes_spec)} 레이어 준비")
        layers = [] if s.get("world") else [build_layer(l, style, provider, path_p.parent)
                                             for l in s.get("layers", [])]
        scenes.append(Scene(
            duration=float(s["duration"]) * ratio, layers=layers, caption=s.get("caption", ""),
            title=s.get("title", ""), camera=s.get("camera", {}) or {},
            effects=normalize_effects(s.get("effects")), typewriter=bool(s.get("typewriter", False)),
            caption_mode=s.get("caption_mode", spec.get("caption_mode", "chunk")),
            captions=s.get("captions"), hud=s.get("hud"),
            world=s.get("world"), crowds=s.get("crowds", []) or [], shot=s.get("shot"),
            time=s.get("time"), seed=int(s.get("seed", (s.get("world") or {}).get("seed", i))),
            props=s.get("props", []) or [], labels=s.get("labels", []) or [],
            vfx=s.get("vfx", []) or [], panel=s.get("panel"), big_text=_as_list(s.get("big_text")),
            markers=_as_list(s.get("markers")), card=s.get("card"), narration=s.get("narration"),
            chyron=_as_list(s.get("chyron")), beat=float(s.get("beat", 0) or 0),
            weather=s.get("weather"), alert=s.get("alert"), credits=s.get("credits"),
            look=str(s.get("look", spec.get("look", "voxel"))),
            readout=s.get("readout"), caption_style=caption_style_of(spec, s),
            bloom=float(s.get("bloom", spec.get("bloom", {"smooth": 0.7, "cinematic": 0.25}.get(spec.get("look"), 0.45))))))
    return spec, style, scenes


def caption_style_of(spec: dict, s: dict | None = None) -> str:
    """시네마틱 화면이면 자막도 기본으로 명조 영화 자막"""
    s = s or {}
    st = s.get("caption_style", spec.get("caption_style"))
    if st:
        return str(st)
    return "cinema" if s.get("look", spec.get("look")) == "cinematic" else "pixel"


def _as_list(v) -> list:
    if not v:
        return []
    return [v] if isinstance(v, dict) else list(v)


def _text_layer(scene: Scene, t: float, style: Style, font: str, layout: bool = False) -> np.ndarray | None:
    """도트 자막·장면 제목·HUD 배지. layout(방송형)이면 상단 띠를 피해 아래로 내린다."""
    tw, th = style.text_canvas
    out = None
    if scene.title:
        out = text.render_text((tw, th), scene.title, font_path=font, size=style.title_size,
                               y=0.3 if layout else 0.2, color=(255, 236, 160), box=(16, 12, 40))
    def merge(layer: np.ndarray) -> None:
        nonlocal out
        if out is None:
            out = layer
        else:
            m = layer[..., 3] == 255
            out[m] = layer[m]

    cap_text = caption_at(scene, t) if (scene.caption or scene.captions) and not scene.card and not scene.credits else ""
    if cap_text and scene.caption_style != "cinema":
        vis = None
        if scene.typewriter or scene.caption_mode == "typewriter":
            vis = int(len(cap_text) * min(1.0, t / max(0.6, scene.duration * 0.45))) + 1
        size = style.caption_size if (scene.typewriter or scene.caption_mode != "chunk") \
            else int(style.caption_size * 1.25)  # 덩어리 자막은 짧으니 더 크게
        merge(text.render_text((tw, th), cap_text, font_path=font, size=size, y=0.74,
                               visible_chars=vis))
    label = hud_label(scene.hud, t, scene.duration)
    if label:
        merge(text.render_badge((tw, th), label, font_path=font, y=0.185 if layout else 0.075,
                                size=int(style.caption_size * 0.8)))
    return out


class Composer:
    """장면 한 프레임을 최종 화면으로: 3D/2D 바탕 → 빛 번짐 → 표적·큰 글자·대시보드·카드 → 자막 → 전환 효과 → 방송 레이아웃.
    렌더링·시험 렌더·감독 검수 시트가 모두 이 순서를 그대로 쓴다."""

    def __init__(self, spec: dict, style: Style, font: str, size: tuple[int, int], vr=None):
        self.spec, self.style, self.font = spec, style, font
        self.w, self.h = size
        self.u = self.w / 1080
        self.vr = vr
        lay = spec.get("layout")
        self.layout = overlay.make_layout(lay, spec.get("title", ""), font, size)
        self.broadcast = isinstance(self.layout, overlay.Layout)
        self._first = None
        self.prev: np.ndarray | None = None
        self._card_bg: dict = {}
        self.serif, self.light = text.find_serif(), text.find_light()

    def viewport(self) -> tuple[int, int]:
        return self.layout.viewport if self.layout else (0, self.h)

    def frame(self, sc: Scene, t: float, prep=None, cam=None) -> np.ndarray:
        W, H = self.w, self.h
        if prep is not None:
            big = render_3d_frame(self.vr, prep, cam, sc, t, self.font)
        elif sc.credits is not None and not sc.layers:
            big = np.zeros((H, W, 3), np.uint8)
            overlay.draw_credits(big, sc.credits, t, sc.duration, self.font, self.u)
        elif sc.card is not None and not sc.layers:
            key = id(sc)
            if key not in self._card_bg:
                self._card_bg[key] = (overlay.darken(self.prev, 0.35, blur=True) if self.prev is not None
                                      else overlay.gradient_bg(W, H))
            big = self._card_bg[key].copy()
        else:
            small, _ = compose_frame(sc, t, self.style)
            fp = self.style.palette if (self.style.final_quantize and self.style.palette is not None) else None
            if fp is not None:
                small = indices_to_rgb(quantize(small.astype(np.float32), fp), fp)
            sh, sw = small.shape[:2]
            if W % sw == 0 and H % sh == 0 and W // sw == H // sh:
                big = small.repeat(W // sw, 0).repeat(W // sw, 1)
            else:
                big = np.asarray(Image.fromarray(small).resize((W, H), Image.Resampling.NEAREST)).copy()
        big = np.ascontiguousarray(big)
        for name, prm in sc.effects:
            if name == "punch":
                big = overlay.punch_zoom(big, prm, t)
        top, bottom = self.viewport()
        if prep is not None:
            for mk in sc.markers:
                pt = self.vr.project([mk["at"]])[0] if mk.get("at") is not None else None
                overlay.draw_marker(big, mk, t, sc.duration, pt, self.font, self.u)
        for bt in sc.big_text:
            pt = self.vr.project([bt["at"]])[0] if (prep is not None and bt.get("at") is not None) else None
            overlay.draw_big_text(big, bt, t, sc.duration, self.font, self.u, pt)
        for ch in sc.chyron:
            overlay.draw_chyron(big, ch, t, sc.duration, self.font, self.u,
                                y_frac=(bottom - int(64 * self.u) - int(14 * self.u)) / self.h)
        if sc.panel:
            overlay.draw_panel(big, sc.panel, t, sc.duration, self.font, top + int(28 * self.u), self.u)
        if sc.alert:
            overlay.draw_alert(big, sc.alert, t, sc.duration, self.font, self.u,
                               top if self.broadcast else int(self.h * 0.06))
        if sc.card is not None:
            if prep is not None:
                big = overlay.darken(big, 0.45)
            overlay.draw_card(big, sc.card, t, sc.duration, self.font, self.u, (top, bottom))
        if sc.readout:
            overlay.draw_readout(big, sc.readout, t, sc.duration, self.serif, self.light, self.u,
                                 top=float(sc.readout.get("top", 0.165)))
        if sc.caption_style == "cinema" and (sc.caption or sc.captions) and not sc.card and not sc.credits:
            cap = caption_at(sc, t)
            alpha = min(1.0, t / 0.18, max(0.0, sc.duration - t) / 0.12)
            overlay.draw_cinema_caption(big, cap, self.serif, self.u, alpha)
        tl = _text_layer(sc, t, self.style, self.font, layout=self.broadcast)
        if tl is not None:
            th, tw = tl.shape[:2]
            if W % tw == 0 and H % th == 0:
                k = W // tw
                tl_big = tl.repeat(k, 0).repeat(k, 1)
            else:
                tl_big = np.asarray(Image.fromarray(tl).resize((W, H), Image.Resampling.NEAREST))
            m = tl_big[..., 3] == 255
            big[m] = tl_big[..., :3][m]
        for name, prm in sc.effects:
            if name == "whiteout":
                big = overlay.whiteout(big, overlay.whiteout_level(prm, t))
            elif name == "rewind":
                big = overlay.rewind(big, t, prm, self.font, self.u, top)
        if self._first is None:
            self._first = id(sc)
        if self.broadcast:
            big = self.layout.apply(big, t)
        elif self.layout is not None:
            big = self.layout.apply(big, t, first=id(sc) == self._first)
        self.prev = big
        return big


# ------------------------------------------------------------------ 내레이션 싱크
LEAD, TAIL, GAP_MIN = 0.2, 0.35, 1.2


def scale_times(sc: Scene, k: float) -> None:
    """장면 길이가 k배가 될 때 연출 시각(효과·VFX·글자·표적·라벨·대시보드·자라나기)도 k배로 옮긴다."""
    if abs(k - 1) < 1e-6:
        return

    def mul(d: dict, keys=("at", "length", "from", "to", "until")) -> None:
        for key in keys:
            if isinstance(d.get(key), (int, float)) and not isinstance(d.get(key), bool):
                d[key] = d[key] * k

    for _, prm in sc.effects:
        mul(prm)
    for v in sc.vfx:
        mul(v, ("from", "to"))
    for item in sc.big_text + sc.markers + sc.labels + sc.chyron + ([sc.alert] if isinstance(sc.alert, dict) else []):
        mul(item, ("from", "to"))
    for g in [p.get("grow") for p in sc.props] + [c.get("grow") for c in sc.crowds]:
        if isinstance(g, dict):
            mul(g, ("from", "to", "until"))
    if isinstance(sc.readout, dict):
        vals = [sc.readout.get("value")] + [r.get("value") for r in sc.readout.get("right") or [] if isinstance(r, dict)]
        for v in vals:
            if isinstance(v, dict):
                mul(v, ("at", "length"))
    if isinstance(sc.panel, dict):
        if isinstance(sc.panel.get("value"), dict):
            mul(sc.panel["value"], ("at", "length"))
        for b in sc.panel.get("bars") or []:
            if isinstance(b, dict):
                mul(b, ("at", "length"))


def narration_of(sc: Scene) -> str:
    if sc.narration is not None:
        return str(sc.narration)
    if sc.card and not sc.caption:
        return str(sc.card.get("question", ""))
    return sc.caption or " ".join(sc.captions or [])


def apply_voice(scenes: list[Scene], voice, log=print) -> np.ndarray:
    """장면마다 목소리를 만들고 장면 길이를 목소리에 맞춘다 → 영상 전체 길이의 음성 트랙(48kHz)."""
    from . import tts
    texts = [narration_of(sc).strip() for sc in scenes]
    clips = []
    for i, (sc, txt) in enumerate(zip(scenes, texts)):
        if not txt:
            clips.append(None)
            continue
        log(f"  · 목소리 {i + 1}/{len(scenes)}: {txt[:24]}{'…' if len(txt) > 24 else ''}")
        ctx = {"prev": texts[i - 1] if i else "", "next": texts[i + 1] if i + 1 < len(texts) else ""}
        clips.append(tts.synthesize(txt, voice, ctx))
    for i, (sc, clip) in enumerate(zip(scenes, clips)):
        if clip is None or len(clip) == 0:
            continue
        sec = len(clip) / tts.SR
        lead = 0.08 if i == 0 else LEAD
        need = lead + sec + TAIL + sc.beat
        if sc.card:                                   # 선택지가 다 나올 시간은 남긴다
            need = max(need, 1.6 + 0.35 * len(sc.card.get("options") or []))
        new = max(need, GAP_MIN)
        scale_times(sc, new / sc.duration)
        sc.duration = new
        sc.audio_start, sc.audio_len = lead, sec
    total = sum(sc.duration for sc in scenes)
    track = np.zeros(int(total * tts.SR) + tts.SR, np.float32)
    t0 = 0.0
    for sc, clip in zip(scenes, clips):
        if clip is not None and len(clip):
            a = int((t0 + sc.audio_start) * tts.SR)
            track[a:a + len(clip)] += clip[:len(track) - a]
        t0 += sc.duration
    return track[:int(total * tts.SR)]


def render(project_path: str, out_path: str, *, style_override: str | None = None,
           length: float | None = None, provider: str | None = None, preview: bool = False,
           stills_dir: str | None = None, fps: int | None = None, voice=None, narrate: bool = True) -> str:
    """장면 YAML → mp4. 내레이션을 켜면 목소리를 먼저 만들고 그 길이에 맞춰 장면·자막을 맞춘다.
    목소리만 따로 담은 _voice.wav도 함께 저장한다 (편집 앱에서 쓰기 좋게)."""
    require_ffmpeg()
    print("▶ 프로젝트 불러오는 중")
    spec, style, scenes = load_project(project_path, style_override, length, provider)
    audio_path = None
    v = None
    if narrate and spec.get("voice", True) is not False:
        from . import tts
        chosen = voice or (spec.get("voice") if isinstance(spec.get("voice"), (str, dict)) else None)
        v = tts.parse_voice(chosen or tts.default_voice())
        print(f"▶ 내레이션 생성 ({v.label}) — 목소리 길이에 맞춰 장면·자막 타이밍을 맞춥니다")
        track = apply_voice(scenes, v)
        audio_path = str(Path(out_path).with_name(Path(out_path).stem + "_voice.wav"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audio_path).write_bytes(tts.encode_wav(track, tts.SR))
    has_3d = any(sc.world for sc in scenes)
    if fps:
        style.fps = int(fps)
    elif preview and has_3d:
        style.fps = min(style.fps, 15)   # 3D 미리보기는 프레임 수를 줄여 빠르게
    font = text.find_font(style.font)
    w, h = style.resolution
    out_scale = max(1, style.scale // 3) if preview else style.scale
    ow, oh = w * out_scale, h * out_scale
    total = sum(s.duration for s in scenes)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    enc = Encoder(out_path, ow, oh, style.fps, audio=audio_path)
    vr = None
    if has_3d:
        from .voxel.renderer import VoxelRenderer
        vr = VoxelRenderer(ow, oh, samples=2 if preview else 4)
    if stills_dir:
        Path(stills_dir).mkdir(parents=True, exist_ok=True)
    comp = Composer(spec, style, font, (ow, oh), vr)
    n_frames_total = 0
    for si, sc in enumerate(scenes):
        n = max(1, int(round(sc.duration * style.fps)))
        print(f"▶ 장면 {si + 1}/{len(scenes)} 렌더링 ({n}프레임{', 3D' if sc.world else ''})")
        prep, cam = prepare_scene(vr, sc, si)
        for fi in range(n):
            t = fi / style.fps
            big = comp.frame(sc, t, prep, cam)
            enc.write(big)
            if stills_dir and fi == n // 2:
                Image.fromarray(big).save(Path(stills_dir) / f"scene_{si + 1:02d}.png")
            n_frames_total += 1
        if prep is not None:
            vr.release(prep)
    enc.close()
    if vr is not None:
        vr.close()
    print(f"✔ 완료: {out_path}  ({total:.1f}초, {n_frames_total}프레임, {ow}x{oh}, "
          f"{'목소리 ' + v.label if audio_path else '무음'})")
    if audio_path:
        print(f"  · 목소리만 따로: {audio_path}")
        from . import tts
        credit = tts.credit_text(v)
        if credit:
            cpath = Path(out_path).with_name(Path(out_path).stem + "_출처표기.txt")
            cpath.write_text(credit + "\n", encoding="utf-8")
            why = ("타입캐스트 무료 플랜은 출처 표기가 필수예요" if v.provider == "typecast"
                   else "AI 음성임을 밝혀야 해요 (Supertonic 라이선스)")
            print(f"  · {why}. 아래를 게시물 설명란에 붙여넣으세요 (인스타 'AI 정보' 표시도 켜 두면 좋아요):")
            for line in credit.splitlines():
                print(f"      {line}")
            print(f"    (같은 내용: {cpath})")
    return out_path


def prepare_scene(vr, sc: Scene, si: int = 0):
    """3D 장면 준비 → (prep, 카메라). 3D가 아니면 (None, None)"""
    if not sc.world or vr is None:
        return None, None
    prep = vr.prepare(sc.world, sc.crowds, seed=sc.seed, shot=sc.shot, props=sc.props, vfx=sc.vfx, look=sc.look,
                      weather=sc.weather)
    cam = sc.shot if isinstance(sc.shot, dict) else prep.shots.get(sc.shot or next(iter(prep.shots)))
    if cam is None:
        vr.release(prep)
        raise ValueError(f"장면 {si + 1}: 샷 '{sc.shot}' 없음 (가능: {', '.join(prep.shots)})")
    return prep, cam


def draw_labels(img: np.ndarray, vr, labels: list, t: float, font_path: str | None) -> np.ndarray:
    """3D 점에 붙는 라벨: 점 위에 말풍선 상자 + 연결선 + 점. 카메라가 움직여도 대상에 붙어 다닌다."""
    active = [lb for lb in labels if lb.get("from", 0) <= t <= lb.get("to", 1e9) and lb.get("text")]
    if not active or font_path is None:
        return img
    from PIL import ImageDraw, ImageFont
    h, w = img.shape[:2]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    size = max(10, int(w / 360 * 13))
    font = ImageFont.truetype(font_path, size=size, index=0)
    pts = vr.project([lb["at"] for lb in active])
    for lb, pt in zip(active, pts):
        if pt is None or not (-0.05 * w < pt[0] < 1.05 * w and -0.05 * h < pt[1] < 1.05 * h):
            continue
        x, y = pt
        accent = text_color(lb.get("color", "#d62848"))
        label_text = text.safe_text(lb["text"])
        tw = font.getlength(label_text)
        pad = size * 0.45
        up = size * float(lb.get("lift", 2.6))
        bx = min(max(x - tw / 2 - pad, 4), w - tw - 2 * pad - 4)
        by = max(y - up - size - 2 * pad, 4)
        d.line([(x, y), (x, by + size + 2 * pad)], fill=accent, width=max(2, size // 6))
        r = size * 0.32
        d.ellipse([x - r, y - r, x + r, y + r], fill=accent, outline=(255, 255, 255), width=max(1, size // 10))
        d.rectangle([bx, by, bx + tw + 2 * pad, by + size + 2 * pad], fill=(255, 255, 255), outline=accent,
                    width=max(2, size // 7))
        d.text((bx + pad, by + pad * 0.7), label_text, font=font, fill=(28, 26, 34))
    return np.array(im)   # 쓰기 가능한 복사본 (뒤에서 자막을 덧그림)


def render_3d_frame(vr, prep, cam: dict, sc: Scene, t: float, font: str | None = None) -> np.ndarray:
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
    out = np.clip(img, 0, 255).astype(np.uint8)
    out = overlay.bloom(out, sc.bloom)
    if sc.look == "smooth":
        out = overlay.grade_smooth(out)
    elif sc.look == "cinematic":
        out = overlay.grade_cinematic(out)
    if sc.labels:
        out = draw_labels(out, vr, sc.labels, t, font)
    if level < 1.0:
        out = (out.astype(np.float32) * level).astype(np.uint8)
    return out


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
