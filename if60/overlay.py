"""방송형 화면 요소 — 최종 해상도에서 매끈하게(안티앨리어싱) 그린다.

  Layout      상단 채널·시리즈·두 줄 제목(흰색+노란색) / 하단 경고 줄무늬 — 영상 전체에 고정
  panel       데이터 대시보드: 큰 숫자 카운터+단위, 작은 수치, 진행 막대, 상태 배지
  big_text    장면 속에 뜨는 크고 빛나는 글자 ("×1억", "30,000℃")
  markers     3D 지점에 붙는 빛나는 원형 표적(펄스)
  card        마무리 카드: 어두운 배경 + 질문 + 번호 선택지 칩 + 댓글 유도
  bloom       밝은 부분이 번지는 빛 번짐 후처리
  whiteout / rewind   화면 전환 효과 (하얗게 날아감 / 되감기)

모든 크기는 가로 1080px 기준 단위(u)로 계산해 미리보기(360px)·검수 시트(180px)에서도 같은 배치가 나온다.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .text import safe_text

YELLOW = (255, 210, 63)
WHITE = (255, 255, 255)
INK = (14, 14, 20)
CHANNEL = "만물제작소"
HEADER = 0.158     # 상단 띠 높이 (화면 높이 비율)
FOOTER = 0.08      # 하단 띠 높이
CIRCLED = "①②③④⑤⑥"


# ------------------------------------------------------------------ 공통
@lru_cache(maxsize=128)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=max(6, int(size)), index=0)


def hexrgb(c, default=YELLOW) -> tuple[int, int, int]:
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
    if isinstance(c, (list, tuple)) and len(c) == 3:
        return tuple(int(x) for x in c)
    named = {"red": (239, 68, 68), "blue": (59, 130, 246), "green": (52, 199, 89), "yellow": YELLOW,
             "orange": (255, 140, 40), "white": WHITE, "cyan": (80, 220, 255), "purple": (168, 85, 247)}
    return named.get(c, default) if isinstance(c, str) else default


def fit(path: str, text: str, size: float, max_w: float) -> ImageFont.FreeTypeFont:
    s = int(size)
    while s > 8 and font(path, s).getlength(text) > max_w:
        s = int(s * 0.92)
    return font(path, s)


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def window(item: dict, duration: float) -> tuple[float, float]:
    a = float(item.get("from", 0.0))
    return a, float(item.get("to", duration))


def bold(d: ImageDraw.ImageDraw, xy, text, f, fill, stroke=None, sw=0):
    """굵은 글씨: 같은 색 외곽선으로 두께를 더한다 (Mac 기본 폰트에 Bold가 없을 때도 굵게)."""
    w = max(1, f.size // 22)
    d.text(xy, text, font=f, fill=fill, stroke_width=sw or w, stroke_fill=stroke or fill)


def paste_rgba(dst: np.ndarray, rgba: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """dst(H,W,3 uint8)에 rgba(h,w,4)를 (x,y) 위치로 알파 합성 (화면 밖은 잘라냄)."""
    H, W = dst.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = src[..., 3:4] / 255.0 * opacity
    reg = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (reg * (1 - a) + src[..., :3] * a).astype(np.uint8)


def add_light(dst: np.ndarray, rgb: np.ndarray, x: int, y: int, k: float = 1.0) -> None:
    """가산 합성 (빛 번짐용). rgb: (h,w,3) float"""
    H, W = dst.shape[:2]
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    reg = dst[y0:y1, x0:x1].astype(np.float32) + rgb[y0 - y:y1 - y, x0 - x:x1 - x] * k
    dst[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)


def glow_of(rgba: np.ndarray, color, radius: float, gain: float = 1.6) -> np.ndarray:
    a = Image.fromarray(rgba[..., 3])
    small = a.resize((max(1, a.width // 4), max(1, a.height // 4)), Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(max(1.0, radius / 4)))
    blur = np.asarray(small.resize(a.size, Image.Resampling.BILINEAR), np.float32) / 255.0
    return blur[..., None] * np.array(color, np.float32)[None, None] * gain


# ------------------------------------------------------------------ 빛 번짐 / 색감
def bloom(img: np.ndarray, strength: float = 0.5, threshold: float = 0.72) -> np.ndarray:
    """밝게 빛나는 '작은' 것(불빛·불티·번개·글로우 재질)만 번지게 한다. 1/6 크기에서 계산해 빠르다.
    밝은 배경 전체(흰 스튜디오 벽, 낮 하늘)는 주변 평균을 빼서 번지지 않게 한다."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    sw, sh = max(8, w // 6), max(8, h // 6)
    small = np.asarray(Image.fromarray(img).resize((sw, sh), Image.Resampling.BILINEAR), np.float32) / 255.0
    lum = small.max(-1, keepdims=True)
    bright = small * np.clip((lum - threshold) / (1 - threshold), 0, 1)
    b8 = Image.fromarray((bright * 255).astype(np.uint8))
    base = np.asarray(b8.filter(ImageFilter.GaussianBlur(sw / 6)), np.float32)
    hp = np.clip(bright * 255 - base * 0.92, 0, 255).astype(np.uint8)
    b = Image.fromarray(hp)
    near = np.asarray(b.filter(ImageFilter.GaussianBlur(sw / 90)), np.float32)
    far = np.asarray(b.filter(ImageFilter.GaussianBlur(sw / 28)), np.float32)
    glow = Image.fromarray(np.clip(near * 0.7 + far * 1.1, 0, 255).astype(np.uint8))
    up = np.asarray(glow.resize((w, h), Image.Resampling.BILINEAR), np.float32)
    return np.clip(img.astype(np.float32) + up * strength, 0, 255).astype(np.uint8)


@lru_cache(maxsize=8)
def _vignette(h: int, w: int, k: float) -> np.ndarray:
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :] * 0.75
    return (1 - k * np.clip(x * x + y * y - 0.25, 0, None)).clip(0.3, 1).astype(np.float32)[..., None]


def grade_smooth(img: np.ndarray) -> np.ndarray:
    """매끈한 룩: 약한 비네트 + 살짝 높인 대비·채도 (필름 같은 색감)."""
    f = img.astype(np.float32) / 255.0
    f = np.clip((f - 0.5) * 1.08 + 0.5, 0, 1)
    g = f.mean(-1, keepdims=True)
    f = np.clip(g + (f - g) * 1.12, 0, 1)
    f *= _vignette(img.shape[0], img.shape[1], 0.35)
    return (f * 255).astype(np.uint8)


def darken(img: np.ndarray, k: float = 0.4, blur: bool = False) -> np.ndarray:
    im = Image.fromarray(img)
    if blur:
        s = im.resize((max(1, im.width // 8), max(1, im.height // 8)), Image.Resampling.BILINEAR)
        im = s.filter(ImageFilter.GaussianBlur(2)).resize(im.size, Image.Resampling.BILINEAR)
    f = np.asarray(im, np.float32) * k * _vignette(img.shape[0], img.shape[1], 0.9)
    return f.astype(np.uint8)


def gradient_bg(w: int, h: int, top=(28, 26, 40), bottom=(6, 6, 10)) -> np.ndarray:
    a = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    g = np.array(top, np.float32) * (1 - a) + np.array(bottom, np.float32) * a
    return np.broadcast_to(g, (h, w, 3)).astype(np.uint8).copy()


# ------------------------------------------------------------------ 방송형 레이아웃
def split_title(title: str) -> list[str]:
    """제목 한 줄 → 두 줄 (가운데 가까운 띄어쓰기에서 나눔)"""
    title = title.strip()
    if "\n" in title:
        return title.split("\n")[:2]
    if len(title) <= 9:
        return [title]
    mid = len(title) / 2
    spaces = [i for i, c in enumerate(title) if c == " "]
    if not spaces:
        return [title]
    i = min(spaces, key=lambda s: abs(s - mid))
    return [title[:i], title[i + 1:]]


class Layout:
    """상단 띠(채널·시리즈·두 줄 제목)와 하단 경고 줄무늬. 영상 전체에 고정되어 '방송' 느낌을 만든다."""

    def __init__(self, cfg: dict | bool, title: str, font_path: str, size: tuple[int, int]):
        cfg = cfg if isinstance(cfg, dict) else {}
        self.w, self.h = size
        self.u = self.w / 1080
        self.font_path = font_path
        lines = cfg.get("title") or split_title(title or "")
        self.lines = [lines] if isinstance(lines, str) else list(lines)[:2]
        self.channel = str(cfg.get("channel", CHANNEL))
        self.series = cfg.get("series")
        self.accent = hexrgb(cfg.get("accent", "#ffd23f"))
        self.hh = int(self.h * HEADER)
        self.fh = int(self.h * FOOTER)
        self.header = self._header()
        self.stripe_h = max(2, int(26 * self.u))
        self.stripes = self._stripes()

    @property
    def viewport(self) -> tuple[int, int]:
        """3D 화면이 보이는 세로 범위 (위, 아래 px)"""
        return self.hh, self.h - self.fh

    def _header(self) -> np.ndarray:
        u, w, hh = self.u, self.w, self.hh
        bg = gradient_bg(w, hh, (24, 24, 32), (10, 10, 14))
        im = Image.fromarray(bg)
        d = ImageDraw.Draw(im)
        # 채널 아이콘: 노란 칸 안의 작은 상자(만물을 '제작'하는 공방) — 채널명은 작게만
        s, x0, y0 = int(64 * u), int(36 * u), int(22 * u)
        d.rounded_rectangle([x0, y0, x0 + s, y0 + s], radius=int(14 * u), fill=self.accent)
        cx, cy, r = x0 + s / 2, y0 + s / 2 + 2 * u, s * 0.27
        top = [(cx, cy - r), (cx + r, cy - r / 2), (cx, cy), (cx - r, cy - r / 2)]
        left = [(cx - r, cy - r / 2), (cx, cy), (cx, cy + r), (cx - r, cy + r / 2)]
        right = [(cx, cy), (cx + r, cy - r / 2), (cx + r, cy + r / 2), (cx, cy + r)]
        d.polygon(top, fill=(255, 255, 255))
        d.polygon(left, fill=(40, 36, 30))
        d.polygon(right, fill=(92, 80, 52))
        nf = font(self.font_path, int(30 * u))
        bold(d, (x0 + s + 16 * u, y0 + 2 * u), self.channel, nf, (236, 236, 242))
        if self.series:
            sf = font(self.font_path, int(21 * u))
            d.text((x0 + s + 16 * u, y0 + 38 * u), safe_text(str(self.series)), font=sf, fill=(150, 150, 168))
        # 두 줄 제목: 첫 줄 흰색, 둘째 줄 노란색
        ys = [int(104 * u), int(180 * u)] if len(self.lines) == 2 else [int(132 * u)]
        for i, (line, y) in enumerate(zip(self.lines, ys)):
            line = safe_text(str(line))
            f = fit(self.font_path, line, 62 * u, w - 80 * u)
            tw = f.getlength(line)
            col = WHITE if (i == 0 and len(self.lines) == 2) else self.accent
            bold(d, ((w - tw) / 2, y), line, f, col)
        d.rectangle([0, hh - max(1, int(3 * u)), w, hh], fill=(*self.accent,))
        return np.asarray(im).copy()

    def _stripes(self) -> np.ndarray:
        u = self.u
        period = max(4, int(56 * u))
        h = self.stripe_h
        yy, xx = np.mgrid[0:h, 0:self.w + period]
        on = ((xx + yy) // (period // 2)) % 2 == 0
        out = np.where(on[..., None], np.array(self.accent, np.uint8), np.array((18, 18, 22), np.uint8))
        return out.astype(np.uint8)

    def apply(self, img: np.ndarray, t: float) -> np.ndarray:
        h, w = img.shape[:2]
        img[:self.hh] = self.header
        img[h - self.fh:] = (10, 10, 14)
        period = max(4, int(56 * self.u))
        off = int(t * 40 * self.u) % period
        img[h - self.fh:h - self.fh + self.stripe_h] = self.stripes[:, off:off + w]
        return img


# ------------------------------------------------------------------ 숫자 카운터
def kr_number(v: float) -> str:
    """150000000 → '1억 5,000만' (한국식 큰 수)"""
    neg = v < 0
    v = abs(v)
    parts = []
    for unit, name in ((1e16, "경"), (1e12, "조"), (1e8, "억"), (1e4, "만")):
        if v >= unit:
            q = int(v // unit)
            parts.append(f"{q:,}{name}")
            v -= q * unit
    if v >= 1 or not parts:
        parts.append(f"{v:,.0f}" if not parts else f"{int(v):,}")
    s = " ".join(parts[:2])
    return ("-" if neg else "") + s


def counter_value(spec, t: float, duration: float) -> tuple[str, str]:
    """value 설정 → (숫자 문자열, 단위). from→to를 시간에 따라 굴린다. scale: log면 자릿수가 고르게 올라감."""
    if not isinstance(spec, dict):
        return safe_text(str(spec)), ""
    a, b = spec.get("from", 0), spec.get("to", spec.get("from", 0))
    unit = str(spec.get("unit", ""))
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return safe_text(str(spec.get("to", spec.get("from", "")))), unit
    at = float(spec.get("at", 0.0))
    length = float(spec.get("length", max(0.1, duration - at - 0.3)))
    p = min(1.0, max(0.0, (t - at) / max(length, 1e-6)))
    p = ease_out(p) if spec.get("ease", True) else p
    log = spec.get("scale") == "log" or (spec.get("scale") != "linear" and a >= 0 and b > 0 and
                                          max(a, 1) and b / max(a, 1) >= 1000)
    if log and a >= 0 and b > 0:
        lo = math.log10(max(a, 1.0))
        v = 10 ** (lo + (math.log10(b) - lo) * p) if p > 0 else a
    else:
        v = a + (b - a) * p
    fmt = spec.get("format", "{:,.0f}")
    if fmt == "kr":
        return kr_number(v), unit
    try:
        return safe_text(fmt.format(v)), unit
    except (ValueError, IndexError, KeyError):
        return f"{v:,.0f}", unit


# ------------------------------------------------------------------ 대시보드 패널
def draw_panel(img: np.ndarray, panel: dict, t: float, duration: float, font_path: str,
               top: int, u: float) -> None:
    """반투명 데이터 패널: 제목 / 상태 배지 / 큰 숫자+단위 / 작은 수치들 / 진행 막대"""
    W = img.shape[1]
    pw = int(min(W - 60 * u, float(panel.get("width", 0.56)) * W))
    pad = int(22 * u)
    ft_title = font(font_path, int(25 * u))
    ft_big = font(font_path, int(80 * u))
    ft_unit = font(font_path, int(36 * u))
    ft_small = font(font_path, int(25 * u))
    ft_bar = font(font_path, int(21 * u))
    stats = [safe_text(str(s)) for s in (panel.get("stats") or [])][:4]
    bars = [b for b in (panel.get("bars") or []) if isinstance(b, dict)][:3]
    has_value = panel.get("value") is not None
    ph = pad * 2 + int(30 * u) + (int(92 * u) if has_value else 0) + len(stats) * int(34 * u) \
        + len(bars) * int(46 * u)
    im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(18 * u), fill=(10, 12, 22, 190),
                        outline=(120, 190, 255, 140), width=max(1, int(2 * u)))
    y = pad
    d.text((pad, y), safe_text(str(panel.get("title", ""))), font=ft_title, fill=(180, 192, 220, 255))
    status = panel.get("status")
    if status:
        st = safe_text(str(status))
        sc = hexrgb(panel.get("status_color", "red"))
        sw = int(ft_bar.getlength(st) + 44 * u)
        sx = pw - pad - sw
        d.rounded_rectangle([sx, y - 2 * u, sx + sw, y + 32 * u], radius=int(16 * u), fill=(*sc, 235))
        if int(t * 2) % 2 == 0:
            d.ellipse([sx + 12 * u, y + 10 * u, sx + 24 * u, y + 22 * u], fill=(255, 255, 255, 255))
        d.text((sx + 32 * u, y + 3 * u), st, font=ft_bar, fill=(255, 255, 255, 255))
    y += int(34 * u)
    if has_value:
        num, unit = counter_value(panel["value"], t, duration)
        f = fit(font_path, num, 80 * u, pw - pad * 2 - ft_unit.getlength(unit) - 14 * u)
        bold(d, (pad, y + (80 * u - f.size) * 0.6), num, f, (255, 255, 255, 255))
        d.text((pad + f.getlength(num) + 12 * u, y + 36 * u), safe_text(unit), font=ft_unit,
               fill=(*YELLOW, 255))
        y += int(92 * u)
    for s in stats:
        d.text((pad, y), "· " + s, font=ft_small, fill=(214, 220, 236, 255))
        y += int(34 * u)
    for b in bars:
        a0, a1 = float(b.get("from", 0)), float(b.get("to", b.get("from", 0)))
        at = float(b.get("at", 0.0))
        length = float(b.get("length", max(0.1, duration - at - 0.3)))
        v = a0 + (a1 - a0) * ease_out((t - at) / max(length, 1e-6))
        v = min(1.0, max(0.0, v))
        col = hexrgb(b.get("color", "blue"))
        d.text((pad, y), safe_text(str(b.get("label", ""))), font=ft_bar, fill=(200, 208, 228, 255))
        txt = safe_text(str(b.get("text", f"{v * 100:.0f}%")))
        d.text((pw - pad - ft_bar.getlength(txt), y), txt, font=ft_bar, fill=(*col, 255))
        by = y + int(26 * u)
        bh = max(2, int(12 * u))
        d.rounded_rectangle([pad, by, pw - pad, by + bh], radius=bh // 2, fill=(255, 255, 255, 40))
        if v > 0.003:
            d.rounded_rectangle([pad, by, pad + (pw - 2 * pad) * v, by + bh], radius=bh // 2, fill=(*col, 255))
        y += int(46 * u)
    arr = np.asarray(im)
    side = panel.get("side", "left")
    x = int(30 * u) if side == "left" else W - pw - int(30 * u)
    k = ease_out(t / 0.3)
    x += int((-1 if side == "left" else 1) * (1 - k) * 80 * u)
    paste_rgba(img, arr, x, top, opacity=k)


# ------------------------------------------------------------------ 빛나는 큰 글자
@lru_cache(maxsize=48)
def _glow_text(text: str, px: int, color: tuple, font_path: str) -> tuple[np.ndarray, np.ndarray]:
    f = font(font_path, px)
    tw = int(f.getlength(text)) + px
    pad = px // 2
    im = Image.new("RGBA", (tw + pad * 2, int(px * 1.5) + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((pad + px // 2, pad), text, font=f, fill=(*color, 255),
           stroke_width=max(2, px // 14), stroke_fill=(*color, 255))
    glow_src = np.asarray(im).copy()
    d.text((pad + px // 2, pad), text, font=f, fill=(255, 255, 255, 255) if color == YELLOW else (*color, 255),
           stroke_width=max(2, px // 14), stroke_fill=(40, 24, 10, 255))
    core = np.asarray(im).copy()
    # 흰 속 + 색 외곽 느낌: 노란 글자는 안쪽을 밝은 노랑으로
    if color == YELLOW:
        m = core[..., 3] > 0
        inner = (core[..., :3].astype(int).sum(-1) > 600) & m
        core[inner, :3] = (255, 244, 190)
    glow = glow_of(glow_src, color, px * 0.45)
    return core, glow


def draw_big_text(img: np.ndarray, item: dict, t: float, duration: float, font_path: str,
                  u: float, point=None) -> None:
    a, b = window(item, duration)
    if not (a <= t <= b) or not item.get("text"):
        return
    H, W = img.shape[:2]
    color = hexrgb(item.get("color", "#ffd23f"))
    px = int(130 * u * float(item.get("size", 1.0)))
    core, glow = _glow_text(safe_text(str(item["text"])), max(8, px), color, font_path)
    k = ease_out((t - a) / 0.22)
    out = min(1.0, (b - t) / 0.2) if b < duration - 0.05 else 1.0
    s = 1.35 - 0.35 * k
    if abs(s - 1) > 0.01:
        sz = (max(1, int(core.shape[1] * s)), max(1, int(core.shape[0] * s)))
        core = np.asarray(Image.fromarray(core).resize(sz, Image.Resampling.BILINEAR))
        glow = np.asarray(Image.fromarray(glow.clip(0, 255).astype(np.uint8)).resize(sz, Image.Resampling.BILINEAR),
                          np.float32)
    if point is not None:
        cx, cy = point
    else:
        pos = item.get("pos", [0.5, 0.42])
        cx, cy = float(pos[0]) * W, float(pos[1]) * H
    x, y = int(cx - core.shape[1] / 2), int(cy - core.shape[0] / 2)
    pulse = 1.0 + 0.15 * math.sin(t * 6.0)
    add_light(img, glow, x, y, k * out * pulse)
    paste_rgba(img, core, x, y, opacity=k * out)


# ------------------------------------------------------------------ 표적 링
def draw_marker(img: np.ndarray, item: dict, t: float, duration: float, point, font_path: str, u: float) -> None:
    a, b = window(item, duration)
    if point is None or not (a <= t <= b):
        return
    color = hexrgb(item.get("color", "#ffd23f"))
    r = 56 * u * float(item.get("size", 1.0))
    k = ease_out((t - a) / 0.25)
    size = int(r * 5)
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = size / 2
    lw = max(2, int(6 * u))
    rr = r * (1.6 - 0.6 * k)
    d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=(*color, 255), width=lw)
    ph = (t * 1.2) % 1.0
    pr = rr * (1 + 0.9 * ph)
    d.ellipse([c - pr, c - pr, c + pr, c + pr], outline=(*color, int(255 * (1 - ph))), width=max(1, lw // 2))
    for ang in (0, 90, 180, 270):   # 조준 눈금
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        d.line([(c + ca * rr * 0.7, c + sa * rr * 0.7), (c + ca * rr * 1.15, c + sa * rr * 1.15)],
               fill=(*color, 255), width=lw)
    arr = np.asarray(im)
    x, y = int(point[0] - c), int(point[1] - c)
    add_light(img, glow_of(arr, color, r * 0.35, 1.2), x, y, k)
    paste_rgba(img, arr, x, y, opacity=k)
    if item.get("text"):
        f = font(font_path, int(30 * u))
        txt = safe_text(str(item["text"]))
        tw = f.getlength(txt)
        tag = Image.new("RGBA", (int(tw + 28 * u), int(46 * u)), (0, 0, 0, 0))
        td = ImageDraw.Draw(tag)
        td.rounded_rectangle([0, 0, tag.width - 1, tag.height - 1], radius=int(10 * u), fill=(*color, 235))
        td.text((14 * u, 6 * u), txt, font=f, fill=(20, 16, 10, 255))
        paste_rgba(img, np.asarray(tag), int(point[0] + rr * 0.8), int(point[1] - rr * 1.2 - 20 * u), opacity=k)


# ------------------------------------------------------------------ 마무리 카드
def draw_card(img: np.ndarray, card: dict, t: float, duration: float, font_path: str, u: float,
              area: tuple[int, int]) -> None:
    """질문 + 번호 선택지 칩(차례로 등장) + 댓글 유도 문구"""
    H, W = img.shape[:2]
    top, bottom = area
    vh = bottom - top
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    accent = hexrgb(card.get("accent", "#ffd23f"))
    q = safe_text(str(card.get("question", "")))
    opts = [safe_text(str(o)) for o in (card.get("options") or [])][:4]
    ch = 104 * u
    gap = 26 * u
    f = font(font_path, int(58 * u))
    lines = _wrap(q, f, W - 120 * u) if q else []
    block = len(lines) * f.size * 1.3 + (30 * u if q else 0) + len(opts) * (ch + gap) + 80 * u
    y = top + max(vh * 0.06, (vh - block) / 2 - vh * 0.04)
    if q:
        k = ease_out(t / 0.3)
        for line in lines:
            tw = f.getlength(line)
            bold(d, ((W - tw) / 2, y + (1 - k) * 30 * u), line, f, (255, 255, 255, int(255 * k)))
            y += f.size * 1.3
        y += 30 * u
    cw = W - 140 * u
    of = font(font_path, int(42 * u))
    nf = font(font_path, int(40 * u))
    for i, o in enumerate(opts):
        k = ease_out((t - 0.35 - i * 0.35) / 0.3)
        if k <= 0:
            continue
        x0 = (W - cw) / 2 + (1 - k) * 160 * u
        y0 = y + i * (ch + gap)
        d.rounded_rectangle([x0, y0, x0 + cw, y0 + ch], radius=int(ch / 2),
                            fill=(255, 255, 255, int(28 * k)), outline=(*accent, int(255 * k)),
                            width=max(1, int(3 * u)))
        cr = ch * 0.36
        cx, cy = x0 + ch * 0.55, y0 + ch / 2
        d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(*accent, int(255 * k)))
        num = str(i + 1)
        bold(d, (cx - nf.getlength(num) / 2, cy - nf.size * 0.62), num, nf, (16, 14, 10, int(255 * k)))
        ft = fit(font_path, o, 42 * u, cw - ch * 1.3)
        d.text((x0 + ch * 1.05, cy - ft.size * 0.62), o, font=ft, fill=(255, 255, 255, int(255 * k)))
    cta = card.get("cta", "댓글로 번호를 남겨주세요")
    if cta:
        k = ease_out((t - 0.5 - len(opts) * 0.35) / 0.3)
        if k > 0 and (int(t * 1.6) % 2 == 0 or t < 0.9 + len(opts) * 0.35):
            cf = font(font_path, int(38 * u))
            cta = safe_text(str(cta))
            tw = cf.getlength(cta)
            yy = y + len(opts) * (ch + gap) + 24 * u
            bold(d, ((W - tw) / 2, yy), cta, cf, (*accent, int(255 * k)))
    img[:] = np.asarray(im)


def _wrap(text: str, f, max_w: float) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and f.getlength(trial) > max_w:
            out.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------------ 전환 효과
def whiteout(img: np.ndarray, level: float) -> np.ndarray:
    if level <= 0:
        return img
    f = img.astype(np.float32)
    return (f + (255 - f) * min(1.0, level)).astype(np.uint8)


def whiteout_level(prm: dict, t: float) -> float:
    """빠르게 하얘졌다가(35%) 천천히 돌아온다. hold: true면 끝까지 하얗게 유지 (장면 전환용)"""
    at, length = float(prm.get("at", 0.0)), float(prm.get("length", 1.2))
    if t < at or (t > at + length and not prm.get("hold")):
        return 0.0
    p = (t - at) / max(length, 1e-6)
    if p < 0.35:
        return ease_out(p / 0.35)
    if prm.get("hold"):
        return 1.0
    return 1.0 - ease_out((p - 0.35) / 0.65)


def rewind(img: np.ndarray, t: float, prm: dict, font_path: str, u: float, top: int) -> np.ndarray:
    """되감기: 가로 줄 밀림 + 탈색 + 주사선 + ◀◀ 표시"""
    at, length = float(prm.get("at", 0.0)), float(prm.get("length", 0.9))
    if not (at <= t < at + length):
        return img
    H, W = img.shape[:2]
    out = img.copy()
    rng = np.random.default_rng(int(t * 30))
    band = max(2, int(8 * u))
    k = 1 - abs((t - at) / length - 0.5) * 2
    for y in range(0, H, band):
        if rng.random() < 0.35:
            out[y:y + band] = np.roll(out[y:y + band], int(rng.normal(0, 40 * u * (0.4 + k))), axis=1)
    g = out.astype(np.float32).mean(-1, keepdims=True)
    out = (out * 0.45 + g * 0.55).astype(np.float32)
    out[::max(2, int(4 * u))] *= 0.7
    out = np.clip(out * 1.05, 0, 255).astype(np.uint8)
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    s = 34 * u
    x0, y0 = 40 * u, top + 40 * u
    for i in range(2):
        xx = x0 + i * s * 0.8
        d.polygon([(xx + s * 0.8, y0), (xx + s * 0.8, y0 + s), (xx, y0 + s / 2)], fill=WHITE)
    f = font(font_path, int(34 * u))
    bold(d, (x0 + s * 1.9, y0 - 2 * u), safe_text(str(prm.get("text", "되감기"))), f, WHITE)
    return np.asarray(im).copy()
