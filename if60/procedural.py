"""절차적(procedural) 도트 레이어.

AI 생성 API 없이도 파이프라인 전체를 돌려볼 수 있도록 우주/행성/지형 레이어를 코드로 그린다.
모든 레이어는 저해상도 RGBA(uint8) 배열이며, 알파는 0 또는 255만 쓴다(도트 경계 유지).
"""
from __future__ import annotations

import math

import numpy as np

from .pixelate import hex_to_rgb

RGBA = np.ndarray


def _rgba(h: int, w: int) -> RGBA:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _c(hexstr: str) -> np.ndarray:
    return np.array(hex_to_rgb(hexstr), dtype=np.uint8)


def value_noise(h: int, w: int, cell: int, seed: int) -> np.ndarray:
    """0~1 값 노이즈 (격자 난수 + 쌍선형 보간)."""
    rng = np.random.default_rng(seed)
    gh, gw = h // cell + 2, w // cell + 2
    grid = rng.random((gh, gw)).astype(np.float32)
    ys = np.arange(h) / cell
    xs = np.arange(w) / cell
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(h: int, w: int, seed: int, octaves: int = 3, cell: int = 24) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        out += value_noise(h, w, max(2, cell >> o), seed + o * 31) * amp
        total += amp
        amp *= 0.5
    return out / total


_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


def _bayer(h: int, w: int) -> np.ndarray:
    return np.tile(_BAYER, (h // 4 + 1, w // 4 + 1))[:h, :w]


# ---------------------------------------------------------------- layers
class Layer:
    """speed: 카메라 이동에 대한 패럴랙스 배율 (0=고정 배경, 1=카메라와 같이 움직임)."""

    def __init__(self, img: RGBA, speed: float = 1.0, animator=None):
        self.img, self.speed, self.animator = img, speed, animator

    def frame(self, t: float) -> RGBA:
        return self.animator(self.img, t) if self.animator else self.img


def sky(w: int, h: int, colors: list[str], **_) -> Layer:
    """색 띠 그라디언트 + 띠 경계 디더링 (레트로 하늘)."""
    img = _rgba(h, w)
    n = len(colors)
    pos = np.linspace(0, n - 1, h)[:, None] + (_bayer(h, w) - 0.5) * 0.9
    idx = np.clip(np.round(pos), 0, n - 1).astype(int)
    pal = np.stack([_c(c) for c in colors])
    img[..., :3] = pal[idx]
    img[..., 3] = 255
    return Layer(img, speed=0.0)


def starfield(w: int, h: int, density: float = 0.012, seed: int = 1,
              colors=("#ffffff", "#9fb8ff", "#ffe8a3"), twinkle: bool = True, **_) -> Layer:
    rng = np.random.default_rng(seed)
    img = _rgba(h, w)
    n = int(w * h * density)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    cols = np.stack([_c(c) for c in colors])[rng.integers(0, len(colors), n)]
    img[ys, xs, :3] = cols
    img[ys, xs, 3] = 255
    phase = rng.random(n) * math.tau
    big = rng.random(n) < 0.06  # 일부는 십자 모양 큰 별

    for y, x, c, b in zip(ys, xs, cols, big):
        if b:
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    img[yy, xx, :3], img[yy, xx, 3] = c // 2, 255

    def animate(base: RGBA, t: float) -> RGBA:
        if not twinkle:
            return base
        out = base.copy()
        off = np.sin(phase + t * 3.0) < -0.75  # 일정 비율의 별이 잠깐 꺼짐
        out[ys[off], xs[off], 3] = 0
        return out

    return Layer(img, speed=0.15, animator=animate)


def planet(w: int, h: int, radius: int = 60, cx: float = 0.5, cy: float = 0.5,
           ocean: str = "#2b5fb3", land: str = "#3f9b4f", ice: str = "#e8f1ff",
           shadow: str = "#0d1a3a", glow: str = "#6fb5ff", land_ratio: float = 0.45,
           light=(-0.6, -0.5), seed: int = 3, spin: float = 6.0, **_) -> Layer:
    """구형 행성. 대륙은 노이즈를 구면 좌표에 감아서 그리고, spin(px/초)으로 자전시킨다.
    spin=0 이면 자전이 멈춘다 → '지구 자전이 멈춘다면' 장면에 그대로 쓸 수 있음."""
    size = radius * 2 + 4
    tex_w = radius * 8
    noise = fbm(radius * 2, tex_w, seed, octaves=4, cell=radius // 2)
    noise = np.concatenate([noise, noise[:, : radius * 2]], axis=1)  # 이음매용
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = (xx - size / 2 + 0.5) / radius, (yy - size / 2 + 0.5) / radius
    r2 = dx * dx + dy * dy
    inside = r2 <= 1.0
    dz = np.sqrt(np.clip(1 - r2, 0, 1))
    lx, ly = light
    lz = math.sqrt(max(0.0, 1 - lx * lx - ly * ly))
    lum = dx * lx + dy * ly + dz * lz
    lat = np.arcsin(np.clip(dy, -1, 1))
    lon = np.arctan2(dx, dz)  # -pi/2..pi/2 보이는 반구
    v = ((lat / math.pi + 0.5) * (radius * 2 - 1)).astype(int).clip(0, radius * 2 - 1)
    u_base = (lon / math.pi + 0.5) * radius * 2
    bayer = _bayer(size, size)
    ring = (r2 > 1.0) & (r2 <= (1 + 2.5 / radius) ** 2)

    def render(_: RGBA, t: float) -> RGBA:
        img = _rgba(h, w)
        u = ((u_base + t * spin) % tex_w).astype(int)
        n = noise[v, u]
        col = np.where((n > 1 - land_ratio)[..., None], _c(land), _c(ocean))
        polar = np.abs(dy) > 0.82 + (n - 0.5) * 0.2
        col = np.where(polar[..., None], _c(ice), col)
        dark = lum + (bayer - 0.5) * 0.35 < 0.05  # 디더링된 명암 경계
        col = np.where(dark[..., None], _c(shadow), col)
        spr = _rgba(size, size)
        spr[inside, :3] = col[inside]
        spr[inside, 3] = 255
        spr[ring, :3], spr[ring, 3] = _c(glow), 255
        ox, oy = int(w * cx - size / 2), int(h * cy - size / 2)
        paste(img, spr, ox, oy)
        return img

    return Layer(_rgba(h, w), speed=0.35, animator=render)


def sun(w: int, h: int, radius: int = 14, cx: float = 0.75, cy: float = 0.3,
        core: str = "#fff4c2", rim: str = "#ffb347", **_) -> Layer:
    img = _rgba(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - w * cx) ** 2 + (yy - h * cy) ** 2)
    img[d <= radius] = (*_c(rim), 255)
    img[d <= radius - 3] = (*_c(core), 255)
    return Layer(img, speed=0.1)


def mountains(w: int, h: int, color: str = "#2a2350", top: float = 0.62,
              rough: float = 0.18, seed: int = 5, snow: str | None = None, speed: float = 0.5,
              **_) -> Layer:
    # 1차원 fbm 능선: 큰 산세 + 잔 굴곡. 봉우리를 살짝 뾰족하게 하려고 제곱근 대신 1.3 거듭제곱
    ridge = fbm(8, w, seed, octaves=4, cell=max(8, w // 4))[4]
    ridge = (ridge - ridge.min()) / (np.ptp(ridge) + 1e-6)
    ridge = ridge ** 1.3
    ys = (h * top - ridge * h * rough).astype(int)
    img = _rgba(h, w)
    for x in range(w):
        img[ys[x]:, x] = (*_c(color), 255)
        if snow:
            img[ys[x]:ys[x] + 2, x, :3] = _c(snow)
    return Layer(img, speed=speed)


def ground(w: int, h: int, color: str = "#16122e", top: float = 0.82,
           detail: str = "#2e2758", seed: int = 9, speed: float = 1.0, **_) -> Layer:
    img = _rgba(h, w)
    y0 = int(h * top)
    img[y0:] = (*_c(color), 255)
    rng = np.random.default_rng(seed)
    n = (h - y0) * w // 12
    ys, xs = rng.integers(y0 + 1, h, n), rng.integers(0, w, n)
    img[ys, xs, :3] = _c(detail)
    img[y0, :, :3] = _c(detail)
    return Layer(img, speed=speed)


def ocean(w: int, h: int, top: float = 0.7, deep: str = "#123a6b", light: str = "#2f74c0",
          foam: str = "#cfe8ff", wave_speed: float = 1.0, **_) -> Layer:
    y0 = int(h * top)

    def render(_: RGBA, t: float) -> RGBA:
        img = _rgba(h, w)
        img[y0:] = (*_c(deep), 255)
        for i, y in enumerate(range(y0, h, 6)):
            xs = np.arange(w)
            on = np.sin(xs * 0.35 + t * 4 * wave_speed + i * 1.7) > 0.55
            img[y, on, :3] = _c(light)
            if i == 0:
                img[y0, on, :3] = _c(foam)
        return img

    return Layer(_rgba(h, w), speed=1.0, animator=render)


# 원작 디자인 마스코트: 'IF 60' 우주비행사 (12x16)
_ASTRO = [
    "....WWWW....",
    "...WWWWWW...",
    "..WWVVVVWW..",
    "..WVVCCVVW..",
    "..WVVVVVVW..",
    "..WWVVVVWW..",
    "...WWWWWW...",
    ".OWWWOOWWWO.",
    "OWWWWOOWWWWO",
    "OWWWWWWWWWWO",
    "OW.WWWWWW.WO",
    "...WWWWWW...",
    "...WW..WW...",
    "...WW..WW...",
    "..OOO..OOO..",
    "............",
]
_ASTRO_COL = {"W": "#e9eef7", "V": "#1d2b5a", "C": "#7fd3ff", "O": "#ff8a3d"}


def mascot(w: int, h: int, cx: float = 0.5, cy: float = 0.72, scale: int = 2,
           float_amp: int = 3, **_) -> Layer:
    sh, sw = len(_ASTRO), len(_ASTRO[0])
    spr = _rgba(sh, sw)
    for y, row in enumerate(_ASTRO):
        for x, ch in enumerate(row):
            if ch in _ASTRO_COL:
                spr[y, x] = (*_c(_ASTRO_COL[ch]), 255)
    spr = spr.repeat(scale, 0).repeat(scale, 1)

    def render(_: RGBA, t: float) -> RGBA:
        img = _rgba(h, w)
        bob = int(round(math.sin(t * 2.2) * float_amp))
        paste(img, spr, int(w * cx - spr.shape[1] / 2), int(h * cy - spr.shape[0] / 2) + bob)
        return img

    return Layer(_rgba(h, w), speed=1.0, animator=render)


def paste(dst: RGBA, src: RGBA, ox: int, oy: int) -> None:
    """알파 0/255 기준으로 src를 dst에 붙인다(범위 밖은 잘림)."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + w, W), min(oy + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    m = s[..., 3] > 127
    d = dst[y0:y1, x0:x1]
    d[m] = s[m]


# ---------------------------------------------------------------- crowd
# 군중용 미니 스프라이트 (원작 디자인). 문자 → 색 슬롯: B=몸/옷, D=어두운 색, H=머리, F=얼굴, P=바지, E=눈
_CROWD_SPRITES = {
    "person": ([".HH.", ".FF.", "BBBB", "BBBB", ".BB.", ".PP.", ".PP.", ".P.P"], 1),
    "cat": (["B...B.", "BB..BB", "BBBBB.", "BEBEB.", "BBBBBB", ".B..B."], 1),
    "dog": (["D.....", "DBBB..", "BEBBBB", "BBBBBB", ".BBBB.", ".B..B."], 1),
    "car": (["..DDDD..", ".DFFFFD.", "BBBBBBBB", "BBBBBBBB", ".D....D."], 1),
}
_CROWD_COLORS = {
    "person": {"B": ["#c8384a", "#2f6fd6", "#4fa05a", "#ffe07a", "#e8f4ff", "#7a2448", "#ff9f43"],
               "H": ["#1a1414", "#3a2418", "#5a3b2e"], "F": ["#f2c9a0", "#d9a77a"],
               "P": ["#221c47", "#3b3a8c", "#2a2a2a"]},
    "cat": {"B": ["#ff9f43", "#f2f2f2", "#2a2a2a", "#8a8aa8", "#c98a4a", "#ffe0b0"], "E": ["#123018"]},
    "dog": {"B": ["#c98a4a", "#f2f2f2", "#5a3b2e", "#e0c080", "#2a2a2a"], "D": ["#3a2418"],
            "E": ["#101010"]},
    "car": {"B": ["#c8384a", "#2f6fd6", "#e8f4ff", "#2a2a2a", "#ffe07a", "#8a8aa8"],
            "D": ["#141414"], "F": ["#9fd3ff"]},
}


def _crowd_sprite(kind: str, rng: np.random.Generator) -> RGBA:
    rows, _ = _CROWD_SPRITES[kind]
    cols = {k: _c(rng.choice(v)) for k, v in _CROWD_COLORS[kind].items()}
    spr = _rgba(len(rows), len(rows[0]))
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in cols:
                spr[y, x] = (*cols[ch], 255)
    if rng.random() < 0.5:
        spr = spr[:, ::-1]  # 좌우 반전으로 다양성
    return spr


def crowd(w: int, h: int, agent: str = "cat", count: int = 800, top: float = 0.55,
          bottom: float = 1.0, near_scale: int = 2, vx: float = 0.0, bob: float = 0.35,
          mix: dict | None = None, grid: float = 0.0, seed: int = 7, **_) -> Layer:
    """수백~수천 개의 미니 스프라이트로 군중을 만든다 ('시뮬레이션' 느낌의 핵심).

    agent: person | cat | dog | car (또는 mix: {cat: 0.8, person: 0.2})
    top/bottom: 군중이 차지할 세로 범위(비율). 위쪽은 멀리(작게), 아래쪽은 가까이(크게).
    vx: 군중 전체가 흘러가는 속도(px/초, +는 오른쪽). bob: 들썩이는 개체 비율.
    grid: 0~1. 클수록 줄 맞춰 선 대열 (시험장, 행진 등).

    성능: 개체를 매 프레임 그리지 않고 들썩임 4단계를 미리 렌더링해 두고 순환시킨다.
    """
    rng = np.random.default_rng(seed)
    kinds = list((mix or {agent: 1.0}).items())
    names = [k for k, _ in kinds]
    probs = np.array([p for _, p in kinds], float)
    probs /= probs.sum()
    y0, y1 = int(h * top), int(h * bottom)
    agents = []
    for i in range(count):
        if grid > 0:
            cols_n = max(1, int(math.sqrt(count * w / max(1, (y1 - y0)))))
            gx, gy = i % cols_n, i // cols_n
            rows_n = math.ceil(count / cols_n)
            x = (gx + 0.5) / cols_n * w + rng.normal(0, (1 - grid) * 4)
            y = y0 + (gy + 0.5) / rows_n * (y1 - y0) + rng.normal(0, (1 - grid) * 3)
        else:
            # 원근: 아래쪽일수록 밀도가 낮고 크게 → y를 제곱근 분포로
            y = y0 + (y1 - y0) * rng.random() ** 0.8
            x = rng.random() * w
        depth = (y - y0) / max(1, y1 - y0)
        scale = near_scale if depth > 0.55 else 1
        spr = _crowd_sprite(rng.choice(names, p=probs), rng)
        if scale > 1:
            spr = spr.repeat(scale, 0).repeat(scale, 1)
        phase = int(rng.integers(0, 4)) if rng.random() < bob else -1
        agents.append((y, int(x), spr, phase))
    agents.sort(key=lambda a: a[0])  # 먼 개체부터 그려서 가까운 개체가 위로

    variants = []
    for v in range(4):
        img = _rgba(h, w)
        for y, x, spr, phase in agents:
            lift = 1 if (phase >= 0 and (v + phase) % 4 < 2) else 0
            yy = int(y) - spr.shape[0] - lift
            paste(img, spr, x - spr.shape[1] // 2, yy)
            if vx:  # 가로로 흐를 때 이음매가 생기지 않게 양쪽에도 그림
                paste(img, spr, x - spr.shape[1] // 2 - w, yy)
                paste(img, spr, x - spr.shape[1] // 2 + w, yy)
        variants.append(img)

    def animate(_: RGBA, t: float) -> RGBA:
        img = variants[int(t * 6) % 4]
        return np.roll(img, int(round(vx * t)), axis=1) if vx else img

    return Layer(variants[0], speed=1.0, animator=animate)


KINDS = {
    "sky": sky, "starfield": starfield, "planet": planet, "sun": sun,
    "mountains": mountains, "ground": ground, "ocean": ocean, "mascot": mascot,
    "crowd": crowd,
}


def make(kind: str, w: int, h: int, **params) -> Layer:
    if kind not in KINDS:
        raise ValueError(f"알 수 없는 procedural kind: {kind} (가능: {', '.join(KINDS)})")
    speed = params.pop("speed", None)
    layer = KINDS[kind](w, h, **params)
    if speed is not None:
        layer.speed = speed
    return layer
