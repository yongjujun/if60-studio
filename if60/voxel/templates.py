"""장면 템플릿 — 대본의 비트를 붙일 '세트장'.

템플릿마다 세계(지형·건물)와 이름 붙은 구역(areas: 군중을 놓을 곳), 카메라 샷(shots)을 제공한다.
장면 YAML에서 world: {template: plaza, seed: 3} 와 shot: aerial 처럼 고른다.

  plaza       넓은 광장 + 양옆 도로·빌딩 + 끝에 성문 + 뒤로 산 (광화문 광장 느낌)
  boulevard   고층 빌딩 협곡 사이 도로 (도심 대로)
  crossroad   위에서 내려다본 교차로 + 횡단보도 + 차량
  river_bridge 강 + 트러스 철교 + 건너편 스카이라인 (한강 철교 느낌)
  apartments  아파트 단지 (판상형 동 + 잔디)
  room        방 안 (침대·책상·창문)
  field       넓은 잔디밭 + 멀리 스카이라인 (공원 광장)
  planet      화성/달 표면 (IF 60 우주 회차용)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import models
from .geometry import WorldBuilder, rgb

CITY = ["#9a9ea6", "#b8b2a6", "#c9c4b8", "#7f8a96", "#a39b8f", "#6f7c8c", "#d4cfc4", "#8d939c",
        "#5f7088", "#bfc5cc", "#a8a090"]


@dataclass
class Built:
    builder: WorldBuilder
    areas: dict = field(default_factory=dict)
    shots: dict = field(default_factory=dict)
    crowds: list = field(default_factory=list)
    shadow_radius: float = 100.0
    sky: str = "auto"
    sun_azimuth: float = 35.0
    lighting: dict = field(default_factory=dict)


def _noise1(n: int, rng, smooth: int = 6) -> np.ndarray:
    v = rng.random(n + smooth * 2)
    k = np.ones(smooth) / smooth
    return np.convolve(v, k, "same")[smooth:smooth + n]


def _value_noise2(h: int, w: int, cell: float, rng) -> np.ndarray:
    gh, gw = int(h / cell) + 2, int(w / cell) + 2
    g = rng.random((gh, gw))
    ys, xs = np.arange(h) / cell, np.arange(w) / cell
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a, b = g[y0][:, x0], g[y0][:, x0 + 1]
    c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def building_row(b: WorldBuilder, rng, x_near: float, x_far: float, z0: float, z1: float,
                 h_range=(18, 80), w_range=(18, 40), palette=CITY, gap=(1, 6)) -> None:
    """도로를 따라 늘어선 빌딩 한 줄. x_near는 도로 쪽 면."""
    sgn = 1 if x_far > x_near else -1
    z = z0
    mins, maxs, cols = [], [], []
    tops_min, tops_max, tops_col = [], [], []
    while z < z1:
        w = rng.uniform(*w_range)
        depth = rng.uniform(20, abs(x_far - x_near))
        h = rng.uniform(*h_range) * (1.6 if rng.random() < 0.12 else 1.0)
        xa, xb = sorted([x_near + sgn * rng.uniform(0, 3), x_near + sgn * depth])
        col = rgb(palette[rng.integers(len(palette))])
        mins.append((xa, 0, z)); maxs.append((xb, h, z + w)); cols.append(col)
        if rng.random() < 0.5:   # 옥상 구조물
            cx, cz = (xa + xb) / 2, z + w / 2
            s = rng.uniform(3, 8)
            tops_min.append((cx - s, h, cz - s)); tops_max.append((cx + s, h + rng.uniform(2, 5), cz + s))
            tops_col.append(col * 0.85)
        z += w + rng.uniform(*gap)
    b.boxes(mins, maxs, cols, "facade")
    if tops_min:
        b.boxes(tops_min, tops_max, tops_col, "stone")


def skyline(b: WorldBuilder, rng, x0, x1, z0, z1, n=120, h_range=(20, 120)) -> None:
    xs = rng.uniform(x0, x1, n); zs = rng.uniform(z0, z1, n)
    w = rng.uniform(15, 45, n); d = rng.uniform(15, 40, n); h = rng.uniform(*h_range, n)
    cols = np.stack([rgb(CITY[i]) for i in rng.integers(0, len(CITY), n)])
    b.boxes(np.stack([xs, np.zeros(n), zs], 1), np.stack([xs + w, h, zs + d], 1), cols, "facade")


def trees_line(b: WorldBuilder, rng, xs, zs, scale=(0.8, 1.2)) -> None:
    t = models.tree()
    for x, z in zip(xs, zs):
        b.stamp(t, (x, 0, z), rng.uniform(0, 6.28), c1=t.palette1[rng.integers(len(t.palette1))],
                c2="#6b4a2e", scale=rng.uniform(*scale))


def lane_marks(b: WorldBuilder, x: float, z0: float, z1: float, color="#e8e8e0", dash=6, gapl=6, w=0.25,
               along="z") -> None:
    zs = np.arange(z0, z1, dash + gapl)
    if along == "z":
        b.boxes(np.stack([np.full_like(zs, x - w / 2), np.full_like(zs, 0.06), zs], 1),
                np.stack([np.full_like(zs, x + w / 2), np.full_like(zs, 0.08), zs + dash], 1), rgb(color))
    else:
        b.boxes(np.stack([zs, np.full_like(zs, 0.06), np.full_like(zs, x - w / 2)], 1),
                np.stack([zs + dash, np.full_like(zs, 0.08), np.full_like(zs, x + w / 2)], 1), rgb(color))


def hills(b: WorldBuilder, rng, x0, x1, z0, z1, cell=10.0, height=110, color="#4a6a3e") -> None:
    nx, nz = int((x1 - x0) / cell), int((z1 - z0) / cell)
    nse = _value_noise2(nz, nx, 6, rng)
    xs = x0 + np.arange(nx) * cell
    zs = z0 + np.arange(nz) * cell
    gx, gz = np.meshgrid(xs, zs)
    cx = (gx - (x0 + x1) / 2) / ((x1 - x0) / 2)
    cz = (gz - (z0 + z1) / 2) / ((z1 - z0) / 2)
    dome = np.clip(1 - (cx ** 2 + cz ** 2), 0, 1)
    h = (dome * 0.8 + nse * 0.4) * height
    keep = h > 2
    mins = np.stack([gx[keep], np.zeros(keep.sum()), gz[keep]], 1)
    maxs = np.stack([gx[keep] + cell, h[keep], gz[keep] + cell], 1)
    shade = 0.85 + 0.25 * rng.random(keep.sum())
    b.boxes(mins, maxs, rgb(color)[None, :] * shade[:, None], "grass")


# ======================================================================== templates
def plaza(rng, gate: bool = True, mountain: bool = True, **_) -> Built:
    b = WorldBuilder()
    b.box((-500, -1, -300), (500, 0, 1200), "#7d7a74", "asphalt")
    b.box((-30, 0, -40), (30, 0.15, 420), "#cfc6b6", "stone")            # 광장 바닥
    b.box((-30, 0.15, -40), (-28, 0.4, 420), "#a89f90", "stone")         # 광장 경계석
    b.box((28, 0.15, -40), (30, 0.4, 420), "#a89f90", "stone")
    for sx in (-1, 1):
        b.box((sx * 30 if sx > 0 else -52, 0, -300), (52 if sx > 0 else -30, 0.05, 1200), "#55575c", "asphalt")
        lane_marks(b, sx * 41, -300, 1200)
        b.box((52 if sx > 0 else -58, 0, -300), (58 if sx > 0 else -52, 0.25, 1200), "#b8b0a2", "stone")
        building_row(b, rng, sx * 58, sx * 130, -300, 1100, h_range=(18, 75))
    trees_line(b, rng, [-25] * 26 + [25] * 26, list(np.arange(-20, 400, 16.5)) * 2, (0.7, 1.0))
    if gate:   # 성문: 석축 + 홍예문 3개 + 누각 + 2층 기와지붕
        b.box((-38, 0, 400), (38, 11, 414), "#c2baa8", "stone")
        for x in (-12, 0, 12):
            b.box((x - 3.2, 0, 399.6), (x + 3.2, 6.5, 400.2), "#2c2a2a", "flat")
        b.box((-24, 11, 402), (24, 16, 412), "#8a3a2a", "stone")
        b.box((-30, 16, 398), (30, 17.6, 416), "#3a3d44", "stone")
        b.box((-26, 17.6, 400), (26, 19.6, 414), "#454952", "stone")
        b.box((-22, 19.6, 405.5), (22, 20.6, 408.5), "#2e3036", "stone")
    if mountain:
        hills(b, rng, -260, 260, 470, 720, height=140)
    skyline(b, rng, -500, 500, 760, 1100, n=90, h_range=(30, 140))
    shots = {
        "forward": {"from": [0, 7, -30], "to": [0, 7, 40], "target": [0, 5, 220], "target_to": [0, 5, 280]},
        "aerial": {"from": [0, 75, -90], "to": [0, 58, 30], "target": [0, 0, 200], "target_to": [0, 0, 260]},
        "low": {"from": [-12, 1.2, 20], "to": [-12, 1.2, 48], "target": [8, 0.8, 120], "target_to": [8, 0.8, 150]},
        "rise": {"from": [0, 1.6, 10], "to": [0, 45, 40], "target": [0, 2, 150], "target_to": [0, 0, 230]},
        "gate": {"from": [0, 6, 250], "to": [0, 9, 330], "target": [0, 12, 410]},
    }
    areas = {"plaza": [-26, -20, 26, 390], "road_l": [-50, -100, -32, 600], "road_r": [32, -100, 50, 600],
             "sidewalk_l": [-58, -100, -52, 600], "sidewalk_r": [52, -100, 58, 600]}
    return Built(b, areas, shots, shadow_radius=120, sun_azimuth=40)


def boulevard(rng, **_) -> Built:
    b = WorldBuilder()
    b.box((-300, -1, -200), (300, 0, 900), "#6d6a66", "asphalt")
    b.box((-8, 0, -200), (8, 0.05, 900), "#4f5156", "asphalt")
    lane_marks(b, 0, -200, 900, color="#e2c040", dash=900, gapl=1, w=0.3)
    lane_marks(b, -4, -200, 900); lane_marks(b, 4, -200, 900)
    for sx in (-1, 1):
        b.box((-13 if sx < 0 else 8, 0, -200), (-8 if sx < 0 else 13, 0.22, 900), "#b4ab9c", "stone")
        building_row(b, rng, sx * 13, sx * 70, -200, 860, h_range=(35, 130), w_range=(14, 30), gap=(0, 1.5))
        zs = np.arange(-180, 880, 24)                                     # 가로등
        b.boxes(np.stack([np.full_like(zs, sx * 9.2 - 0.12), np.zeros_like(zs), zs], 1),
                np.stack([np.full_like(zs, sx * 9.2 + 0.12), np.full_like(zs, 6.5), zs + 0.24], 1), rgb("#3a3c40"))
        b.boxes(np.stack([np.full_like(zs, sx * 9.2 - 0.6), np.full_like(zs, 6.4), zs - 0.2], 1),
                np.stack([np.full_like(zs, sx * 9.2 + 0.6), np.full_like(zs, 6.7), zs + 0.44], 1), rgb("#fff1c4"), "glow")
    shots = {
        "forward": {"from": [0, 2.2, 0], "to": [0, 2.2, 60], "target": [0, 12, 320], "target_to": [0, 12, 380]},
        "sidewalk": {"from": [-10.5, 1.6, 10], "to": [-10.5, 1.6, 40], "target": [-7, 1.6, 200], "target_to": [-7, 1.6, 230]},
        "canyon": {"from": [0, 70, -40], "to": [0, 64, 40], "target": [0, 0, 220], "target_to": [0, 0, 280]},
        "up": {"from": [0, 1.8, 40], "to": [0, 3, 46], "target": [0, 60, 120], "target_to": [0, 80, 140]},
    }
    areas = {"road": [-7, -100, 7, 700], "sidewalk_l": [-12.6, -100, -8.6, 700],
             "sidewalk_r": [8.6, -100, 12.6, 700]}
    return Built(b, areas, shots, shadow_radius=110, sun_azimuth=20)


def crossroad(rng, **_) -> Built:
    b = WorldBuilder()
    b.box((-400, -1, -400), (400, 0, 400), "#55575c", "asphalt")
    R, S = 14.0, 20.0     # 도로 반폭, 보도 바깥 경계
    for sx in (-1, 1):
        for sz in (-1, 1):   # 네 귀퉁이 블록: 보도 + 건물
            x0, x1 = sorted([sx * R, sx * 300]); z0, z1 = sorted([sz * R, sz * 300])
            b.box((x0, 0, z0), (x1, 0.25, z1), "#b9b1a3", "stone")
            bx0, bx1 = sorted([sx * S, sx * 300]); bz0, bz1 = sorted([sz * S, sz * 300])
            n = 26
            xs = rng.uniform(bx0, bx1 - 25, n); zs = rng.uniform(bz0, bz1 - 25, n)
            ws = rng.uniform(14, 28, n); ds = rng.uniform(14, 28, n); hs = rng.uniform(12, 55, n)
            cols = np.stack([rgb(CITY[i]) for i in rng.integers(0, len(CITY), n)])
            b.boxes(np.stack([xs, np.zeros(n), zs], 1), np.stack([xs + ws, hs, zs + ds], 1), cols, "facade")
            b.boxes(np.stack([xs + 2, hs, zs + 2], 1), np.stack([xs + ws - 2, hs + 0.6, zs + ds - 2], 1),
                    cols * 0.8, "stone")
    white = rgb("#ecebe4")
    for d in (-1, 1):        # 횡단보도 4개
        xs = np.arange(-R + 0.6, R - 0.6, 1.6)
        b.boxes(np.stack([xs, np.full_like(xs, 0.06), np.full_like(xs, d * 16.5 - 2.5)], 1),
                np.stack([xs + 0.8, np.full_like(xs, 0.08), np.full_like(xs, d * 16.5 + 2.5)], 1), white)
        b.boxes(np.stack([np.full_like(xs, d * 16.5 - 2.5), np.full_like(xs, 0.06), xs], 1),
                np.stack([np.full_like(xs, d * 16.5 + 2.5), np.full_like(xs, 0.08), xs + 0.8], 1), white)
        b.box((0.3, 0.06, d * 20) if d > 0 else (0.3, 0.06, -300), (0.6, 0.08, 300) if d > 0 else (0.6, 0.08, -20), "#e2c040")
        b.box((-0.6, 0.06, d * 20) if d > 0 else (-0.6, 0.06, -300), (-0.3, 0.08, 300) if d > 0 else (-0.3, 0.08, -20), "#e2c040")
        b.box((d * 20, 0.06, 0.3) if d > 0 else (-300, 0.06, 0.3), (300, 0.08, 0.6) if d > 0 else (-20, 0.08, 0.6), "#e2c040")
        b.box((d * 20, 0.06, -0.6) if d > 0 else (-300, 0.06, -0.6), (300, 0.08, -0.3) if d > 0 else (-20, 0.08, -0.3), "#e2c040")
        for off in (-7, 7):
            lane_marks(b, off, 20 if d > 0 else -300, 300 if d > 0 else -20)
            lane_marks(b, off, 20 if d > 0 else -300, 300 if d > 0 else -20, along="x")
    shots = {
        "top": {"type": "topdown", "center": [0, 0, 0], "height": 95, "height_to": 82},
        "top_drift": {"type": "topdown", "center": [0, 0, -14], "center_to": [0, 0, 14], "height": 85},
        "street": {"from": [-6, 1.7, -60], "to": [-6, 1.7, -35], "target": [0, 2, 40]},
    }
    areas = {
        "lane_s": [1.5, 22, 12.5, 150], "lane_n": [-12.5, -150, -1.5, -22],
        "lane_e": [22, -12.5, 150, -1.5], "lane_w": [-150, 1.5, -22, 12.5],
        "crosswalk_n": [-13, 14.5, 13, 18.5], "crosswalk_s": [-13, -18.5, 13, -14.5],
        "crosswalk_e": [14.5, -13, 18.5, 13], "crosswalk_w": [-18.5, -13, -14.5, 13],
        "corner_ne": [15, 15, 40, 40], "corner_nw": [-40, 15, -15, 40],
        "corner_se": [15, -40, 40, -15], "corner_sw": [-40, -40, -15, -15],
    }
    return Built(b, areas, shots, shadow_radius=90, sun_azimuth=60)


def river_bridge(rng, **_) -> Built:
    b = WorldBuilder()
    b.box((-1200, -3, -40), (1200, 0, 430), "#5d8fcf", "water")
    b.box((-1200, -1, -400), (1200, 2.0, -40), "#9c9a8e", "stone")       # 이쪽 강변
    b.box((-1200, 0, -48), (1200, 2.4, -40), "#b5ae9e", "stone")
    b.box((-1200, -1, 430), (1200, 3.0, 1000), "#8e8c84", "stone")       # 건너편
    skyline(b, rng, -900, 900, 470, 900, n=260, h_range=(20, 110))
    Z0, Z1, DECK = 100.0, 112.0, 15.0
    b.box((-900, DECK - 1.2, Z0), (900, DECK, Z1), "#b8b8b4", "stone")   # 상판
    for zr in (104.0, 108.0):                                             # 레일
        b.box((-900, DECK, zr - 0.8), (900, DECK + 0.15, zr - 0.6), "#3a3a3e")
        b.box((-900, DECK, zr + 0.6), (900, DECK + 0.15, zr + 0.8), "#3a3a3e")
    xs = np.arange(-900, 900, 60.0)                                       # 교각
    b.boxes(np.stack([xs - 2.5, np.full_like(xs, -3), np.full_like(xs, Z0 + 1)], 1),
            np.stack([xs + 2.5, np.full_like(xs, DECK - 1.2), np.full_like(xs, Z1 - 1)], 1), rgb("#a6a39a"), "stone")
    # 트러스: 위·아래 현재 + 수직재 + 계단식 사재 (복셀 느낌)
    steel = rgb("#c9ccd0")
    for z in (Z0, Z1 - 0.6):
        b.box((-900, DECK + 7.5, z), (900, DECK + 8.1, z + 0.6), "#c9ccd0")
        b.box((-900, DECK, z), (900, DECK + 0.6, z + 0.6), "#c9ccd0")
        px = np.arange(-900, 900, 10.0)
        b.boxes(np.stack([px, np.full_like(px, DECK), np.full_like(px, z)], 1),
                np.stack([px + 0.5, np.full_like(px, DECK + 7.5), np.full_like(px, z + 0.6)], 1), steel)
        steps = np.linspace(0, 1, 14)
        dx = (px[:, None] + steps[None, :] * 10).ravel()
        dy = (DECK + steps[None, :] * 7.2 + np.zeros_like(px)[:, None]).ravel()
        dz = np.full_like(dx, z)
        b.boxes(np.stack([dx, dy, dz], 1), np.stack([dx + 0.8, dy + 0.6, dz + 0.6], 1), steel)
    px = np.arange(-900, 900, 10.0)                                        # 상부 가로재
    b.boxes(np.stack([px, np.full_like(px, DECK + 7.5), np.full_like(px, Z0)], 1),
            np.stack([px + 0.4, np.full_like(px, DECK + 8.1), np.full_like(px, Z1)], 1), steel)
    b.box((-1200, 6.0, 300), (1200, 7.0, 318), "#a9a8a2", "stone")         # 먼 자동차 다리
    xs2 = np.arange(-1200, 1200, 45.0)
    b.boxes(np.stack([xs2, np.full_like(xs2, -3), np.full_like(xs2, 304)], 1),
            np.stack([xs2 + 3, np.full_like(xs2, 6), np.full_like(xs2, 314)], 1), rgb("#9a978e"), "stone")
    trees_line(b, rng, rng.uniform(-300, 300, 40), rng.uniform(-200, -60, 40))
    shots = {
        "side": {"from": [-60, 13, 40], "to": [10, 15, 46], "target": [40, 15, 106], "target_to": [80, 15, 106]},
        "along": {"from": [-150, 27, 88], "to": [-70, 25, 90], "target": [100, 15, 106], "target_to": [140, 15, 106]},
        "aerial": {"from": [-220, 90, -80], "to": [-130, 75, -30], "target": [0, 10, 160], "target_to": [40, 10, 180]},
        "bank": {"from": [0, 4.5, -85], "to": [30, 4.5, -85], "target": [0, 12, 106], "target_to": [30, 12, 106]},
    }
    areas = {"track_1": [-900, 104, 900, 104], "track_2": [-900, 108, 900, 108],
             "bank": [-120, -72, 150, -50], "deck": [-300, 101, 300, 111]}
    built = Built(b, areas, shots, shadow_radius=140, sun_azimuth=-30)
    return built


def apartments(rng, rows: int = 8, cols: int = 7, **_) -> Built:
    b = WorldBuilder()
    b.box((-700, -1, -400), (700, 0, 1000), "#5f8f45", "grass")
    for z in np.arange(-60, 60 * rows, 60.0) - 22:                          # 단지 내 도로
        b.box((-600, 0, z), (600, 0.05, z + 7), "#5a5c60", "asphalt")
    whites = ["#e8e3d8", "#ddd6c8", "#d2d8de", "#e4ddd0", "#cfc8bb"]
    mins, maxs, cols_ = [], [], []
    for i in range(-(cols // 2), cols - cols // 2):
        for j in range(rows):
            x, z = i * 75.0, j * 60.0
            h = rng.choice([45, 51, 57, 63, 69])
            mins.append((x - 27, 0, z)); maxs.append((x + 27, h, z + 13))
            cols_.append(rgb(whites[rng.integers(len(whites))]))
    mins, maxs, cols_ = np.array(mins), np.array(maxs), np.array(cols_)
    b.boxes(mins, maxs, cols_, "facade")
    b.boxes(mins + [20, 0, 2] + [0, 0, 0], np.stack([mins[:, 0] + 28, maxs[:, 1] + 4, mins[:, 2] + 11], 1),
            cols_ * 0.9, "stone")                                           # 옥탑
    n = 260
    trees_line(b, rng, rng.uniform(-280, 280, n), rng.uniform(-40, 60 * rows, n), (0.8, 1.3))
    shots = {
        "aerial": {"from": [-160, 100, -150], "to": [-70, 88, -95], "target": [60, 0, 170], "target_to": [100, 0, 210]},
        "low": {"from": [37, 1.6, -30], "to": [37, 1.6, 5], "target": [37, 10, 160]},
        "rise": {"from": [37, 2, -20], "to": [37, 70, -5], "target": [37, 20, 100], "target_to": [37, 0, 200]},
    }
    areas = {"lawn": [-300, -40, 300, 60 * rows], "path": [33, -40, 42, 60 * rows]}
    return Built(b, areas, shots, shadow_radius=130, sun_azimuth=25)


def room(rng, wall: str = "#d8cfbf", floor: str = "#9a6a40", window_sky: str = "#22305e", **_) -> Built:
    b = WorldBuilder()
    W, D, H = 3.0, 2.6, 2.9
    b.box((-W, -0.1, -D - 2), (W, 0.0, D), floor, "stone")
    for x in np.arange(-W, W, 0.5):                                         # 마루 줄눈
        b.box((x, 0.0, -D - 2), (x + 0.03, 0.005, D), "#6e4a2c")
    b.box((-W - 0.1, 0, -D - 2), (-W, H, D), wall, "stone")                  # 왼쪽 벽
    b.box((W, 0, -D - 2), (W + 0.1, H, D), wall, "stone")                    # 오른쪽 벽
    b.box((-W, H, -D - 2), (W, H + 0.1, D), "#e8e2d6", "stone")              # 천장
    wx0, wx1, wy0, wy1 = -1.6, 0.4, 1.0, 2.2                                 # 창문 뚫린 뒷벽
    b.box((-W, 0, D), (W, wy0, D + 0.1), wall, "stone")
    b.box((-W, wy1, D), (W, H, D + 0.1), wall, "stone")
    b.box((-W, wy0, D), (wx0, wy1, D + 0.1), wall, "stone")
    b.box((wx1, wy0, D), (W, wy1, D + 0.1), wall, "stone")
    b.box((wx0, wy0, D + 0.08), (wx1, wy1, D + 0.12), window_sky, "glow")
    for k in range(9 if rgb(window_sky).sum() < 1.2 else 0):                 # 창밖 별 (밤 하늘일 때만)
        sx, sy = rng.uniform(wx0 + 0.1, wx1 - 0.1), rng.uniform(wy0 + 0.1, wy1 - 0.1)
        b.box((sx, sy, D + 0.07), (sx + 0.04, sy + 0.04, D + 0.075), "#ffffff", "glow")
    b.box((wx0 - 0.08, wy0 - 0.08, D - 0.04), (wx1 + 0.08, wy0, D + 0.02), "#f2efe8")   # 창틀
    b.box((-0.65, wy0, D - 0.04), (-0.55, wy1, D + 0.02), "#f2efe8")
    b.box((-0.6, H - 0.06, -0.6), (0.6, H, 0.6), "#fff6dc", "glow")         # 천장 조명
    b.stamp(models.bed(), (1.7, 0, 1.45), 0.0, c1="#4a78c8", c2="#f2f2f2")
    b.box((-2.9, 0, 1.6), (-1.5, 0.75, 2.5), "#8a5a3a", "stone")            # 책상
    b.box((-2.4, 0.75, 2.0), (-2.2, 1.15, 2.2), "#ffd27a", "glow")          # 스탠드
    b.box((-1.2, 0.0, -0.8), (1.2, 0.02, 0.8), "#c86a5a", "stone")          # 러그
    shots = {
        "inside": {"from": [2.6, 1.7, -3.6], "to": [2.2, 1.6, -3.0], "target": [0.0, 0.9, 1.4], "fov": 60},
        "close": {"from": [0.6, 1.2, -1.0], "to": [0.8, 1.1, -0.5], "target": [1.7, 0.8, 1.2], "fov": 50},
    }
    areas = {"floor": [-2.5, -1.5, 2.5, 1.0], "bed": [1.0, 0.8, 2.4, 1.9]}
    lighting = {"sun_col": [0.0, 0.0, 0.0], "amb_sky": [0.86, 0.78, 0.64], "amb_ground": [0.52, 0.44, 0.36],
                "fog_density": 0.0}
    return Built(b, areas, shots, shadow_radius=8, sun_azimuth=0, lighting=lighting)


def field(rng, **_) -> Built:
    b = WorldBuilder()
    b.box((-800, -1, -200), (800, 0, 1200), "#5f9a45", "grass")
    b.box((-3, 0, -200), (3, 0.04, 400), "#c8bfa8", "stone")                # 산책로
    skyline(b, rng, -700, 700, 420, 750, n=220, h_range=(25, 130))
    trees_line(b, rng, np.concatenate([rng.uniform(-300, -60, 70), rng.uniform(60, 300, 70)]),
               rng.uniform(-50, 380, 140), (0.9, 1.4))
    shots = {
        "forward": {"from": [0, 3, -15], "to": [0, 3, 30], "target": [0, 8, 320], "target_to": [0, 8, 360]},
        "low": {"from": [0, 0.6, 0], "to": [0, 0.6, 22], "target": [0, 1.2, 100], "target_to": [0, 1.2, 130]},
        "aerial": {"from": [0, 60, -70], "to": [0, 50, -10], "target": [0, 0, 140], "target_to": [0, 0, 180]},
    }
    areas = {"field": [-60, 0, 60, 200], "near": [-15, 5, 15, 45], "path": [-2.5, -50, 2.5, 300]}
    return Built(b, areas, shots, shadow_radius=110, sun_azimuth=50)


def planet(rng, surface: str = "mars", **_) -> Built:
    b = WorldBuilder()
    cell = 3.0
    x0, x1, z0, z1 = -180, 180, -80, 460
    nx, nz = int((x1 - x0) / cell), int((z1 - z0) / cell)
    h = _value_noise2(nz, nx, 18, rng) * 7 + _value_noise2(nz, nx, 5, rng) * 1.5
    xs, zs = x0 + np.arange(nx) * cell, z0 + np.arange(nz) * cell
    gx, gz = np.meshgrid(xs, zs)
    for _ in range(14):                                                      # 크레이터
        cx, cz, r = rng.uniform(x0, x1), rng.uniform(40, z1), rng.uniform(8, 30)
        d = np.sqrt((gx - cx) ** 2 + (gz - cz) ** 2) / r
        h += np.where(d < 1, -(1 - d ** 2) * r * 0.25, 0) + np.where((d > 0.85) & (d < 1.25), 1.6, 0)
    flat = (np.abs(gx) < 30) & (gz > -20) & (gz < 120)                       # 착륙 지대는 평평하게
    h = np.where(flat, h * 0.15 + 1.0, h)
    base = ["#b5643c", "#9c5233", "#c47a4c"] if surface == "mars" else ["#9a9a9c", "#808083", "#b0b0b2"]
    cols = np.stack([rgb(base[i]) for i in rng.integers(0, 3, h.size)]) * (0.9 + 0.1 * rng.random((h.size, 1)))
    b.boxes(np.stack([gx.ravel(), np.full(h.size, -12.0), gz.ravel()], 1),
            np.stack([gx.ravel() + cell, h.ravel(), gz.ravel() + cell], 1), cols, "stone")
    n = 120                                                                  # 바위
    rx, rz = rng.uniform(-80, 80, n), rng.uniform(-10, 300, n)
    ri = np.clip(((rz - z0) / cell).astype(int), 0, nz - 1), np.clip(((rx - x0) / cell).astype(int), 0, nx - 1)
    ry = h[ri]
    rs = rng.uniform(0.4, 1.6, n)
    b.boxes(np.stack([rx, ry - 0.2, rz], 1), np.stack([rx + rs, ry + rs * 0.7, rz + rs], 1),
            rgb(base[1]) * 0.8, "stone")
    # 착륙선 (원작 디자인): 금박 하강단 + 흰 상승단 + 계단식 다리 + 안테나
    gold, gold2 = "#d9b44a", "#b8922e"
    b.box((-3.2, 1.6, 37.8), (3.2, 4.4, 44.2), gold, "stone")
    b.box((-3.6, 2.0, 38.6), (-3.2, 4.0, 43.4), gold2, "stone")
    b.box((3.2, 2.0, 38.6), (3.6, 4.0, 43.4), gold2, "stone")
    b.box((-2.4, 2.0, 37.4), (2.4, 4.0, 37.8), gold2, "stone")
    b.box((-2.6, 4.4, 38.6), (2.6, 7.2, 43.4), "#e6e6e2", "stone")
    b.box((-2.0, 7.2, 39.2), (2.0, 8.0, 42.8), "#cfcfcb", "stone")
    b.box((-1.0, 5.4, 38.5), (1.0, 6.6, 38.6), "#1d2b5a", "glass")      # 창
    b.box((-0.9, 1.6, 37.3), (0.9, 3.4, 37.4), "#2a2a2a")                # 해치
    b.box((-0.7, 0.4, 35.6), (0.7, 0.5, 37.3), "#9a9a9a")                # 사다리 발판
    for sx in (-1, 1):                                                   # 계단식 다리 4개
        for sz in (-1, 1):
            for k in range(5):
                x = sx * (3.0 + k * 0.45); z = 41 + sz * (3.0 + k * 0.45); y = 2.6 - k * 0.55
                b.box((x - 0.18, y - 0.6, z - 0.18), (x + 0.18, y, z + 0.18), "#9c9ca0")
            b.box((sx * 5.3 - 0.5, 0.3, 41 + sz * 5.3 - 0.5), (sx * 5.3 + 0.5, 0.45, 41 + sz * 5.3 + 0.5), "#8a8a8e")
    b.box((1.2, 8.0, 40.6), (1.35, 9.6, 40.75), "#dddddd")              # 안테나
    b.box((0.6, 9.6, 40.0), (1.95, 9.75, 41.35), "#eeeeee")
    b.box((6, 0.5, 34), (6.12, 5.5, 34.12), "#dddddd")                  # 깃대 + 깃발
    b.box((6.12, 4.4, 34.04), (8.2, 5.5, 34.1), "#e8e8e8", "glow")
    def height_at(x, z):
        i = np.clip(((np.asarray(z) - z0) / cell).astype(int), 0, nz - 1)
        j = np.clip(((np.asarray(x) - x0) / cell).astype(int), 0, nx - 1)
        return h[i, j]
    shots = {
        "forward": {"from": [0, 4, -15], "to": [0, 4, 25], "target": [0, 2, 220], "target_to": [0, 2, 260]},
        "aerial": {"from": [0, 70, -70], "to": [0, 55, -10], "target": [0, 0, 160], "target_to": [0, 0, 200]},
        "lander": {"type": "orbit", "center": [0, 1, 41], "radius": 17, "height": 3.2, "from_deg": 150,
                   "to_deg": 195, "look_height": 3.8},
    }
    areas = {"surface": [-25, -10, 25, 110], "near_lander": [-12, 28, 12, 54], "front_of_lander": [-2.6, 29.5, 2.6, 33], "__height__": height_at}
    return Built(b, areas, shots, shadow_radius=110, sky="mars" if surface == "mars" else "space",
                 sun_azimuth=60)


TEMPLATES = {"plaza": plaza, "boulevard": boulevard, "crossroad": crossroad, "river_bridge": river_bridge,
             "apartments": apartments, "room": room, "field": field, "planet": planet}


def build_template(world: dict, rng) -> Built:
    name = world.get("template", "plaza")
    if name not in TEMPLATES:
        raise ValueError(f"알 수 없는 템플릿: {name} (가능: {', '.join(TEMPLATES)})")
    params = dict(world.get("params", {}))
    return TEMPLATES[name](rng, **params)
