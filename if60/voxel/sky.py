"""시간대별 하늘·조명. HUD 시계(05:00 → 23:30)와 연동해 새벽 분홍 → 낮 파랑 → 노을 → 밤 남색으로 바뀐다."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 시각(시) → (천정색, 지평선색, 햇빛색, 하늘 쪽 주변광, 땅 쪽 주변광, 안개 밀도, 밤 정도)
_KEYS = [
    (0.0, (0.03, 0.04, 0.10), (0.10, 0.11, 0.20), (0.16, 0.18, 0.26), (0.20, 0.22, 0.34), (0.10, 0.10, 0.15), 0.0030, 1.0),
    (4.5, (0.05, 0.06, 0.15), (0.20, 0.18, 0.30), (0.18, 0.18, 0.26), (0.24, 0.24, 0.36), (0.12, 0.12, 0.16), 0.0032, 0.9),
    (5.5, (0.30, 0.32, 0.50), (0.86, 0.66, 0.62), (0.75, 0.55, 0.45), (0.50, 0.48, 0.56), (0.30, 0.27, 0.28), 0.0026, 0.35),
    (6.5, (0.50, 0.56, 0.72), (0.94, 0.80, 0.70), (0.95, 0.75, 0.55), (0.58, 0.58, 0.64), (0.38, 0.34, 0.32), 0.0024, 0.05),
    (8.5, (0.46, 0.62, 0.86), (0.82, 0.85, 0.88), (0.90, 0.86, 0.76), (0.62, 0.66, 0.72), (0.42, 0.40, 0.37), 0.0019, 0.0),
    (12.0, (0.38, 0.60, 0.92), (0.78, 0.86, 0.94), (0.92, 0.90, 0.84), (0.64, 0.68, 0.74), (0.44, 0.42, 0.39), 0.0017, 0.0),
    (16.0, (0.42, 0.60, 0.86), (0.84, 0.84, 0.86), (0.92, 0.84, 0.70), (0.62, 0.64, 0.70), (0.42, 0.40, 0.36), 0.0025, 0.0),
    (18.2, (0.40, 0.42, 0.64), (0.98, 0.64, 0.44), (0.98, 0.58, 0.34), (0.56, 0.50, 0.54), (0.36, 0.30, 0.28), 0.0034, 0.15),
    (19.3, (0.13, 0.13, 0.32), (0.46, 0.30, 0.42), (0.36, 0.28, 0.34), (0.32, 0.30, 0.42), (0.16, 0.14, 0.18), 0.0032, 0.7),
    (21.0, (0.04, 0.05, 0.13), (0.12, 0.13, 0.24), (0.16, 0.18, 0.26), (0.22, 0.24, 0.36), (0.10, 0.10, 0.15), 0.0030, 1.0),
    (24.0, (0.03, 0.04, 0.10), (0.10, 0.11, 0.20), (0.16, 0.18, 0.26), (0.20, 0.22, 0.34), (0.10, 0.10, 0.15), 0.0030, 1.0),
]


@dataclass
class Lighting:
    zenith: np.ndarray
    horizon: np.ndarray
    sun_dir: np.ndarray
    sun_col: np.ndarray
    amb_sky: np.ndarray
    amb_ground: np.ndarray
    fog_density: float
    night: float
    sun_size: float = 0.0012
    planet_dir: np.ndarray | None = None
    planet_col: np.ndarray | None = None
    planet_size: float = 0.0


def _lerp(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def parse_time(t) -> float:
    """'05:30' → 5.5 (시)"""
    if isinstance(t, (int, float)):
        return float(t)
    h, m = str(t).split(":")
    return int(h) + int(m) / 60.0


def time_of_day(hours: float, azimuth_deg: float = 35.0) -> Lighting:
    h = hours % 24.0
    for i in range(len(_KEYS) - 1):
        if _KEYS[i][0] <= h <= _KEYS[i + 1][0]:
            a, b = _KEYS[i], _KEYS[i + 1]
            break
    t = (h - a[0]) / max(1e-6, b[0] - a[0])
    vals = [_lerp(a[k], b[k], t) for k in range(1, 6)]
    fog = a[6] * (1 - t) + b[6] * t
    night = a[7] * (1 - t) + b[7] * t
    # 해: 6시 동쪽에서 떠서 19시 서쪽으로. 밤에는 달빛 방향(높게)으로 약한 그림자.
    day_p = (h - 6.0) / 13.0
    if 0.0 <= day_p <= 1.0:
        elev = math.radians(8 + 58 * math.sin(math.pi * day_p))
        az = math.radians(azimuth_deg + 90 - 180 * day_p)
    else:
        elev, az = math.radians(55), math.radians(azimuth_deg - 30)
    sun_dir = np.array([math.cos(elev) * math.sin(az), math.sin(elev), math.cos(elev) * math.cos(az)], np.float32)
    # 밝은 재질이 하얗게 날아가지 않도록 노출 조정 (주변광 + 햇빛 합이 1을 크게 넘지 않게)
    return Lighting(zenith=vals[0], horizon=vals[1], sun_dir=sun_dir, sun_col=vals[2] * 0.62,
                    amb_sky=vals[3] * 0.74, amb_ground=vals[4] * 0.8, fog_density=fog, night=night,
                    sun_size=0.0 if night > 0.6 else 0.0012)


def preset(name: str, hours: float = 12.0) -> Lighting:
    """특수 하늘: space(검은 우주+별+지구), mars(버터스카치색 하늘)"""
    if name == "space":
        l = time_of_day(12.0)
        l.zenith = np.array([0.01, 0.01, 0.03], np.float32)
        l.horizon = np.array([0.03, 0.03, 0.06], np.float32)
        l.amb_sky = np.array([0.28, 0.28, 0.34], np.float32)
        l.amb_ground = np.array([0.10, 0.10, 0.12], np.float32)
        l.sun_col = np.array([1.05, 1.02, 0.96], np.float32)
        l.fog_density, l.night, l.sun_size = 0.0008, 1.0, 0.0015
        l.sun_dir = np.array([0.55, 0.38, -0.2], np.float32) / np.linalg.norm([0.55, 0.38, -0.2])
        l.planet_dir = np.array([-0.35, 0.42, 0.84], np.float32) / np.linalg.norm([-0.35, 0.42, 0.84])
        l.planet_col = np.array([0.30, 0.52, 0.86], np.float32)
        l.planet_size = 0.006
        return l
    if name == "mars":
        l = time_of_day(hours)
        l.zenith = np.array([0.62, 0.48, 0.36], np.float32) * (1 - 0.8 * l.night)
        l.horizon = np.array([0.86, 0.66, 0.48], np.float32) * (1 - 0.75 * l.night)
        l.amb_sky = np.array([0.62, 0.52, 0.44], np.float32) * (1 - 0.6 * l.night)
        l.amb_ground = np.array([0.40, 0.30, 0.24], np.float32) * (1 - 0.6 * l.night)
        l.fog_density = 0.0045
        return l
    return time_of_day(hours)
