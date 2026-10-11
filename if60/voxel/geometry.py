"""상자(box) 기반 복셀 지오메트리.

세계는 '상자 수천 개'로 짓는다. 상자 N개를 numpy로 한 번에 삼각형으로 바꾸므로 빠르다.
꼭짓점 형식: pos(3) normal(3) color(3) material(1) → float32 10개
"""
from __future__ import annotations

import math

import numpy as np

from ..pixelate import hex_to_rgb

# 단위 정육면체의 6면: (법선, 네 꼭짓점)
_FACES = np.array([
    [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[0, -1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    [[0, 0, -1], [0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
], dtype=np.float32)
_NORMALS = _FACES[:, 0]          # (6,3)
_CORNERS = _FACES[:, 1:]         # (6,4,3)
_TRI = np.array([0, 1, 2, 0, 2, 3])
_TRI_CORNERS = _CORNERS[:, _TRI]  # (6,6,3)

MAT = {"flat": 0, "stone": 1, "facade": 2, "water": 3, "asphalt": 4, "glow": 5, "grass": 6, "glass": 7}


def rgb(c) -> np.ndarray:
    """'#rrggbb' 또는 (r,g,b) 0~1 → float32 (3,)"""
    if isinstance(c, str):
        return np.array(hex_to_rgb(c), np.float32) / 255.0
    return np.asarray(c, np.float32)


def box_vertices(mins: np.ndarray, maxs: np.ndarray, faces: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """상자 N개 → (N*F*6, 3) 위치, (N*F*6, 3) 법선"""
    mins = np.asarray(mins, np.float32).reshape(-1, 3)
    maxs = np.asarray(maxs, np.float32).reshape(-1, 3)
    fidx = np.arange(6) if faces is None else np.asarray(faces)
    size = maxs - mins
    pos = mins[:, None, None, :] + _TRI_CORNERS[None, fidx] * size[:, None, None, :]
    nrm = np.broadcast_to(_NORMALS[fidx][None, :, None, :], pos.shape)
    return pos.reshape(-1, 3), np.ascontiguousarray(nrm).reshape(-1, 3)


class WorldBuilder:
    """정적인 세계(땅, 건물, 다리…)를 모은다. build()로 float32 꼭짓점 배열을 만든다."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []

    def boxes(self, mins, maxs, colors, mat: str | int = "flat", skip_bottom: bool = True) -> None:
        mins = np.asarray(mins, np.float32).reshape(-1, 3)
        maxs = np.asarray(maxs, np.float32).reshape(-1, 3)
        n = len(mins)
        if n == 0:
            return
        faces = [0, 1, 2, 4, 5] if skip_bottom else None
        nf = 5 if skip_bottom else 6
        pos, nrm = box_vertices(mins, maxs, faces)
        cols = np.asarray(colors, np.float32)
        if cols.ndim == 1:
            cols = np.broadcast_to(cols, (n, 3))
        cols = np.repeat(cols, nf * 6, axis=0)
        m = MAT[mat] if isinstance(mat, str) else int(mat)
        mats = np.full((len(pos), 1), m, np.float32)
        self.chunks.append(np.hstack([pos, nrm, cols, mats]).astype(np.float32))

    def box(self, mn, mx, color, mat: str | int = "flat", skip_bottom: bool = True) -> None:
        self.boxes([mn], [mx], rgb(color), mat, skip_bottom)

    def stamp(self, model: "Model", pos, yaw: float = 0.0, c1="#ffffff", c2="#888888",
              scale: float = 1.0) -> None:
        """인스턴싱 모델을 정적 세계에 '도장' 찍듯 박아 넣는다 (나무, 가구 등)."""
        v = model.data.copy()
        c, s = math.cos(yaw), math.sin(yaw)
        p = v[:, 0:3] * scale
        n = v[:, 3:6]
        rot = lambda a: np.stack([c * a[:, 0] + s * a[:, 2], a[:, 1], -s * a[:, 0] + c * a[:, 2]], 1)
        p, n = rot(p) + np.asarray(pos, np.float32), rot(n)
        slot = v[:, 9]
        col = np.where(slot[:, None] < 0.5, v[:, 6:9], np.where(slot[:, None] < 1.5, rgb(c1), rgb(c2)))
        self.chunks.append(np.hstack([p, n, col, v[:, 10:11]]).astype(np.float32))

    def build(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, 10), np.float32)
        return np.vstack(self.chunks)


class Model:
    """군중용 모델. 꼭짓점 형식: pos(3) normal(3) color(3) slot(1) material(1)
    slot 0=고정색, 1=개체 주색(옷·털), 2=개체 보조색(바지·무늬)."""

    def __init__(self, name: str, palette1: list[str], palette2: list[str]):
        self.name, self.palette1, self.palette2 = name, palette1, palette2
        self._parts: list[np.ndarray] = []

    def part(self, mn, mx, color: str | int, mat: str = "flat") -> "Model":
        pos, nrm = box_vertices([mn], [mx])
        if isinstance(color, int):
            slot, col = float(color), np.zeros(3, np.float32)
        else:
            slot, col = 0.0, rgb(color)
        k = len(pos)
        self._parts.append(np.hstack([
            pos, nrm, np.broadcast_to(col, (k, 3)), np.full((k, 1), slot),
            np.full((k, 1), MAT[mat])]).astype(np.float32))
        return self

    @property
    def data(self) -> np.ndarray:
        return np.vstack(self._parts)
