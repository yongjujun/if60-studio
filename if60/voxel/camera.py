"""카메라: 행렬 계산과 장면 동안의 이동 경로(dolly, orbit, topdown)."""
from __future__ import annotations

import math

import numpy as np


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(l, r, b, t, n, f) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return m


def look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, np.float32) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-9
    if abs(np.dot(fwd, up)) > 0.999:  # 바로 아래를 볼 때
        up = np.array([0, 0, 1], np.float32)
    # 앞(+z)을 볼 때 +x가 화면 오른쪽에 오도록 오른쪽 벡터를 up×fwd로 잡는다 (좌우 직관 유지)
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right) + 1e-9
    true_up = np.cross(fwd, right)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = right, true_up, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def gl_bytes(m: np.ndarray) -> bytes:
    """numpy(행 우선) → OpenGL(열 우선) 유니폼 바이트"""
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


def _ease(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * p)


def _lerp(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def camera_at(spec: dict, t: float, duration: float) -> tuple[np.ndarray, np.ndarray, float]:
    """장면 시각 t의 (눈 위치, 바라보는 점, 화각)."""
    p = t / max(duration, 1e-6)
    p = _ease(p) if spec.get("ease", True) else min(max(p, 0.0), 1.0)
    fov = float(spec.get("fov", 55))
    kind = spec.get("type", "dolly")
    if kind == "orbit":
        c = np.asarray(spec.get("center", [0, 0, 0]), np.float32)
        a = math.radians(spec.get("from_deg", 0) + (spec.get("to_deg", 40) - spec.get("from_deg", 0)) * p)
        r, hgt = float(spec.get("radius", 30)), float(spec.get("height", 12))
        eye = c + np.array([math.sin(a) * r, hgt, math.cos(a) * r], np.float32)
        return eye, c + np.array([0, spec.get("look_height", 1.0), 0], np.float32), fov
    if kind == "topdown":
        c = _lerp(spec.get("center", [0, 0, 0]), spec.get("center_to", spec.get("center", [0, 0, 0])), p)
        h = float(spec.get("height", 80)) + (float(spec.get("height_to", spec.get("height", 80))) - float(spec.get("height", 80))) * p
        eye = c + np.array([0, h, -0.01], np.float32)
        return eye, c, fov
    eye = _lerp(spec["from"], spec.get("to", spec["from"]), p)
    tgt = _lerp(spec["target"], spec.get("target_to", spec["target"]), p)
    return eye, tgt, fov
