"""픽셀화 후처리 — AI 이미지/영상을 '진짜 도트'로 바꾸는 핵심 모듈.

처리 순서
  1) 9:16 중앙 크롭
  2) 박스 평균으로 저해상도 축소 (예: 180x320) → AI 특유의 잔노이즈 제거
  3) 고정 팔레트로 색 양자화 (선택: 디더링)
  4) 최근접 보간으로 정수배 확대 → 픽셀 격자가 정확히 맞음

영상은 여기에 '시간축 안정화'를 더한다. 프레임 간 원본 색 차이가 작은 픽셀은
이전 프레임의 팔레트 인덱스를 그대로 유지해서 AI 영상의 지글거림(플리커)을 없앤다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- palette
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def palette_array(colors: list[str]) -> np.ndarray:
    return np.array([hex_to_rgb(c) for c in colors], dtype=np.float32)


def _mediancut(rgb: np.ndarray, n: int) -> np.ndarray:
    img = Image.fromarray(rgb.reshape(-1, 1, 3).astype(np.uint8))
    q = img.quantize(colors=n, method=Image.Quantize.MEDIANCUT, kmeans=2)
    used = sorted(set(np.asarray(q).ravel().tolist()))
    return np.array(q.getpalette()[: 256 * 3], dtype=np.float32).reshape(-1, 3)[used]


def auto_palette(img: Image.Image, n: int = 24) -> np.ndarray:
    """이미지에서 대표색 n개를 뽑는다. 영상은 첫 프레임 기준으로 한 번만 뽑아 고정.

    median cut만 쓰면 넓은 면적 색 위주라 네온사인·불빛 같은 작은 강조색이 사라진다.
    그래서 3/4는 전체에서, 1/4는 채도·밝기가 높은 상위 픽셀에서 따로 뽑아 합친다.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    n_accent = max(2, n // 4)
    base = _mediancut(rgb, n - n_accent)
    mx, mn = rgb.max(1), rgb.min(1)
    vivid = (mx - mn) * 0.7 + mx * 0.3
    top = rgb[vivid >= np.percentile(vivid, 95)]
    accent = _mediancut(top, n_accent) if len(top) >= n_accent else np.zeros((0, 3), np.float32)
    return np.concatenate([base, accent])


# ---------------------------------------------------------------- geometry
def crop_to_aspect(img: Image.Image, w: int, h: int) -> Image.Image:
    src_w, src_h = img.size
    target = w / h
    if src_w / src_h > target:  # 너무 넓음 → 좌우 자르기
        new_w = int(src_h * target)
        x0 = (src_w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, src_h))
    new_h = int(src_w / target)
    y0 = (src_h - new_h) // 2
    return img.crop((0, y0, src_w, y0 + new_h))


def downscale(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    img = crop_to_aspect(img.convert("RGB"), *size)
    return np.asarray(img.resize(size, Image.Resampling.BOX), dtype=np.float32)


def upscale(arr: np.ndarray, scale: int) -> np.ndarray:
    """최근접 정수배 확대 (numpy repeat — PIL보다 빠르고 격자가 정확)."""
    return arr.repeat(scale, axis=0).repeat(scale, axis=1)


# ---------------------------------------------------------------- quantize
_BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                    dtype=np.float32) / 16.0 - 0.5)


def color_dist(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """(H,W,3) × (N,3) → (H,W,N) 'redmean' 지각 색차. 단순 RGB 거리보다 색상 왜곡이 적다."""
    a = rgb[..., None, :]
    b = pal[None, None, :, :]
    rbar = (a[..., 0] + b[..., 0]) / 2
    diff = a - b
    return ((2 + rbar / 256) * diff[..., 0] ** 2 + 4 * diff[..., 1] ** 2
            + (2 + (255 - rbar) / 256) * diff[..., 2] ** 2)


def nearest_index(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    return color_dist(rgb, pal).argmin(-1).astype(np.int16)


def quantize(rgb: np.ndarray, pal: np.ndarray, dither: float = 0.0) -> np.ndarray:
    """rgb(H,W,3 float) → 팔레트 인덱스(H,W).

    dither는 Bayer 순서 디더링 강도(0=없음, 24 정도가 레트로 느낌).
    오차확산(Floyd-Steinberg)은 영상에서 프레임마다 패턴이 바뀌어 지글거리므로
    위치에 고정된 Bayer 패턴만 쓴다.
    """
    if dither > 0:
        h, w = rgb.shape[:2]
        tile = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        rgb = rgb + tile[..., None] * dither
    return nearest_index(rgb, pal)


def indices_to_rgb(idx: np.ndarray, pal: np.ndarray) -> np.ndarray:
    return pal[idx].astype(np.uint8)


# ---------------------------------------------------------------- public API
def pixelate_image(img: Image.Image, size=(180, 320), palette: np.ndarray | None = None,
                   n_auto: int = 24, dither: float = 0.0) -> Image.Image:
    """한 장을 도트로. 결과는 저해상도(size) 이미지. 최종 확대는 렌더 단계에서."""
    small = downscale(img, size)
    pal = palette if palette is not None else auto_palette(Image.fromarray(small.astype(np.uint8)), n_auto)
    return Image.fromarray(indices_to_rgb(quantize(small, pal, dither), pal))


class TemporalPixelator:
    """AI 영상 프레임을 순서대로 넣으면 안정화된 도트 프레임을 돌려준다.

    플리커를 두 겹으로 막는다.
    1) hold: 원본 색 변화가 이 값(0~255 RGB 거리)보다 작은 픽셀은 이전 색을 그대로 유지.
             AI 영상의 미세 노이즈가 여기서 걸러진다.
    2) 히스테리시스: 두 팔레트 색의 경계에 걸친 픽셀은 새 색이 '확실히' 더 가까울 때만 바꾼다.
             양자화 후 생기는 깜빡임의 주원인.
    값이 클수록 안정적이지만 느린 움직임이 '뚝뚝' 끊길 수 있다. hold 18~30 권장.
    """

    def __init__(self, size=(180, 320), palette: np.ndarray | None = None,
                 n_auto: int = 24, dither: float = 0.0, hold: float = 22.0,
                 hysteresis: float = 0.6):
        self.size, self.palette, self.n_auto = size, palette, n_auto
        self.dither, self.hold, self.hysteresis = dither, hold, hysteresis
        self._prev_rgb: np.ndarray | None = None
        self._prev_idx: np.ndarray | None = None

    def __call__(self, frame: Image.Image) -> np.ndarray:
        small = downscale(frame, self.size)
        if self.palette is None:  # 첫 프레임에서 팔레트를 정하고 끝까지 고정
            self.palette = auto_palette(Image.fromarray(small.astype(np.uint8)), self.n_auto)
        work = small
        if self.dither > 0:
            h, w = small.shape[:2]
            tile = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
            work = small + tile[..., None] * self.dither
        dist = color_dist(work, self.palette)
        idx = dist.argmin(-1).astype(np.int16)
        if self._prev_idx is not None:
            # 1) 히스테리시스: 이전 색과의 거리가 새 최적 색 거리의 (1+h)배 이내면 유지
            d_new = np.take_along_axis(dist, idx[..., None].astype(np.int64), -1)[..., 0]
            d_old = np.take_along_axis(dist, self._prev_idx[..., None].astype(np.int64), -1)[..., 0]
            sticky = d_old <= d_new * (1 + self.hysteresis) + 1e-3
            idx = np.where(sticky, self._prev_idx, idx)
            # 2) hold: 원본이 거의 안 바뀐 픽셀은 무조건 유지 (기준 색도 고정해 누적 변화만 반영)
            diff = np.sqrt(((small - self._prev_rgb) ** 2).sum(-1))
            keep = diff < self.hold
            idx = np.where(keep, self._prev_idx, idx)
            small = np.where(keep[..., None], self._prev_rgb, small)
        self._prev_rgb, self._prev_idx = small, idx
        return indices_to_rgb(idx, self.palette)
