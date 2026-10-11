import numpy as np
from PIL import Image

from if60 import procedural
from if60.pixelate import TemporalPixelator, auto_palette, pixelate_image, quantize
from if60.render import Scene, Style, compose_frame


def _noisy_frames(n=12, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:640, 0:360]
    base = np.dstack([xx / 360 * 255, yy / 640 * 255, np.full_like(xx, 140.0)])
    for _ in range(n):
        yield Image.fromarray(np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8))


def test_pixelate_image_uses_only_palette_colors():
    img = next(_noisy_frames(1))
    out = np.asarray(pixelate_image(img, (90, 160), n_auto=12))
    assert out.shape == (160, 90, 3)
    assert len(np.unique(out.reshape(-1, 3), axis=0)) <= 12


def test_auto_palette_keeps_small_vivid_accent():
    arr = np.full((200, 200, 3), (40, 40, 60), np.uint8)
    arr[5:9, 5:9] = (255, 40, 200)  # 작은 네온 포인트 (0.04% 면적)
    pal = auto_palette(Image.fromarray(arr), 8)
    assert np.min(np.abs(pal - np.array([255, 40, 200])).sum(1)) < 40


def test_temporal_stabilization_reduces_flicker():
    frames = list(_noisy_frames(12))

    def flicker(**kw):
        tp, prev, ch = TemporalPixelator((90, 160), **kw), None, []
        for f in frames:
            o = tp(f)
            if prev is not None:
                ch.append(np.any(o != prev, -1).mean())
            prev = o
        return np.mean(ch)

    assert flicker(hold=20, hysteresis=0.6) < flicker(hold=0, hysteresis=0) * 0.3


def test_compose_frame_shape_and_fade():
    style = Style.from_preset("cinematic_pixel")
    w, h = style.resolution
    big = (w + 96, h + 96)
    sc = Scene(duration=2, layers=[procedural.make("sky", *big, colors=["#ffffff"])])
    mid, _ = compose_frame(sc, 1.0, style)
    start, _ = compose_frame(sc, 0.0, style)
    assert mid.shape == (h, w, 3)
    assert mid.mean() > 250 and start.mean() < 5  # 첫 프레임은 디더 페이드로 검정


def test_quantize_fixed_palette():
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.float32)
    rgb = np.array([[[10, 10, 10], [240, 240, 240]]], np.float32)
    assert quantize(rgb, pal).tolist() == [[0, 1]]
