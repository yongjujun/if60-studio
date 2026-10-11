"""3D 복셀 렌더러 테스트 (GPU 없이 소프트웨어 OpenGL로도 동작)."""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from if60.voxel.renderer import VoxelRenderer, make_crowd, scene_hours  # noqa: E402
from if60.voxel.sky import time_of_day  # noqa: E402
from if60.voxel.templates import TEMPLATES  # noqa: E402


@pytest.fixture(scope="module")
def renderer():
    try:
        r = VoxelRenderer(72, 128, samples=0, shadow_size=512)
    except Exception as e:  # OpenGL 컨텍스트를 못 만드는 환경
        pytest.skip(f"OpenGL 사용 불가: {e}")
    yield r
    r.close()


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_renders(renderer, name):
    p = renderer.prepare({"template": name}, [{"agent": "cat", "count": 50, "area": [-5, -5, 5, 5]}])
    img = renderer.render(p, next(iter(p.shots.values())), 1.0, 4.0, 12.0)
    renderer.release(p)
    assert img.shape == (128, 72, 3)
    assert img.std() > 5  # 단색 화면이 아님


def test_time_of_day_changes_light():
    dawn, noon, night = time_of_day(5.5), time_of_day(12), time_of_day(22)
    assert noon.sun_dir[1] > dawn.sun_dir[1]
    assert night.night > 0.9 and noon.night == 0
    assert noon.zenith[2] > night.zenith[2]


def test_scene_hours_follows_hud_clock():
    hud = {"from": "05:00", "to": "07:00"}
    assert scene_hours(None, hud, 0, 10) == 5.0
    assert scene_hours(None, hud, 5, 10) == 6.0
    assert scene_hours("21:30", hud, 5, 10) == 21.5


def test_crowd_keeps_clear_of_camera_path():
    rng = np.random.default_rng(0)
    avoid = np.array([[0.0, z] for z in range(0, 50, 2)], np.float32)
    cr = make_crowd({"agent": "person", "count": 2000, "area": [-10, 0, 10, 50]}, {}, rng, avoid)
    assert len(cr.instances) < 2000
    inside = cr.instances[(cr.instances[:, 2] > 2) & (cr.instances[:, 2] < 46)]
    assert np.abs(inside[:, 0]).min() > 2.8  # 경로 샘플 간격(2m) 사이에서는 최소 √(3²−1²)≈2.83m
