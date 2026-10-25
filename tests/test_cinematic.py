"""시네마틱 시뮬레이션 형식: 공원 세트 · 끌려가는 군중 · 한국어 미니멀 수치(readout) · 명조 자막 · 형식 검사."""
import copy

import numpy as np
import yaml

from if60 import overlay, planner, text
from if60 import render as R
from if60.voxel.templates import TEMPLATES

EX = "scenes/blackhole_park.yaml"


def _spec():
    return yaml.safe_load(open(EX, encoding="utf-8"))


def test_example_cinematic_episode_passes_everything():
    ytxt = open(EX, encoding="utf-8").read()
    spec = yaml.safe_load(ytxt)
    assert spec["format"] == "cinematic" and spec["look"] == "cinematic"
    assert planner.validate(spec, planner.catalog(), None) == []
    assert planner.retention_issues(spec) == []
    assert planner.smoke_test(ytxt) == []
    assert planner.korean_screen_issues(spec) == []


def test_cinematic_checks_catch_missing_pieces():
    spec = _spec()
    s2 = copy.deepcopy(spec)
    s2["look"] = "smooth"
    s2["layout"] = {"style": "simple", "title": "질문"}
    for sc in s2["scenes"]:
        sc.pop("readout", None)
    issues = " ".join(planner.retention_issues(s2))
    assert "look: cinematic" in issues and "cinema" in issues and "readout" in issues
    s3 = copy.deepcopy(spec)
    s3["scenes"][3]["readout"]["status"] = "EVERYTHING IS PULLED"
    assert any("영어" in i for i in planner.retention_issues(s3))
    s4 = copy.deepcopy(spec)
    s4["scenes"][3]["readout"] = "50m"
    assert any("readout" in e for e in planner.validate(s4, planner.catalog(), None))


def test_park_set_and_attract_crowd():
    built = TEMPLATES["park"](np.random.default_rng(0))
    assert {"low", "wide", "aerial"} <= set(built.shots) and "lawn" in built.areas
    from if60.voxel.renderer import VoxelRenderer
    vr = VoxelRenderer(90, 160, samples=0, shadow_size=256)
    prep = vr.prepare({"template": "park", "seed": 1},
                      [{"agent": "person_lying", "count": 40, "area": "lawn", "attract": {"to": [0, 20], "speed": 2}}],
                      shot="wide", look="cinematic")
    cr = [c for c, *_ in prep.crowds if c.model == "person_lying"][0]
    assert cr.sink is not None
    img = vr.render(prep, prep.shots["wide"], 0.5, 2, 12)
    vr.release(prep)
    vr.close()
    assert img.std() > 10


def test_readout_and_cinema_caption_draw():
    serif, light = text.find_serif(), text.find_light()
    img = np.full((640, 360, 3), 80, np.uint8)
    ro = {"status": "모든 것이 끌려가는 중", "label": "50m 지점의 당기는 힘",
          "value": {"from": 1, "to": 9.2e9, "format": "kr1", "unit": "g"}, "note": "몸무게가 약 6억 톤",
          "right": [{"label": "경과", "value": {"from": 0, "to": 2, "format": "{:.1f}", "unit": "µs"}},
                    {"label": "느린 화면", "big": True, "value": "200만 배", "note": "영상 1초 = 실제 0.5µs"}]}
    overlay.draw_readout(img, ro, 3.0, 4.0, serif, light, 1 / 3)
    top = img[int(640 * 0.15):int(640 * 0.32)]
    assert (top[:, :160] > 200).mean() > 0.01 and (top[:, 220:] > 200).mean() > 0.005   # 왼쪽·오른쪽 위 모두
    cap = np.full((640, 360, 3), 80, np.uint8)
    overlay.draw_cinema_caption(cap, "중심에서 50미터. 모든 것이 끌려갑니다.", serif, 1 / 3)
    assert (cap[500:600] > 200).mean() > 0.005 and (cap[:450] != 80).mean() < 0.001
    assert overlay.kr_short(9.16e9) == "92억" and overlay.kr_short(2.29e7) == "2,290만" and overlay.kr_short(217) == "217"


def test_cinema_caption_replaces_pixel_caption():
    spec = _spec()
    assert R.caption_style_of(spec) == "cinema" and R.caption_style_of({"look": "voxel"}) == "pixel"
    sc = R.Scene(duration=3, layers=[], caption="자막", caption_style="cinema")
    font = text.find_font()
    assert R._text_layer(sc, 1.0, R.Style.from_preset("voxel"), font) is None   # 도트 자막은 그리지 않음
    lay = overlay.make_layout({"style": "cinema", "title": "질문?"}, "", font, (360, 640))
    assert isinstance(lay, overlay.SimpleLayout) and lay.cinema
