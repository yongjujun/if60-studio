"""방송형 연출(레이아웃·대시보드·빛나는 글자·VFX·마무리 카드)과 무음 출력 테스트."""
import json
import shutil
import subprocess

import numpy as np
import pytest
import yaml

from if60 import overlay, planner
from if60 import render as R
from if60.voxel.renderer import grow_factor, vfx_draws

EX = "scenes/lightning_power.yaml"


def test_korean_big_numbers():
    assert overlay.kr_number(150_000_000) == "1억 5,000만"
    assert overlay.kr_number(300_000_000) == "3억"
    assert overlay.kr_number(12_345) == "1만 2,345"
    assert overlay.kr_number(980) == "980"


def test_counter_rolls_up_and_uses_log_for_huge_ranges():
    spec = {"from": 0, "to": 300_000_000, "format": "kr", "unit": "V"}
    first, unit = overlay.counter_value(spec, 0.0, 5)
    last, _ = overlay.counter_value(spec, 5.0, 5)
    mid, _ = overlay.counter_value(spec, 1.0, 5)
    assert unit == "V" and first == "0" and last == "3억"
    assert "억" not in mid                         # 로그 스케일: 초반엔 자릿수가 낮게 시작
    assert overlay.counter_value({"from": 20, "to": 27700, "unit": "℃", "scale": "linear"}, 5, 5)[0] == "27,700"


def test_grow_and_vfx_timing():
    g = {"from": 1.0, "to": 2.0}
    assert grow_factor(g, 0.5) == 0 and grow_factor(g, 2.5) == 1.0 and 0 < grow_factor(g, 1.5) < 1
    assert grow_factor({"from": 0, "to": 1, "until": 3}, 4) == 0
    wave = {"type": "shockwave", "at": [0, 0, 10], "from": 1, "to": 3, "radius": [0, 40]}
    assert vfx_draws(wave, 0.5) == []
    d = vfx_draws(wave, 2.0)[0]
    assert 0 < d["scale"][0] < 40 and d["floor"] < 0.0 + 1e-6   # 땅 위 반구
    assert len(vfx_draws({"type": "mushroom", "at": [0, 0, 0], "from": 0, "to": 4, "radius": 10}, 2)) == 3


def test_example_episode_is_valid_and_renders_every_scene():
    ytxt = open(EX, encoding="utf-8").read()
    spec = yaml.safe_load(ytxt)
    assert planner.validate(spec, planner.catalog(), 45) == []
    assert planner.smoke_test(ytxt) == []


def test_validate_broadcast_fields():
    cat = planner.catalog()
    spec = yaml.safe_load(open(EX, encoding="utf-8"))
    sc = spec["scenes"][0]
    sc["vfx"] = [{"type": "laser", "at": [0, 0, 0]}]
    sc["big_text"] = [{"from": 1}]
    sc["markers"] = [{"text": "여기"}]
    spec["scenes"][-1]["card"] = {"question": "?", "options": ["하나"]}
    errs = planner.validate(spec, cat, 45)
    assert any("laser" in e or "vfx" in e for e in errs)
    assert any("big_text" in e for e in errs) and any("marker" in e for e in errs) and any("card" in e for e in errs)


def test_layout_frame_and_card(tmp_path):
    font = R.text.find_font()
    lay = overlay.Layout({"title": ["번개 한 방,", "얼마나 셀까?"]}, "", font, (360, 640))
    img = np.full((640, 360, 3), 90, np.uint8)
    out = lay.apply(img.copy(), 0.0)
    top, bottom = lay.viewport
    assert 0 < top < 640 * 0.2 and bottom > 640 * 0.85
    assert out[:top].mean() < 90                           # 어두운 상단 띠
    assert (out[top + 5:bottom - 5] == 90).all()           # 가운데 3D 화면은 그대로
    yellow = (np.abs(out[bottom:].astype(int) - overlay.YELLOW).sum(-1) < 30).mean()
    assert yellow > 0.05                                   # 하단 경고 줄무늬
    card = np.zeros((640, 360, 3), np.uint8)
    overlay.draw_card(card, {"question": "당신의 선택은?", "options": ["가", "나", "다"]}, 3.0, 6, font, 1 / 3,
                      (top, bottom))
    assert card.max() > 200


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffmpeg 필요")
def test_render_without_voice_is_silent(tmp_path):
    src = tmp_path / "s.yaml"
    src.write_text(
        "title: 무음 테스트\nstyle: voxel\nlayout: {title: [무음, 테스트]}\nscenes:\n"
        "  - duration: 1\n    caption: 안녕하세요.\n    world: {template: studio, seed: 1}\n    shot: front\n"
        "    big_text: [{text: \"×1억\"}]\n    panel: {title: 값, value: {from: 0, to: 100}}\n"
        "  - duration: 1\n    card: {question: 골라요, options: [하나, 둘]}\n", encoding="utf-8")
    out = tmp_path / "o.mp4"
    R.render(str(src), str(out), preview=True, fps=4, narrate=False)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
                                     capture_output=True, text=True).stdout)
    kinds = [s["codec_type"] for s in info["streams"]]
    assert kinds == ["video"]


def test_satire_devices_render():
    font = R.text.find_font()
    img = np.full((640, 360, 3), 200, np.uint8)
    stamped = img.copy()
    overlay.draw_big_text(stamped, {"text": "기각", "style": "stamp", "from": 0}, 1.0, 5, font, 1 / 3)
    red = (stamped[..., 0].astype(int) - stamped[..., 1] > 80).sum()
    assert red > 200                                              # 빨간 도장이 찍힘
    ch = img.copy()
    overlay.draw_chyron(ch, {"tag": "속보", "text": "정부 발표"}, 1.0, 5, font, 1 / 3)
    assert (ch != img).any(axis=-1)[int(640 * 0.815):].mean() > 0.01
    z = overlay.punch_zoom(np.arange(640 * 360 * 3, dtype=np.uint8).reshape(640, 360, 3), {"at": 0, "zoom": 1.5}, 0.5)
    assert z.shape == (640, 360, 3)


def test_beat_adds_pause_and_example_is_valid(tmp_path, monkeypatch):
    from if60 import tts
    monkeypatch.setattr(tts, "CACHE", tmp_path)
    monkeypatch.setitem(tts.SYNTH, "supertonic", lambda text, v, c: (np.full(24000, 0.3, np.float32), 24000))
    a = R.Scene(duration=3, layers=[], caption="펀치라인.")
    b = R.Scene(duration=3, layers=[], caption="펀치라인.", beat=0.8)
    R.apply_voice([a], "supertonic:M1", log=lambda *x: None)
    R.apply_voice([b], "supertonic:M1", log=lambda *x: None)
    assert b.duration - a.duration == pytest.approx(0.8)
    ytxt = open("scenes/juvenile_law_abolished.yaml", encoding="utf-8").read()
    assert planner.validate(yaml.safe_load(ytxt), planner.catalog(), 90) == []
