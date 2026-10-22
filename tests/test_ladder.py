"""결과 사다리 형식: 3D 속 글자(sign), 생활 세트, 얼굴 클로즈업, 심플 틀, 형식 검사."""
import copy

import numpy as np
import pytest
import yaml

from if60 import overlay, planner
from if60 import render as R
from if60.voxel import models, signs
from if60.voxel.templates import TEMPLATES

EX = "scenes/sleep_bank.yaml"


def _spec():
    return yaml.safe_load(open(EX, encoding="utf-8"))


def test_sign_texture_and_quad():
    font = R.text.find_font()
    pr = {"shape": "sign", "style": "card", "title": "1층", "text": "개울 앞\n**모기 많음**", "size": [0.8, 1.0]}
    img = signs.sign_texture(pr, font)
    assert img.shape[2] == 4 and img.shape[0] > img.shape[1]            # 세로로 긴 카드
    red = (img[..., 0].astype(int) - img[..., 1] > 80)
    assert red[: img.shape[0] // 4].mean() > 0.5                         # 위쪽 빨간 제목 띠
    quad, n = signs.sign_quad({**pr, "yaw": 0})
    assert quad.shape == (6, 5) and np.allclose(n, [0, 0, -1])          # 앞면이 카메라(-z) 쪽
    for style in signs.STYLES:
        assert signs.sign_texture({"style": style, "text": "가격 1,000원"}, font).size > 0


def test_sign_renders_text_in_3d():
    from if60.voxel.renderer import VoxelRenderer
    vr = VoxelRenderer(90, 160, samples=0, shadow_size=256)
    world = {"template": "studio", "params": {"color": "#202020"}}
    base = vr.prepare(world, [], shot="front")
    cam = base.shots["front"]
    a = vr.render(base, cam, 0.5, 2, 12).astype(int)
    vr.release(base)
    p = vr.prepare(world, [], shot="front", props=[
        {"shape": "sign", "style": "banner", "text": "모기 보장", "size": [1.4, 0.6], "at": [0, 0.5, 0], "color": "#d7262e"}])
    b = vr.render(p, cam, 0.5, 2, 12).astype(int)
    vr.release(p)
    vr.close()
    changed = np.abs(a - b).sum(-1) > 60
    assert changed.mean() > 0.05


def test_bust_expressions_and_new_animals():
    for e in models.EXPRESSIONS:
        name = "bust" if e == "neutral" else f"bust_{e}"
        assert name in models.MODELS and f"bust_long{'' if e == 'neutral' else '_' + e}" in models.MODELS
        assert models.get(name).data.shape[1] == 11
    assert models.get("pig").data.size and models.get("chicken").data.size
    a, b = models.get("bust").data, models.get("bust_surprised").data
    assert a.shape != b.shape or not np.allclose(a, b)                  # 표정이 실제로 다름


@pytest.mark.parametrize("name", ["office", "elevator", "shop", "shopfront", "clinic", "pool", "farm"])
def test_everyday_sets_build(name):
    built = TEMPLATES[name](np.random.default_rng(0))
    assert built.shots and built.areas
    assert built.builder.build().shape[0] > 100
    if name == "shop":
        for kind in ("snack", "butcher", "store", "cafe"):
            TEMPLATES[name](np.random.default_rng(0), kind=kind)


def test_simple_layout_title_only_at_start_of_first_scene():
    font = R.text.find_font()
    lay = overlay.make_layout({"style": "simple", "title": "만약에 잠을 저축할 수 있다면?"}, "", font, (360, 640))
    assert isinstance(lay, overlay.SimpleLayout) and lay.viewport == (0, 640)
    blank = lambda: np.full((640, 360, 3), 60, np.uint8)
    on = lay.apply(blank(), 1.5, first=True)
    later = lay.apply(blank(), 5.0, first=True)
    other = lay.apply(blank(), 1.5, first=False)
    mid = slice(int(640 * 0.15), int(640 * 0.45))
    assert (on[mid] != 60).any(axis=-1).mean() > 0.03                   # 큰 질문 제목
    assert (later[mid] != 60).any(axis=-1).mean() < 0.005 and (other[mid] != 60).any(axis=-1).mean() < 0.005
    assert isinstance(overlay.make_layout({"title": ["a", "b"]}, "", font, (360, 640)), overlay.Layout)


def test_example_ladder_episode_passes_everything():
    ytxt = open(EX, encoding="utf-8").read()
    spec = yaml.safe_load(ytxt)
    assert spec["format"] == "ladder"
    assert planner.validate(spec, planner.catalog(), None) == []
    assert planner.retention_issues(spec) == []
    assert planner.smoke_test(ytxt) == []


def test_ladder_checks_catch_missing_pieces():
    spec = _spec()
    s2 = copy.deepcopy(spec)
    for sc in s2["scenes"]:
        sc.pop("domain", None)
        sc["props"] = [p for p in sc.get("props") or [] if p.get("shape") != "sign"]
    s2["layout"] = {"title": ["두 줄", "제목"]}
    issues = " ".join(planner.retention_issues(s2))
    assert "domain" in issues and "sign" in issues and "simple" in issues
    s3 = copy.deepcopy(spec)
    s3["scenes"] = s3["scenes"][:12] + s3["scenes"][-2:]
    assert any("장면이" in i for i in planner.retention_issues(s3))
    s4 = copy.deepcopy(spec)
    s4["format"] = "documentary"
    assert any("format" in e for e in planner.validate(s4, planner.catalog(), None))
    s5 = copy.deepcopy(spec)
    s5["scenes"][1]["props"] = [{"shape": "sign", "style": "hologram", "size": [1, 1]}]
    errs = planner.validate(s5, planner.catalog(), None)
    assert any("hologram" in e for e in errs) and any("text" in e for e in errs)


def test_broadcast_format_unchanged_by_ladder_rules():
    spec = yaml.safe_load(open("scenes/juvenile_law_abolished.yaml", encoding="utf-8"))
    assert spec.get("format", "broadcast") == "broadcast"
    assert not any("domain" in i or "sign" in i for i in planner.retention_issues(spec))


def test_plan_passes_format_to_writer(monkeypatch):
    seen = []

    def fake_post(body, key):
        seen.append(body["messages"][0]["content"])
        return {"content": [{"type": "text", "text": "```yaml\n" + open(EX, encoding="utf-8").read() + "```"}],
                "stop_reason": "end_turn"}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    _, spec, _ = planner.plan("잠을 저축할 수 있다면?", fmt="ladder", interactive=False, review_pass=False,
                              punchup=False, log=lambda *a: None)
    assert "format: ladder" in seen[0] and "결과 사다리" in seen[0]
    assert spec["format"] == "ladder"
    assert "결과 사다리" in planner.system_prompt(planner.catalog())
