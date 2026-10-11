"""주제 → 장면 파일 자동 생성 흐름 테스트 (Claude API는 가짜 응답으로 대체)."""
import json

import pytest
import yaml

from if60 import planner

GOOD = """```yaml
title: 테스트 주제
style: voxel
music: {mood: wonder, bpm: 110, seed: 1}
facts_to_check: ["사실 하나 — https://example.com"]
scenes:
  - duration: 30
    title: 테스트?
    caption: 첫 장면이에요.
    world: {template: plaza, seed: 1}
    shot: forward
    crowds:
      - {agent: cat, count: 50, area: plaza}
  - duration: 30
    caption: 마지막 장면.
    world: {template: room, seed: 1}
    shot: inside
```"""

BAD = GOOD.replace("shot: forward", "shot: helicopter").replace("area: plaza", "area: moon_base")


@pytest.fixture(scope="module")
def cat():
    return planner.catalog()


def test_catalog_lists_templates_and_shots(cat):
    assert "plaza" in cat["templates"] and "forward" in cat["templates"]["plaza"]["shots"]
    assert "fly" in cat["agents"] and "swatter" in cat["agents"]


def test_validate_catches_unknown_names(cat):
    errs = planner.validate(yaml.safe_load(planner.extract_yaml(BAD)), cat, 60)
    assert any("helicopter" in e for e in errs)
    assert any("moon_base" in e for e in errs)
    assert planner.validate(yaml.safe_load(planner.extract_yaml(GOOD)), cat, 60) == []


def test_validate_checks_length(cat):
    spec = yaml.safe_load(planner.extract_yaml(GOOD))
    assert any("전체 길이" in e for e in planner.validate(spec, cat, 120))


def _resp(text, stop="end_turn", cites=None):
    block = {"type": "text", "text": text}
    if cites:
        block["citations"] = cites
    return {"content": [block], "stop_reason": stop}


def test_plan_repairs_invalid_yaml(monkeypatch):
    calls = []

    def fake_post(body, key):
        calls.append(body)
        if len(calls) == 1:
            return _resp(BAD, cites=[{"url": "https://example.com/a", "title": "A"}])
        return _resp(GOOD)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(planner, "_post", fake_post)
    ytxt, spec, cites = planner.plan("테스트 주제", 60, interactive=False, log=lambda *a: None)
    assert len(calls) == 2                       # 처음 작성 + 자동 수정 1회
    assert "tools" in calls[0] and "tools" not in calls[1]   # 조사할 때만 웹 검색
    assert spec["scenes"][0]["shot"] == "forward"
    assert cites[0]["url"] == "https://example.com/a"


def test_plan_continues_after_pause_turn(monkeypatch):
    seq = [_resp("검색 중…", stop="pause_turn"), _resp(GOOD)]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(planner, "_post", lambda body, key: seq.pop(0))
    _, spec, _ = planner.plan("주제", 60, interactive=False, log=lambda *a: None)
    assert spec["title"] == "테스트 주제"


def test_web_search_falls_back_on_400(monkeypatch):
    seen = []

    def fake_post(body, key):
        seen.append(json.dumps(body.get("tools")))
        if body.get("tools"):
            raise RuntimeError("Claude API 오류 400: tool not supported")
        return _resp(GOOD)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(planner, "_post", fake_post)
    _, spec, _ = planner.plan("주제", 60, interactive=False, log=lambda *a: None)
    assert seen[-1] == "null" and len(seen) == 3


def test_key_saved_to_env_file(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    planner.save_key("sk-ant-abc")
    planner.save_key("sk-ant-def")               # 덮어쓰기
    assert (tmp_path / ".env").read_text().count("ANTHROPIC_API_KEY") == 1
    planner.load_env()
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-def"
