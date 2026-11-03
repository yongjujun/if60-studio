"""주제 → 장면 파일 자동 생성 흐름 테스트 (Claude API는 가짜 응답으로 대체)."""
import json

import pytest
import yaml

from if60 import planner

GOOD = """```yaml
title: 테스트 주제
style: voxel
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
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    ytxt, spec, cites = planner.plan("테스트 주제", 60, interactive=False, review_pass=False, log=lambda *a: None, punchup=False)
    assert len(calls) == 2                       # 처음 작성 + 자동 수정 1회
    assert "tools" in calls[0] and "tools" not in calls[1]   # 조사할 때만 웹 검색
    assert spec["scenes"][0]["shot"] == "forward"
    assert cites[0]["url"] == "https://example.com/a"


def test_plan_continues_after_pause_turn(monkeypatch):
    seq = [_resp("검색 중…", stop="pause_turn"), _resp(GOOD)]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", lambda body, key: seq.pop(0))
    _, spec, _ = planner.plan("주제", 60, interactive=False, log=lambda *a: None, punchup=False)
    assert spec["title"] == "테스트 주제"


def test_web_search_falls_back_on_400(monkeypatch):
    seen = []

    def fake_post(body, key):
        seen.append(json.dumps(body.get("tools")))
        if body.get("tools"):
            raise RuntimeError("Claude API 오류 400: tool not supported")
        return _resp(GOOD)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    _, spec, _ = planner.plan("주제", 60, interactive=False, review_pass=False, log=lambda *a: None, punchup=False)
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


FAKE_CLAUDE = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    sys.exit(0)
log = os.environ.get("FAKE_LOG")
with open(log, "a") as f:
    f.write(json.dumps({"args": args, "has_key": "ANTHROPIC_API_KEY" in os.environ}) + "\n")
print(open(os.environ["FAKE_REPLY"]).read())
"""


def test_claude_code_backend_uses_subscription(tmp_path, monkeypatch):
    import os
    import stat
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    reply = tmp_path / "reply.txt"
    reply.write_text(GOOD)
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_REPLY", str(reply))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.delenv("IF60_BACKEND", raising=False)
    monkeypatch.setattr(planner, "_post", lambda *a: (_ for _ in ()).throw(AssertionError("API를 쓰면 안 됨")))
    _, spec, _ = planner.plan("주제", 60, interactive=False, log=lambda *a: None, punchup=False)
    assert spec["title"] == "테스트 주제"
    call = json.loads(log.read_text().splitlines()[0])
    a = call["args"]
    assert a[0] == "-p" and "--append-system-prompt-file" in a
    assert a[a.index("--tools") + 1] == "WebSearch,WebFetch"      # 웹 도구만 (파일 수정·명령 실행 불가)
    assert call["has_key"] is False                                # 구독으로 처리되도록 API 키 제거


def test_no_backend_gives_clear_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    monkeypatch.setattr(planner.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("IF60_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="claude.ai/install.sh"):
        planner.choose_backend(interactive=False)


def test_colon_inside_caption_is_quoted():
    raw = "```yaml\ntitle: 질소: 왜?\nscenes:\n  - duration: 5\n    caption: 한 연구: 질소 포장 60일, 일반 45일.\n    world: {template: room}\n```"
    spec = yaml.safe_load(planner.extract_yaml(raw))
    assert spec["title"] == "질소: 왜?"
    assert spec["scenes"][0]["caption"] == "한 연구: 질소 포장 60일, 일반 45일."


def test_validate_flags_unknown_effects(cat):
    spec = yaml.safe_load(planner.extract_yaml(GOOD))
    spec["scenes"][0]["effects"] = [{"zoom": {"at": 1}}]
    assert any("zoom" in e for e in planner.validate(spec, cat, 60))
    spec["scenes"][0]["effects"] = [{"shake": {"at": 1}, "flash": {"at": 2}}]
    assert planner.validate(spec, cat, 60) == []


def test_smoke_test_catches_render_errors():
    ok = planner.extract_yaml(GOOD)
    assert planner.smoke_test(ok) == []
    broken = ok.replace("hud:", "x:")  # 무해한 변경
    bad = ok.replace("    caption: 첫 장면이에요.", "    caption: 첫 장면이에요.\n    hud: {from: 시작, to: 끝}")
    errs = planner.smoke_test(bad)
    assert errs and "장면 1" in errs[0]


def test_review_revises_mismatched_scene_with_image(monkeypatch):
    """감독 검수: 미리보기 이미지가 첨부되고, 고친 YAML이 오면 그걸 채택"""
    fixed = GOOD.replace("shot: forward", "shot: aerial")
    sent = []

    def fake_post(body, key):
        sent.append(body)
        content = body["messages"][0]["content"]
        if isinstance(content, list):                    # 검수 호출 (이미지 + 텍스트)
            assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"
            assert "1. 자막: 첫 장면이에요." in content[1]["text"]
            return _resp(fixed)
        return _resp(GOOD)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    _, spec, _ = planner.plan("주제", 60, interactive=False, log=lambda *a: None, punchup=False)
    assert len(sent) == 2
    assert spec["scenes"][0]["shot"] == "aerial"


def test_review_pass_keeps_original(monkeypatch):
    def fake_post(body, key):
        if isinstance(body["messages"][0]["content"], list):
            return _resp("통과")
        return _resp(GOOD)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    _, spec, _ = planner.plan("주제", 60, interactive=False, log=lambda *a: None, punchup=False)
    assert spec["scenes"][0]["shot"] == "forward"


def test_review_with_claude_code_reads_image(tmp_path, monkeypatch):
    import os
    import stat
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE.replace('print(open(os.environ["FAKE_REPLY"]).read())',
        'import pathlib\nif "--tools" in args and args[args.index("--tools") + 1] == "Read":\n'
        '    assert pathlib.Path("contact_sheet.png").exists()\n    print("통과")\n'
        'else:\n    print(open(os.environ["FAKE_REPLY"]).read())'))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    reply = tmp_path / "reply.txt"
    reply.write_text(GOOD)
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_REPLY", str(reply))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("IF60_BACKEND", raising=False)
    _, spec, _ = planner.plan("주제", 60, interactive=False, log=lambda *a: None, punchup=False)
    calls = [json.loads(l)["args"] for l in log.read_text().splitlines()]
    assert len(calls) == 2
    assert calls[1][calls[1].index("--tools") + 1] == "Read"   # 검수 때는 파일 읽기만 허용


def test_contact_sheet_has_every_scene():
    from PIL import Image
    import io
    png = planner.contact_sheet(planner.extract_yaml(GOOD))
    assert png is not None
    im = Image.open(io.BytesIO(png))
    assert im.size == (4 * 186, 1 * 326)


def test_validate_props_and_labels(cat):
    spec = yaml.safe_load(planner.extract_yaml(GOOD))
    spec["scenes"][0]["props"] = [{"model": "pizza", "at": [0, 0, 0]}, {"shape": "cone", "at": [0, 0, 0]},
                                  {"shape": "sphere", "r": 0.2, "at": [0, 1, 0]}]
    spec["scenes"][0]["labels"] = [{"text": "질소"}]
    errs = planner.validate(spec, cat, 60)
    assert any("pizza" in e for e in errs) and any("cone" in e for e in errs) and any("라벨" in e for e in errs)
    assert len(errs) == 3


# ------------------------------------------------------------------ 시청 유지 구조 (3초 훅 · 20초 반전 · 끝)
def _scene(caption, **kw):
    return {"duration": 5, "caption": caption, "world": {"template": "studio", "seed": 1}, **kw}


def _retention_spec(hook="만 10세, 지우개 절도. 판결은…", twist_at=4):
    body = "가" * 45    # 약 7.5초짜리 대사 → 전체 약 66초
    acts = ["도입", "도입", "도입", "도입", "전개", "전개", "반전", "반전"]
    images = [1, 2, 2, 3, 4, 4, 5, 5]
    scenes = [_scene(hook, role="hook")] + [_scene(body) for _ in range(7)]
    for s, act, img in zip(scenes, acts, images):
        s["act"], s["image"] = act, img
    scenes[twist_at - 1]["role"] = "twist"
    for s, src in zip(scenes[2:5], ("경찰청 2025", "통계청 2024", "정부 자료 2026")):   # 화면 출처
        s["source"] = src
    scenes[6]["uncertainty"] = True
    for s, tang in zip(scenes[4:7], (1, 2, 2)):                 # 전개~반전: 1단계(직접 결과) → 2단계(현실적인 파급)
        s["tangent"] = tang
    scenes.append(_scene("결국 가장 많은 사건은 가벼운 절도입니다.", role="payoff", act="결말", image=6))
    scenes.append({"duration": 6, "act": "결말", "card": {"question": "?", "options": ["1", "2", "3"]}})
    keyframes = [{"id": i, "desc": f"그림 {i}", "prompt": f"scene {i}, vertical 9:16, no text"} for i in range(1, 7)]
    facts = ["송치 21,095명 — https://example.com/a", "절도 약 48% — https://example.com/b", "21,095 ÷ 25 ≈ 843건 (계산)"]
    return {"genre": "satire", "keyframes": keyframes, "facts_to_check": facts, "scenes": scenes}


def test_retention_good_structure_passes():
    assert planner.retention_issues(_retention_spec()) == []


def test_retention_example_scene_passes():
    spec = yaml.safe_load(open(planner.ROOT / "scenes" / "juvenile_law_abolished.yaml", encoding="utf-8"))
    assert planner.retention_issues(spec) == []


def test_retention_flags_long_hook_late_twist_and_missing_payoff():
    spec = _retention_spec(hook="가" * 40, twist_at=8)
    spec["scenes"][-2].pop("role")
    issues = " / ".join(planner.retention_issues(spec))
    assert "훅" in issues and "15~30초" in issues and "payoff" in issues


def test_plan_restructures_for_retention(monkeypatch):
    """구조 권고가 있으면 한 번 다듬고, 더 나아진 대본을 채택"""
    flat = "```yaml\n" + yaml.safe_dump({"title": "t", "style": "voxel", "scenes": [
        {k: v for k, v in s.items() if k != "role"} for s in _retention_spec()["scenes"]]},
        allow_unicode=True) + "```"
    good = "```yaml\n" + yaml.safe_dump({"title": "t", "style": "voxel", **_retention_spec()}, allow_unicode=True) + "```"
    sent = []

    def fake_post(body, key):
        sent.append(body["messages"][0]["content"])
        return _resp(flat if len(sent) == 1 else good)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    monkeypatch.setattr(planner, "smoke_test", lambda y: [])
    _, spec, _ = planner.plan("주제", 60, interactive=False, review_pass=False, log=lambda *a: None, punchup=False, factcheck=False)
    assert len(sent) == 2 and "시청 유지" in sent[1]
    assert spec["scenes"][0]["role"] == "hook"


def test_plan_keeps_original_when_restructure_fails(monkeypatch):
    flat = "```yaml\n" + yaml.safe_dump({"title": "t", "style": "voxel", "scenes": [
        {k: v for k, v in s.items() if k != "role"} for s in _retention_spec()["scenes"]]},
        allow_unicode=True) + "```"
    seq = [_resp(flat), _resp("```yaml\nscenes: oops\n```")]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", lambda body, key: seq.pop(0))
    monkeypatch.setattr(planner, "smoke_test", lambda y: [])
    _, spec, _ = planner.plan("주제", 60, interactive=False, review_pass=False, log=lambda *a: None, punchup=False)
    assert "role" not in spec["scenes"][0]


# ------------------------------------------------------------------ 만물제작소 제작 조건 (길이 · 4단 구성 · 이미지 6~10장 · 장르)
def test_structure_flags_missing_act_genre_and_keyframes():
    spec = _retention_spec()
    spec.pop("genre")
    spec["scenes"][6]["act"] = "도입"          # 반전 구간이 사라지고 순서도 뒤섞임
    spec["keyframes"] = spec["keyframes"][:4]  # 그림 4장뿐 + 장면이 없는 번호를 가리킴
    issues = " / ".join(planner.retention_issues(spec))
    assert "genre" in issues and "순서" in issues and "4장" in issues and "keyframes에 없는 번호" in issues


def test_structure_flags_length_out_of_range():
    spec = _retention_spec()
    for s in spec["scenes"][1:7]:
        s["caption"] = "가" * 120             # 너무 긴 대사 → 120초 초과
    assert any("120초" in i for i in planner.retention_issues(spec))


def test_image_prompts_file(tmp_path):
    out = planner.write_image_prompts(_retention_spec(), tmp_path / "img.txt")
    txt = out.read_text(encoding="utf-8")
    assert "AI 이미지 6장" in txt and "[2] 그림 2  (장면 2, 3)" in txt


def test_topics_file_is_valid():
    items = planner.load_topics()
    assert len(items) >= 12
    assert all(t["genre"] in planner.GENRES and t.get("angle") and t.get("tangents") for t in items)
    assert [t["id"] for t in planner.pick_topics("1-3, 12")] == [1, 2, 3, 12]
    assert planner.pick_topics("달이 사라진다면?") is None
    with pytest.raises(ValueError):
        planner.pick_topics("999")


def test_plan_sends_genre_and_angle(monkeypatch):
    good = "```yaml\n" + yaml.safe_dump({"title": "t", "style": "voxel", **_retention_spec()}, allow_unicode=True) + "```"
    sent = []

    def fake_post(body, key):
        sent.append(body["messages"][0]["content"])
        return _resp(good)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    monkeypatch.setattr(planner, "smoke_test", lambda y: [])
    t = planner.pick_topics("4")[0]
    planner.plan(t["title"], 60, interactive=False, review_pass=False, log=lambda *a: None,
                 genre=t["genre"], angle=t["angle"], punchup=False, tangents=t["tangents"], factcheck=False)
    assert len(sent) == 1 and "과학 — 정확하게" in sent[0] and "진중한" in sent[0] and "방향 힌트" in sent[0] and "전개 아이디어" in sent[0] and "사람 이름 금지" in sent[0]


# ------------------------------------------------------------------ 진중한 정보 다큐 (결과의 연쇄 · 개그 금지 · 출처, 이름 금지)
def test_chain_flags_missing_steps_and_any_gag():
    spec = _retention_spec()
    for s in spec["scenes"]:
        if s.get("tangent") == 2:
            s.pop("tangent")
    issues = " / ".join(planner.retention_issues(spec))
    assert "2단계" in issues and "웃음 장치" not in issues
    spec = _retention_spec()
    spec["scenes"][3]["gag"] = "정색 반전"
    issues = " / ".join(planner.retention_issues(spec))
    assert "gag" in issues and "진중한" in issues


def test_chain_rejects_tangent_3_and_wrong_order():
    spec = _retention_spec()
    spec["scenes"][4]["tangent"], spec["scenes"][6]["tangent"] = 2, 1
    assert "순서" in " / ".join(planner.retention_issues(spec))
    spec = _retention_spec()
    spec["scenes"][6]["tangent"] = 3
    assert "tangent는 1" in " / ".join(planner.retention_issues(spec))


def test_accuracy_needs_sourced_facts():
    spec = _retention_spec()
    spec["facts_to_check"] = ["판사는 25명"]
    issues = " / ".join(planner.retention_issues(spec))
    assert "facts_to_check가 1개" in issues and "출처" in issues
    assert "개그" not in planner.POLISH_SYSTEM.split("\n")[0] and "진중" in planner.POLISH_SYSTEM
    assert "웃음 장치(gag:)는 쓰지 않는다" in planner.system_prompt(planner.catalog())


def test_no_person_names():
    spec = _retention_spec()
    assert planner.name_issues(spec) == []
    spec["scenes"][2]["caption"] = "그 시각, 하준 씨는 혼자 알람시계를 샀습니다."
    spec["scenes"][3]["caption"] = "김 대리는 오늘도 야근입니다."
    spec["cast"] = [{"name": "순자", "role": "시장 상인"}]
    msg = " ".join(planner.name_issues(spec))
    assert "하준 씨" in msg and "김 대리" in msg and "순자" in msg
    ok = _retention_spec()
    ok["scenes"][2]["caption"] = "한 직장인은 아저씨에게 길을 물었고, 편의점 알바생은 웃었습니다."
    assert planner.name_issues(ok) == []


def test_punchup_applies_safe_lines_only():
    spec = _retention_spec()
    spec["scenes"][1]["caption"] = "판사는 25명이에요."
    spec["scenes"][1]["narration"] = "판사는 스물다섯 명이에요."
    reply = json.dumps([
        {"n": 2, "caption": "판사는 25명. 주말은 0일.", "narration": "판사는 스물다섯 명. 주말은 영 일."},  # 숫자 추가 → 버림
        {"n": 6, "caption": "회의도 줄고 야근도 줄고 잔소리도 줄었는데, 이혼은 늘었습니다."},                              # 반영
        {"n": 4, "caption": "가" * 200},                                                              # 너무 김 → 버림
        {"n": 10, "caption": "카드는 못 바꿈"},                                                       # 카드 → 버림
        {"n": 5, "caption": "판사 25명"},                                                             # 읽는 법 없음 → 버림
    ], ensure_ascii=False)
    new, n = planner.apply_punchup(spec, "다듬었어요:\n" + reply)
    assert n == 1
    assert new["scenes"][5]["caption"].endswith("이혼은 늘었습니다.")
    assert new["scenes"][1]["caption"] == "판사는 25명이에요."
    assert spec["scenes"][5]["caption"] != new["scenes"][5]["caption"]     # 원본은 그대로


def test_plan_runs_punchup_with_light_prompt(monkeypatch):
    good = "```yaml\n" + yaml.safe_dump({"title": "t", "style": "voxel", **_retention_spec()}, allow_unicode=True) + "```"
    systems = []

    def fake_post(body, key):
        systems.append(body["system"])
        if len(systems) == 1:
            return _resp(good)
        return _resp(json.dumps([{"n": 6, "caption": "회의와 야근은 줄었지만, 이혼 건수는 늘어난 것으로 추정됩니다."}], ensure_ascii=False))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    monkeypatch.setattr(planner, "smoke_test", lambda y: [])
    ytxt, spec, _ = planner.plan("주제", 60, interactive=False, review_pass=False, log=lambda *a: None, factcheck=False)
    assert len(systems) == 2 and systems[1] == planner.POLISH_SYSTEM
    assert len(systems[1]) < len(systems[0]) / 10            # 편집자 다듬기는 짧은 지시문으로
    assert spec["scenes"][5]["caption"].endswith("추정됩니다.")
    assert yaml.safe_load(ytxt)["scenes"][5]["caption"] == spec["scenes"][5]["caption"]


# ------------------------------------------------------------------ 전문가식 전개 · 화면 출처 · 팩트 재검증
def test_sources_and_uncertainty_required():
    spec = _retention_spec()
    for s in spec["scenes"]:
        s.pop("source", None)
        s.pop("uncertainty", None)
    issues = " / ".join(planner.retention_issues(spec))
    assert "source" in issues and "uncertainty" in issues
    spec = _retention_spec()
    spec["scenes"][2]["source"] = "https://example.com/very/long/link"
    assert any("URL" in i for i in planner.retention_issues(spec))
    assert "전문가식 전개" in planner.system_prompt(planner.catalog())


def test_source_label_drawn():
    from if60 import overlay, text
    import numpy as np
    img = np.full((640, 360, 3), 200, np.uint8)
    overlay.draw_source(img, "경찰청 2025", text.find_font(), 1 / 3)
    band = img[int(640 * 0.64):int(640 * 0.69)]
    assert (band[:, 200:] < 120).mean() > 0.2 and (band[:, :100] == 200).all()   # 오른쪽에 어두운 띠


def test_factcheck_prompt_is_tiny_and_parsed():
    spec = {"facts_to_check": ["송치 21,095명 — https://www.example.com/news/123456789?ref=abc",
                               "나머지는 상상", "21,095 ÷ 25 ≈ 843건 (계산)"]}
    prompt, facts = planner.factcheck_prompt(spec)
    assert len(facts) == 2 and "example.com" in prompt and "123456789" not in prompt
    assert len(planner.FACTCHECK_SYSTEM) < 400
    warns = planner.parse_factcheck('결과: [{"n": 2, "note": "843.8 → 844건"}, {"n": 9, "note": "x"}]', facts)
    assert len(warns) == 1 and "844" in warns[0]
    assert planner.parse_factcheck("[]", facts) == [] and planner.parse_factcheck("모름", facts) == []


def test_plan_runs_light_factcheck(monkeypatch):
    good = "```yaml\n" + yaml.safe_dump({"title": "t", "style": "voxel", **_retention_spec()}, allow_unicode=True) + "```"
    bodies = []

    def fake_post(body, key):
        bodies.append(body)
        if len(bodies) == 1:
            return _resp(good)
        return _resp('[{"n": 1, "note": "2025년 실제 값은 21,094명"}]')

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("IF60_BACKEND", "api")
    monkeypatch.setattr(planner, "_post", fake_post)
    monkeypatch.setattr(planner, "smoke_test", lambda y: [])
    ytxt, spec, _ = planner.plan("주제", 60, interactive=False, review_pass=False, punchup=False, log=lambda *a: None)
    chk = bodies[-1]
    assert chk["system"] == planner.FACTCHECK_SYSTEM and chk["max_tokens"] <= 1000
    assert chk["tools"][0]["max_uses"] <= 3 and len(chk["messages"][0]["content"]) < 600
    assert spec["fact_warnings"] and "21,094" in yaml.safe_load(ytxt)["fact_warnings"][0]
    bodies.clear()
    planner.plan("주제", 60, interactive=False, review_pass=False, punchup=False, factcheck=False, log=lambda *a: None)
    assert len(bodies) == 1
