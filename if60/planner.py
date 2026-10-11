"""주제 → (웹 조사) → 장면 YAML → 검증·자동 수정.

Claude API의 웹 검색 도구로 사실을 확인한 뒤, IF 60 Studio가 렌더링할 수 있는 장면 파일을 만든다.
대본 수치와 근거 URL은 facts_to_check에 남겨 사람이 확인할 수 있게 한다.

두 가지 방식으로 Claude를 부른다 (자동 선택, IF60_BACKEND=claude-code|api로 고정 가능)
  1) claude-code: 컴퓨터에 설치된 Claude Code(`claude -p`)를 사용 → Pro/Max 구독 사용량으로 처리, 추가 비용 없음
  2) api: ANTHROPIC_API_KEY(.env)로 Claude API 직접 호출 → 사용량만큼 별도 과금
모델 변경: IF60_PLANNER_MODEL (api 기본 claude-sonnet-5-5 / claude-code 기본 sonnet)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"


# ------------------------------------------------------------------ API 키
def load_env() -> None:
    """저장소 루트의 .env(KEY=VALUE)를 환경변수로 읽는다. .env는 .gitignore로 저장소에서 제외됨."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def save_key(key: str) -> None:
    env = ROOT / ".env"
    lines = []
    if env.exists():
        lines = [l for l in env.read_text(encoding="utf-8").splitlines() if not l.startswith("ANTHROPIC_API_KEY=")]
    lines.append(f"ANTHROPIC_API_KEY={key.strip()}")
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(env, 0o600)   # 내 계정만 읽을 수 있게
    except OSError:
        pass


def require_key(interactive: bool = True) -> str:
    load_env()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    if not interactive:
        raise RuntimeError("ANTHROPIC_API_KEY가 없어요. `python -m if60 key`로 저장하세요.")
    import getpass
    print("\nAnthropic API 키가 필요해요 (처음 한 번만).")
    print("  1) https://console.anthropic.com 에서 로그인 → API Keys → Create Key")
    print("  2) 만든 키(sk-ant-로 시작)를 아래에 붙여넣고 Enter. 화면에는 표시되지 않아요.")
    key = getpass.getpass("API 키: ").strip()
    if not key.startswith("sk-"):
        raise RuntimeError("API 키 형식이 아니에요 (sk-ant-…로 시작해야 해요).")
    save_key(key)
    os.environ["ANTHROPIC_API_KEY"] = key
    print("✔ 이 컴퓨터의 .env 파일에 저장했어요. (GitHub에는 올라가지 않아요)\n")
    return key


# ------------------------------------------------------------------ 카탈로그
def catalog() -> dict:
    """렌더러가 실제로 지원하는 템플릿·샷·구역·모델 목록 (프롬프트와 검증에 같이 쓴다)."""
    from .voxel import models
    from .voxel.templates import TEMPLATES
    cat = {"templates": {}, "agents": sorted(models.MODELS)}
    for name, fn in TEMPLATES.items():
        b = fn(np.random.default_rng(0))
        cat["templates"][name] = {
            "shots": list(b.shots),
            "areas": [a for a in b.areas if not a.startswith("__")],
            "sky": b.sky,
        }
    return cat


TEMPLATE_NOTES = {
    "plaza": "광화문 같은 넓은 광장, 양옆 도로·빌딩, 끝에 성문, 뒤에 산. 군중 장면의 기본",
    "boulevard": "고층 빌딩 협곡 사이 대로. 출퇴근·도심·속도감",
    "crossroad": "위에서 내려다본 교차로·횡단보도·차량. 교통·이동",
    "river_bridge": "강 + 트러스 철교(지하철 train) + 건너편 스카이라인. 한강 느낌",
    "apartments": "아파트 단지 항공샷. 주거·생활",
    "room": "방 안(침대·책상·창문). 일상·클로즈업·작은 생물. params: window_sky(창밖 색, 낮 '#a9cbe8' 밤 '#22305e')",
    "field": "넓은 잔디 공원 + 먼 스카이라인. 야외·마무리 장면",
    "planet": "화성/달 표면 + 착륙선. params: surface: mars|moon. 우주 회차",
}


def system_prompt(cat: dict) -> str:
    tpl_lines = []
    for name, info in cat["templates"].items():
        tpl_lines.append(f"- {name}: {TEMPLATE_NOTES.get(name, '')}\n"
                         f"    shots: {', '.join(info['shots'])}\n"
                         f"    areas: {', '.join(info['areas'])}")
    example = (ROOT / "scenes" / "fly_escape.yaml")
    example_txt = example.read_text(encoding="utf-8") if example.exists() else ""
    return f"""너는 유튜브 과학 상상 쇼츠 채널 'IF 60'의 작가 겸 연출가다.
주제를 받으면 (1) 웹 검색으로 핵심 사실과 수치를 확인하고 (2) 3D 복셀 렌더러용 장면 YAML을 쓴다.

## 콘텐츠 공식 (조회수가 검증된 구조)
- 0~4초 훅: 상단 title(18자 이내 질문) + 궁금증을 만드는 caption
- 본편: 한 장면 = 한 가지 '실제 사실'을 '한국인에게 익숙한 장소·상황'에 대입한 비트. 4~7초씩.
- HUD(우측 상단 배지)로 진행감: 시계 {{from: "05:00", to: "05:30", step: 1}}, 카운터 {{from: 0, to: 0.2, format: "{{:.2f}}초", ease: false}}, 또는 고정 문구 "슬로 모션 ×4"
- 후반 반전 1회(의외의 사실), 마지막 장면은 title + 댓글을 부르는 질문(양자택일이면 더 좋음)
- 말투: 친근한 존댓말. caption은 한 장면 45자 이내, 숫자는 구체적으로.
- 사실은 검색으로 확인된 것만. 불확실하면 쓰지 않는다. 연구 대상이 특정 종이면 그 사실을 facts_to_check에 적는다.
- 위험한 행동을 따라 하게 만드는 내용, 실존 인물·상표·저작권 캐릭터는 넣지 않는다.

## 렌더러가 지원하는 것 (이 목록 밖의 이름을 쓰면 렌더링이 실패한다)
템플릿(world.template):
{chr(10).join(tpl_lines)}
군중 agent: {', '.join(cat['agents'])}

장면 필드:
- duration(초), caption, title(선택), hud(선택), time(선택: 조명 시각 "14:00" 또는 {{from, to}}; 없으면 hud 시계를 따르고 그것도 없으면 정오)
- world: {{template: 이름, seed: 정수, params: {{...}}}}  — 같은 장소를 여러 장면에 쓰면 seed를 같게
- shot: 위 목록의 샷 이름, 또는 직접 카메라:
    {{from: [x,y,z], to: [x,y,z], target: [x,y,z], target_to: [x,y,z], fov: 50}}
    {{type: orbit, center: [x,y,z], radius: r, height: h, from_deg: a, to_deg: b, look_height: h2, fov: 45}}
    {{type: topdown, center: [x,y,z], height: 90, height_to: 75}}
  좌표는 미터, x 오른쪽, y 위, z 앞. room 템플릿은 바닥 중앙이 (0,0,0)이고 방은 x -3~3, z -4.6~2.6, 높이 2.9.
- crowds: 목록. 각 항목 {{agent, count, area(구역 이름 또는 [x0,z0,x1,z1]), layout: random|grid|queue, axis: x|z,
    facing: random|각도|[x,z]|velocity, velocity: [vx,vz] 또는 [vx,vy,vz] (m/s), bob(점프 높이 m), bob_freq,
    scale: 숫자 또는 [최소,최대], y(높이 고정), colors: ["#hex"]}}
  가이드: 광장 군중 1000~3000, 보도 200~300, 고양이·강아지는 넓은 샷에서 scale [1.6, 2.0],
  파리 같은 작은 생물은 room에서 scale 0.2 안팎 + 직접 카메라로 클로즈업(카메라와 0.3~0.8m),
  상상 장면에서 생물을 거대하게 보여줄 땐 scale 10~16. 열차는 river_bridge에서
  {{agent: train, count: 8, area: [-90,104,400,104], layout: queue, axis: x, gap: 18.6, velocity: [4,0], y: 15, speed_jitter: 0}}
- effects(선택): shake {{at, length, amp}}, flash {{at, length}}, tint {{color, at, length}}
- camera(선택): {{fade_in: 초, fade_out: 초}}  — 마지막 장면에 fade_out 0.6

## 출력 형식
설명 없이 ```yaml 코드블록 하나만 출력한다. 최상위 키:
title, style: voxel, music: {{mood: wonder|tense|calm|epic, bpm: 90~130, seed: 정수, volume: 0.18}},
facts_to_check: ["사실 한 줄 — 출처 URL", ...]  (대본에 쓴 모든 수치),
scenes: [...]

## 참고 예시 (형식과 연출 수준)
{example_txt}
"""


# ------------------------------------------------------------------ API 호출
def _post(body: dict, key: str) -> dict:
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Claude API 오류 {e.code}: {detail}") from None


def ask(system: str, user: str, key: str, web_search: bool = True, max_tokens: int = 12000) -> tuple[str, list]:
    """텍스트 응답과 인용(citations) 목록을 돌려준다. 웹 검색이 거부되면 검색 없이 다시 시도."""
    model = os.environ.get("IF60_PLANNER_MODEL", DEFAULT_MODEL)
    tools_variants = [
        [{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
        [{"type": "web_search_20250305", "name": "web_search", "max_uses": 6, "allowed_callers": ["direct"]}],
        None,
    ] if web_search else [None]
    last_err = None
    for tools in tools_variants:
        messages = [{"role": "user", "content": user}]
        body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
        if tools:
            body["tools"] = tools
        try:
            data = _post(body, key)
            for _ in range(4):   # 검색이 길어지면 pause_turn → 이어서 계속
                if data.get("stop_reason") != "pause_turn":
                    break
                messages = messages + [{"role": "assistant", "content": data["content"]}]
                data = _post({**body, "messages": messages}, key)
        except RuntimeError as e:
            last_err = e
            if "오류 400" in str(e) and tools is not None:
                continue
            raise
        texts, cites = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
                for c in block.get("citations") or []:
                    if c.get("url"):
                        cites.append({"url": c["url"], "title": c.get("title", "")})
        if tools is None and web_search:
            print("  (웹 검색을 쓸 수 없어 검색 없이 작성했어요 — 사실 확인을 더 꼼꼼히 해주세요)")
        return "".join(texts), cites
    raise last_err or RuntimeError("Claude API 호출 실패")


def claude_cli() -> str | None:
    """설치된 Claude Code 실행 파일 경로 (없으면 None)"""
    found = shutil.which("claude")
    if found:
        return found
    for cand in (Path.home() / ".local" / "bin" / "claude", Path.home() / ".local" / "bin" / "claude.exe"):
        if cand.exists():
            return str(cand)
    return None


def _cli_env() -> dict:
    # API 키가 환경에 있으면 Claude Code가 구독 대신 API로 과금하므로 하위 프로세스에서는 뺀다
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


def claude_code_logged_in(cli: str) -> bool:
    try:
        r = subprocess.run([cli, "auth", "status"], capture_output=True, text=True, timeout=30, env=_cli_env())
        return r.returncode == 0
    except Exception:
        return False


def ask_cli(system: str, user: str, web_search: bool = True) -> tuple[str, list]:
    """Claude Code 비대화 모드로 질문. 웹 검색·웹 읽기 도구만 허용(파일 수정·명령 실행 불가)."""
    cli = claude_cli()
    if not cli:
        raise RuntimeError("Claude Code가 설치되어 있지 않아요.")
    tools = "WebSearch,WebFetch" if web_search else ""
    with tempfile.TemporaryDirectory() as tmp:
        sp = Path(tmp) / "system.txt"
        sp.write_text(system, encoding="utf-8")
        cmd = [cli, "-p", user, "--append-system-prompt-file", str(sp), "--tools", tools,
               "--output-format", "text", "--no-session-persistence",
               "--model", os.environ.get("IF60_PLANNER_MODEL", "sonnet")]
        if tools:
            cmd += ["--allowedTools", tools]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=tmp, env=_cli_env(),
                               encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code 응답이 20분 안에 오지 않았어요. 잠시 후 다시 시도하세요.") from None
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()[-600:]
        if "limit" in msg.lower():
            msg += "\n(Pro 사용량 한도에 도달했을 수 있어요. 5시간 주기로 초기화됩니다.)"
        raise RuntimeError(f"Claude Code 오류: {msg}")
    return r.stdout, []


def choose_backend(interactive: bool = True) -> str:
    forced = os.environ.get("IF60_BACKEND", "").strip()
    load_env()
    if forced in ("claude-code", "api"):
        return forced
    cli = claude_cli()
    if cli and claude_code_logged_in(cli):
        return "claude-code"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "api"
    if cli:
        raise RuntimeError("Claude Code가 설치돼 있지만 로그인이 안 되어 있어요.\n"
                           "  터미널에서 `claude` 를 한 번 실행해 Pro 계정으로 로그인한 뒤 다시 시도하세요.")
    raise RuntimeError(
        "대본을 쓸 Claude를 찾지 못했어요. 둘 중 하나를 준비하세요:\n"
        "  [추천·Pro 구독으로 무료] Claude Code 설치:  curl -fsSL https://claude.ai/install.sh | bash\n"
        "      → 새 터미널에서 `claude` 실행 → Pro 계정으로 로그인\n"
        "  [별도 과금] API 키 저장:  python -m if60 key")


_TEXT_KEY = re.compile(r"^(\s*(?:-\s+)?(?:caption|title|hud):\s+)(.+)$")


def quote_text_values(ytxt: str) -> str:
    """'caption: 연구: 60일' 처럼 값 안에 콜론·#이 있으면 YAML이 깨지므로 따옴표로 감싼다."""
    out = []
    for line in ytxt.splitlines():
        m = _TEXT_KEY.match(line)
        if m:
            val = m.group(2).rstrip()
            if val[0] not in "\"'{[|>&*!" and (": " in val or " #" in val or val.endswith(":")):
                line = m.group(1) + '"' + val.replace("\\", "\\\\").replace('"', '\\"') + '"'
        out.append(line)
    return "\n".join(out) + "\n"


def extract_yaml(text: str) -> str:
    blocks = re.findall(r"```(?:yaml|yml)?\s*\n(.*?)```", text, re.S)
    if blocks:
        body = max(blocks, key=len).strip()
    else:
        i = text.find("title:")
        body = (text[i:] if i >= 0 else text).strip()
    return quote_text_values(body)


# ------------------------------------------------------------------ 검증
def validate(spec, cat: dict, target_len: float | None = None) -> list[str]:
    errs = []
    if not isinstance(spec, dict) or not isinstance(spec.get("scenes"), list) or not spec["scenes"]:
        return ["최상위에 scenes 목록이 없어요"]
    total = 0.0
    for i, s in enumerate(spec["scenes"], 1):
        if not isinstance(s, dict):
            errs.append(f"장면 {i}: 형식 오류"); continue
        try:
            total += float(s.get("duration", 0))
        except (TypeError, ValueError):
            errs.append(f"장면 {i}: duration이 숫자가 아님")
        w = s.get("world")
        if not isinstance(w, dict) or w.get("template") not in cat["templates"]:
            errs.append(f"장면 {i}: world.template이 없거나 지원하지 않음 ({(w or {}).get('template') if isinstance(w, dict) else w})")
            continue
        info = cat["templates"][w["template"]]
        shot = s.get("shot")
        if shot is None:
            pass
        elif isinstance(shot, str):
            if shot not in info["shots"]:
                errs.append(f"장면 {i}: '{w['template']}'에 샷 '{shot}' 없음 (가능: {', '.join(info['shots'])})")
        elif isinstance(shot, dict):
            kind = shot.get("type", "dolly")
            need = {"dolly": ["from", "target"], "orbit": ["center"], "topdown": ["center"]}.get(kind)
            if need is None:
                errs.append(f"장면 {i}: shot.type '{kind}' 지원 안 함 (dolly/orbit/topdown)")
            else:
                for k in need:
                    if k not in shot:
                        errs.append(f"장면 {i}: 카메라에 '{k}' 필요")
        else:
            errs.append(f"장면 {i}: shot 형식 오류")
        effs = s.get("effects") or []
        for e in (effs if isinstance(effs, list) else [effs]):
            names = ([e] if isinstance(e, str) else
                     [e.get("type") or e.get("name") or e.get("effect")] if isinstance(e, dict) and
                     (e.get("type") or e.get("name") or e.get("effect")) else
                     list(e) if isinstance(e, dict) else [str(e)])
            for n in names:
                if n not in ("shake", "flash", "tint"):
                    errs.append(f"장면 {i}: 효과 '{n}' 지원 안 함 (shake/flash/tint만 가능, 형식: - shake: {{at: 1, length: 1.2, amp: 4}})")
        for j, c in enumerate(s.get("crowds") or [], 1):
            if not isinstance(c, dict):
                errs.append(f"장면 {i} 군중 {j}: 형식 오류"); continue
            ag = c.get("agent", c.get("model", "person"))
            if ag not in cat["agents"]:
                errs.append(f"장면 {i} 군중 {j}: agent '{ag}' 없음 (가능: {', '.join(cat['agents'])})")
            area = c.get("area")
            if isinstance(area, str) and area not in info["areas"]:
                errs.append(f"장면 {i} 군중 {j}: '{w['template']}'에 구역 '{area}' 없음 (가능: {', '.join(info['areas'])})")
            elif isinstance(area, list) and len(area) != 4:
                errs.append(f"장면 {i} 군중 {j}: area 좌표는 [x0, z0, x1, z1] 4개")
            v = c.get("velocity")
            if v is not None and (not isinstance(v, list) or len(v) not in (2, 3)):
                errs.append(f"장면 {i} 군중 {j}: velocity는 [vx, vz] 또는 [vx, vy, vz]")
            try:
                if int(c.get("count", 1)) > 8000:
                    errs.append(f"장면 {i} 군중 {j}: count가 너무 많음 (8000 이하)")
            except (TypeError, ValueError):
                errs.append(f"장면 {i} 군중 {j}: count가 숫자가 아님")
    if target_len and not (target_len * 0.75 <= total <= target_len * 1.3):
        errs.append(f"전체 길이 {total:.0f}초 — 목표 {target_len:.0f}초에 맞춰 장면 길이를 조정")
    return errs


def smoke_test(ytxt: str) -> list[str]:
    """장면마다 아주 작은 프레임을 실제로 그려 본다. 렌더링 중 날 오류를 미리 잡아 자동 수정 단계로 보낸다."""
    from . import render as R
    errs: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check.yaml"
        path.write_text(ytxt, encoding="utf-8")
        try:
            spec, style, scenes = R.load_project(str(path))
            font = R.text.find_font(style.font)
        except Exception as e:
            return [f"장면 파일을 불러오다 오류: {type(e).__name__}: {e}"]
        vr = None
        try:
            from .voxel.renderer import VoxelRenderer
            vr = VoxelRenderer(36, 64, samples=0, shadow_size=256)
        except Exception:
            vr = None   # OpenGL을 못 쓰는 환경이면 3D 확인은 건너뜀
        for i, sc in enumerate(scenes, 1):
            try:
                for t in (0.0, sc.duration * 0.5):
                    R._text_layer(sc, t, style, font)
                    R.hud_label(sc.hud, t, sc.duration)
                if sc.world and vr is not None:
                    prep = vr.prepare(sc.world, sc.crowds, seed=sc.seed, shot=sc.shot)
                    cam = sc.shot if isinstance(sc.shot, dict) else prep.shots.get(sc.shot or next(iter(prep.shots)))
                    if cam is None:
                        raise ValueError(f"샷 '{sc.shot}' 없음")
                    R.render_3d_frame(vr, prep, cam, sc, sc.duration * 0.5)
                    vr.release(prep)
                elif not sc.world:
                    R.compose_frame(sc, 0.0, style)
            except Exception as e:
                errs.append(f"장면 {i} 렌더링 시험 중 오류: {type(e).__name__}: {e}")
        if vr is not None:
            vr.close()
    return errs


# ------------------------------------------------------------------ 전체 흐름
def plan(topic: str, length: int = 60, *, interactive: bool = True, web_search: bool = True,
         log=print) -> tuple[str, dict, list]:
    """주제 → 검증된 장면 YAML 텍스트, 파싱된 spec, 인용 목록"""
    backend = choose_backend(interactive)
    if backend == "api":
        key = require_key(interactive)
        call = lambda u, ws: ask(system, u, key, web_search=ws)
        log("  · Claude API 사용 (사용량만큼 별도 과금)")
    else:
        call = lambda u, ws: ask_cli(system, u, web_search=ws)
        log("  · Claude Code + 내 Claude 구독 사용 (추가 비용 없음, 구독 사용량에서 차감)")
    cat = catalog()
    system = system_prompt(cat)
    user = (f"주제: {topic}\n목표 길이: 약 {length}초\n"
            f"먼저 웹 검색으로 핵심 사실과 수치를 확인한 뒤, 지정한 형식의 YAML만 출력해.")
    log("▶ 1/3 자료 조사 + 대본·연출 작성 중 (1~5분)")
    t0 = time.time()
    text, cites = call(user, web_search)
    log(f"  · 완료 ({time.time() - t0:.0f}초)")
    ytxt = extract_yaml(text)
    for attempt in range(3):
        try:
            spec = yaml.safe_load(ytxt)
            errs = validate(spec, cat, length)
        except yaml.YAMLError as e:
            spec, errs = None, [f"YAML 문법 오류: {e}"]
        if not errs:
            log("  · 장면별 시험 렌더링으로 오류 확인 중")
            errs = smoke_test(ytxt)
        if not errs:
            break
        log(f"▶ 장면 파일 자동 수정 중 ({attempt + 1}/3): " + "; ".join(errs[:3]) + (" …" if len(errs) > 3 else ""))
        fix = (f"아래 YAML을 렌더러가 읽을 수 있게 고쳐. 문제:\n- " + "\n- ".join(errs) +
               f"\n\n내용(대본·사실)은 유지하고 문제만 고쳐서 ```yaml 블록 하나로 출력해.\n\n```yaml\n{ytxt}```")
        text2, _ = call(fix, False)
        ytxt = extract_yaml(text2)
    else:
        raise RuntimeError("장면 파일을 자동으로 고치지 못했어요: " + "; ".join(errs))
    return ytxt, spec, cites


def slugify(title: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M")
    ascii_part = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:30]
    return f"{stamp}_{ascii_part}" if ascii_part else stamp


def write_scene_file(topic: str, ytxt: str, spec: dict, cites: list) -> Path:
    out_dir = ROOT / "scenes"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{slugify(spec.get('title', topic))}.yaml"
    head = [f"# 주제: {topic}", f"# 자동 생성: {time.strftime('%Y-%m-%d %H:%M')} (if60 make)",
            "# 올리기 전에 facts_to_check의 수치와 출처를 꼭 확인하세요."]
    seen = set()
    for c in cites:
        if c["url"] not in seen:
            seen.add(c["url"])
            head.append(f"# 참고: {c['title']} — {c['url']}")
    path.write_text("\n".join(head) + "\n" + ytxt, encoding="utf-8")
    return path
