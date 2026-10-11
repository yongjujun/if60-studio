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
    "studio": "★사물·물질·원리 설명의 기본 무대. 단색 배경(params: color) + 바닥(y=0). 물체를 (0,0,0) 근처에 놓는다. "
              "front 샷에서 보이는 범위: 가로 x -0.8~0.8, 높이 y 0~2.2 / close: x -0.4~0.4, y 0~1",
    "tabletop": "★일상 사물 클로즈업. 방 안 식탁, 상판 높이 y=0.9, 상판 범위 x -1.2~1.2, z -0.6~0.6. "
                "table 샷에서 보이는 폭 약 x -0.55~0.55. 소품 at의 y는 0.9",
    "plaza": "광화문 같은 넓은 광장, 양옆 도로·빌딩, 끝에 성문, 뒤에 산. 많은 사람·동물이 모이는 장면",
    "boulevard": "고층 빌딩 협곡 사이 대로. 출퇴근·도심·속도감",
    "crossroad": "위에서 내려다본 교차로·횡단보도·차량. 교통·이동",
    "river_bridge": "강 + 트러스 철교(지하철 train) + 건너편 스카이라인. 한강 느낌",
    "apartments": "아파트 단지 항공샷. 주거·생활",
    "room": "방 안(침대·책상·창문). 일상·작은 생물. params: window_sky(창밖 색, 낮 '#a9cbe8' 밤 '#22305e')",
    "field": "넓은 잔디 공원 + 먼 스카이라인. 야외·사람들·하늘",
    "planet": "화성/달 표면 + 착륙선. params: surface: mars|moon. 우주",
}


def system_prompt(cat: dict) -> str:
    from .voxel.models import DESCRIPTIONS
    tpl_lines = []
    for name in ["studio", "tabletop"] + [n for n in cat["templates"] if n not in ("studio", "tabletop")]:
        info = cat["templates"][name]
        tpl_lines.append(f"- {name}: {TEMPLATE_NOTES.get(name, '')}\n"
                         f"    shots: {', '.join(info['shots'])}\n"
                         f"    areas: {', '.join(info['areas'])}")
    model_lines = [f"  {n}: {DESCRIPTIONS.get(n, '')}" for n in cat["agents"]]
    ex1 = ROOT / "scenes" / "lightning_power.yaml"
    ex2 = ROOT / "scenes" / "nitrogen_snack.yaml"
    example_txt = "\n".join(
        f"### {label}\n```yaml\n{p.read_text(encoding='utf-8')}```" for label, p in
        (("예시 1 — 방송형 전체 구조 (콜드 오픈·대시보드·VFX·마무리 카드)", ex1),
         ("예시 2 — 사물을 화면에 직접 만들어 보여주기", ex2)) if p.exists())
    return f"""너는 유튜브 과학·상식 쇼츠 채널 'IF 60'의 작가 겸 연출가다. 주제는 무엇이든 올 수 있다(사물, 음식, 몸, 동물, 우주, 사회, 상상).
주제를 받으면 (1) 웹 검색으로 핵심 사실과 수치를 확인하고 (2) 3D 복셀 렌더러용 장면 YAML을 쓴다.

## 제1 원칙: 화면이 자막을 직접 보여준다
시청자가 소리 없이 화면만 봐도 그 장면 자막이 무슨 말인지 알 수 있어야 한다.
- 장면마다 `visual:` 한 줄로 "화면에 무엇이 어떻게 보이는지"를 먼저 쓰고, 그 visual이 caption의 핵심 대상(명사)과 동작을 그대로 보여주게 연출한다.
- 사물·음식·물질·원리·작은 것·몸속·분자 이야기 → studio 또는 tabletop 세트 + props(사물) + crowds(입자·여러 개)로 그 대상을 직접 만든다.
- 도시 템플릿(plaza, boulevard, crossroad…)은 자막이 실제로 장소·사람들·교통·사회 현상을 말할 때만 쓴다. 주제와 상관없는 도시 풍경을 '배경용'으로 쓰지 않는다.
- 목록에 딱 맞는 모델이 없으면 shape(box/sphere/cylinder)를 조합해 직접 만든다. 예: 탄산음료 캔 = 빨간 cylinder, 피자 = 납작한 주황 cylinder + 작은 빨간 sphere들, 얼음 = 하늘색 box.
- labels로 핵심 대상에 이름표를 붙인다 (예: "질소 N2", "산소 → 기름과 반응"). 숫자는 hud로 보여준다.
- 화면은 세로 9:16이라 좁다. 카메라와 거리 d일 때 보이는 가로 폭은 약 0.45×d, 세로 높이는 약 0.9×d (fov 50 기준). 주인공 물체가 화면 가운데 크게 보이게 거리·scale을 정한다.
- 좌표: 미터, x는 화면 오른쪽(+z를 바라볼 때), y 위, z 앞(카메라 반대쪽). 템플릿 샷들은 대부분 +z를 바라본다.

## 콘텐츠 공식 (조회수가 검증된 방송형 구조 — 채널 '만물제작소')
- 화면 틀: 최상위 layout으로 상단에 두 줄 제목(첫 줄 흰색, 둘째 줄 노란색 — 각 10자 안팎)이 영상 내내 고정된다.
  그래서 장면별 title은 쓰지 않는다(필요하면 짧게 한 번만). 채널명은 레이아웃이 알아서 작게 넣는다.
- ★콜드 오픈: 1번 장면(2~3초)은 영상에서 가장 센 순간(절정·결과)을 먼저 보여준다 — big_text로 핵심 숫자,
  vfx·shake·whiteout(hold: true)로 강하게 끝낸다. 2번 장면은 effects에 rewind {{at: 0}}를 넣고
  "어떻게 이렇게 됐을까요? 처음부터 볼게요." 같은 caption으로 시작점부터 다시 보여준다.
- 본편: 한 장면 = 한 가지 사실. 4~6초씩. 숫자가 나오면 panel(대시보드)의 카운터가 0부터 굴러 올라가게,
  핵심 수치는 big_text로 크게 빛나게, 가리키는 지점은 markers로 표시한다. 비유는 한국인에게 익숙한 것으로.
- 후반 반전 1회(의외의 사실이나 비교), 마지막 장면은 card(질문 + 번호 선택지 3개 + "댓글로 번호를 남겨주세요").
- HUD(우측 상단 배지)는 시계·짧은 문구에만: 시계 {{from: "05:00", to: "05:30", step: 1}}, 고정 문구 "공기 = 질소 78%"
- 말투: 친근한 존댓말. caption은 한 장면 45자 이내, 숫자는 구체적으로.
- 사실은 검색으로 확인된 것만. 불확실하면 쓰지 않는다. 출처를 facts_to_check에 URL과 함께.
- 위험한 행동을 따라 하게 만드는 내용, 실존 인물·상표·저작권 캐릭터는 넣지 않는다.

## 렌더러가 지원하는 것 (이 목록 밖의 이름을 쓰면 렌더링이 실패한다)
템플릿(world.template):
{chr(10).join(tpl_lines)}

모델(props의 model, crowds의 agent에 공통 사용):
{chr(10).join(model_lines)}

장면 필드:
- duration(초), visual(화면 설명 한 줄, 필수), caption, title(선택), hud(선택), time(선택: 조명 시각 "14:00")
- world: {{template: 이름, seed: 정수, params: {{...}}}}
- shot: 템플릿 샷 이름, 또는 직접 카메라:
    {{from: [x,y,z], to: [x,y,z], target: [x,y,z], target_to: [x,y,z], fov: 50}}
    {{type: orbit, center: [x,y,z], radius: r, height: h, from_deg: 150, to_deg: 210, look_height: h2, fov: 45}}  (180도 = 물체 앞쪽 -z에서 봄)
    {{type: topdown, center: [x,y,z], height: 4, height_to: 3}}
- props (정지한 사물, 여러 개):
    {{model: 이름, at: [x,y,z](바닥 중심), yaw: 각도, scale: 배율, color: "#hex", color2: "#hex"}}
    {{shape: box, size: [가로,높이,깊이], at: [x,y,z](밑면 중심), color: "#hex", mat: flat|stone|glass|glow}}
    {{shape: sphere, r: 반지름, at: [x,y,z](중심), color, mat}}     {{shape: cylinder, r, h, at: [x,y,z](밑면 중심), color, mat, axis: y|x|z}}
    mat: glow는 스스로 빛남(불빛·화면), stone은 거친 질감
- crowds (여러 개 / 움직이는 것):
    {{agent, count, area(구역 이름 또는 [x0,z0,x1,z1]), y 또는 y_range: [최저,최고], scale,
      layout: random|grid|queue, facing: random|각도|[x,z]|velocity,
      velocity: [vx,vz] 또는 [vx,vy,vz](m/s, 모두 같은 방향), wander: 속도(제각각 떠다님 — 기체 분자·먼지),
      spin: 회전 속도, bob: 점프 높이, bob_freq, wrap: false(구역 밖으로 나가게), colors: ["#hex"]}}
    분자·입자는 y_range + wander + spin. 탁자 위 물건은 area: table (y 자동 0.9).
    광장 군중 1000~3000, 동물은 넓은 샷에서 scale [1.6, 2.0].
- labels: [{{text: "이름표", at: [x,y,z](가리킬 지점), from: 초, to: 초, color: "#hex"}}]  — 대상 바로 위에 말풍선으로 붙는다
- props·crowds 공통 grow: {{from: 초, to: 초, scale: [0, 1], until: 초}} — 그 시간에 커지며 나타남(번개가 꽂힘, 건물이 솟음)
- effects (한 항목에 하나씩):
    shake {{at, length, amp}}, flash {{at, length}}, tint {{color, at, length}},
    whiteout {{at, length, hold: true면 끝까지 하얗게}}, rewind {{at: 0, length: 1}} (되감기 화면 — 콜드 오픈 다음 장면)
- vfx (3D 특수효과, 좌표는 미터):
    {{type: shockwave, at: [x,y,z], from: 초, to: 초, radius: [0, 40]}}   반투명 충격파 돔(땅 위 반구)
    {{type: orb, at, from, to, radius: [시작, 끝], color, opacity}}       반투명 빛 구 (자기장·보호막·대기)
    {{type: fireball, at, from, to, radius: [0.1, 8], color, core, cool: false면 식지 않음}}  커지는 불덩이
    {{type: mushroom, at, from, to, radius: 15, height: 45}}             솟아오르는 버섯구름
    {{type: embers, at, radius, height, count: 200, color, from}}       빛나는 불티·입자가 솟아오름
- big_text: [{{text: "×1억", from: 초, to: 초, pos: [0.5, 0.42](화면 비율) 또는 at: [x,y,z], color: "#ffd23f", size: 1.0}}]
    — 장면 속에 크고 빛나는 글자. 한 장면에 1개, 6자 이내. caption(자막, 화면 74% 높이)과 겹치지 않게 pos y는 0.35~0.62
- markers: [{{at: [x,y,z], from: 초, text: "낙뢰 지점", color}}]  — 3D 지점에 빛나는 표적 링
- panel (데이터 대시보드, 화면 왼쪽 위):
    {{title: "중심 온도", value: {{from: 0, to: 15000000, unit: "℃", format: kr|"{{:,.0f}}", scale: log|linear, at: 초, length: 초}},
      stats: ["반지름 3km", "시간 0.001초"], bars: [{{label: "에너지", from: 0, to: 0.9(0~1), color: blue|red|green|yellow, text: "90%"}}],
      status: "×0.4 슬로모션" 또는 "일시정지", status_color: red|orange|blue}}
    format: kr이면 "1억 5,000만"처럼 한국식. 큰 수(1000배 이상 차이)는 자동으로 log — 자릿수가 고르게 올라간다
- card (마무리 장면 전용, world 없이 써도 됨 — 앞 장면을 흐리게 깔아 줌):
    {{question: "당신의 선택은?", options: ["선택지1", "선택지2", "선택지3"], cta: "댓글로 번호를 남겨주세요"}}
    card가 있는 장면은 caption이 화면에 표시되지 않는다(질문·선택지가 자막 역할)
- camera: {{fade_out: 0.6}}

## 출력 형식
설명 없이 ```yaml 코드블록 하나만 출력한다. 문자열에 콜론(:)이 있으면 큰따옴표로 감싼다. 최상위 키:
title, style: voxel,
layout: {{title: ["첫 줄(흰색)", "둘째 줄(노란색)"], series: "시리즈명 N편"(선택)}},
look: voxel (기본, 픽셀 복셀) 또는 smooth (매끈한 질감 — 우주·과학 실험처럼 깔끔한 느낌이 어울릴 때),
facts_to_check: ["사실 한 줄 — 출처 URL", ...],
scenes: [...]
배경음악·효과음은 만들지 않는다(무음 영상 — 음악은 업로드할 때 따로 넣는다).

## 참고 예시
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


def ask_cli(system: str, user: str, web_search: bool = True, tools: str | None = None,
            workdir: str | None = None) -> tuple[str, list]:
    """Claude Code 비대화 모드로 질문. 웹 검색·웹 읽기(또는 검수 때 파일 읽기)만 허용 — 파일 수정·명령 실행 불가."""
    cli = claude_cli()
    if not cli:
        raise RuntimeError("Claude Code가 설치되어 있지 않아요.")
    if tools is None:
        tools = "WebSearch,WebFetch" if web_search else ""
    with tempfile.TemporaryDirectory() as tmp0:
        tmp = workdir or tmp0
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


_TEXT_KEY = re.compile(r"^(\s*(?:-\s+)?(?:caption|title|hud|question|cta|status|series):\s+)(.+)$")


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


def ask_with_image(system: str, user: str, image_png: bytes, key: str, max_tokens: int = 12000) -> str:
    import base64
    model = os.environ.get("IF60_PLANNER_MODEL", DEFAULT_MODEL)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(image_png).decode()}},
               {"type": "text", "text": user}]
    data = _post({"model": model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": content}]}, key)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


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
        errs += _validate_broadcast(s, i)
        w = s.get("world")
        if w is None and isinstance(s.get("card"), dict):
            continue   # 마무리 카드 장면은 3D 세계 없이도 됨 (앞 장면을 흐리게 깔아 줌)
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
                if n not in EFFECTS:
                    errs.append(f"장면 {i}: 효과 '{n}' 지원 안 함 ({'/'.join(EFFECTS)}만 가능, 형식: - shake: {{at: 1, length: 1.2, amp: 4}})")
        for j, pr in enumerate(s.get("props") or [], 1):
            if not isinstance(pr, dict):
                errs.append(f"장면 {i} 소품 {j}: 형식 오류"); continue
            if "model" in pr:
                if pr["model"] not in cat["agents"]:
                    errs.append(f"장면 {i} 소품 {j}: model '{pr['model']}' 없음 — 목록에 없으면 shape(box/sphere/cylinder)로 직접 만들 것")
            elif pr.get("shape", "box") not in ("box", "sphere", "cylinder"):
                errs.append(f"장면 {i} 소품 {j}: shape '{pr.get('shape')}' 지원 안 함 (box/sphere/cylinder 중 하나)")
            at = pr.get("at", [0, 0, 0])
            if not (isinstance(at, list) and len(at) == 3):
                errs.append(f"장면 {i} 소품 {j}: at은 [x, y, z]")
        for j, lb in enumerate(s.get("labels") or [], 1):
            if not isinstance(lb, dict) or not lb.get("text") or not (isinstance(lb.get("at"), list) and len(lb["at"]) == 3):
                errs.append(f"장면 {i} 라벨 {j}: {{text: \"...\", at: [x, y, z]}} 형식이어야 함")
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
    lay = spec.get("layout")
    if lay is not None and not isinstance(lay, (dict, bool)):
        errs.append("layout은 {title: [첫 줄, 둘째 줄], series: 시리즈명} 형식")
    if spec.get("look", "voxel") not in ("voxel", "smooth"):
        errs.append(f"look '{spec.get('look')}' 지원 안 함 (voxel 또는 smooth)")
    if target_len and not (target_len * 0.75 <= total <= target_len * 1.3):
        errs.append(f"전체 길이 {total:.0f}초 — 목표 {target_len:.0f}초에 맞춰 장면 길이를 조정")
    return errs


EFFECTS = ("shake", "flash", "tint", "whiteout", "rewind")
VFX_TYPES = ("shockwave", "orb", "fireball", "mushroom", "embers")


def _validate_broadcast(s: dict, i: int) -> list[str]:
    """방송형 연출 필드(vfx, big_text, markers, panel, card) 형식 검사"""
    errs = []
    xyz = lambda v: isinstance(v, list) and len(v) == 3
    for j, v in enumerate(s.get("vfx") or [], 1):
        if not isinstance(v, dict) or v.get("type") not in VFX_TYPES:
            errs.append(f"장면 {i} vfx {j}: type은 {'/'.join(VFX_TYPES)} 중 하나")
        elif not xyz(v.get("at", [0, 0, 0])):
            errs.append(f"장면 {i} vfx {j}: at은 [x, y, z]")
        elif "radius" in v and not (isinstance(v["radius"], (int, float)) or
                                    (isinstance(v["radius"], list) and len(v["radius"]) == 2)):
            errs.append(f"장면 {i} vfx {j}: radius는 숫자 또는 [시작, 끝]")
    bts = s.get("big_text") or []
    for j, b in enumerate([bts] if isinstance(bts, dict) else bts, 1):
        if not isinstance(b, dict) or not b.get("text"):
            errs.append(f"장면 {i} big_text {j}: {{text: \"×1억\", from: 초, pos: [0.5, 0.42]}} 형식")
        elif "at" in b and not xyz(b["at"]):
            errs.append(f"장면 {i} big_text {j}: at은 [x, y, z]")
    mks = s.get("markers") or []
    for j, m in enumerate([mks] if isinstance(mks, dict) else mks, 1):
        if not isinstance(m, dict) or not xyz(m.get("at")):
            errs.append(f"장면 {i} marker {j}: {{at: [x, y, z], from: 초, text: 이름}} 형식")
    pn = s.get("panel")
    if pn is not None:
        if not isinstance(pn, dict):
            errs.append(f"장면 {i}: panel 형식 오류")
        else:
            for b in pn.get("bars") or []:
                if not isinstance(b, dict) or "to" not in b:
                    errs.append(f"장면 {i}: panel.bars 항목은 {{label, from: 0, to: 0~1, color}}")
            if isinstance(pn.get("stats"), str):
                errs.append(f"장면 {i}: panel.stats는 문자열 목록")
    cd = s.get("card")
    if cd is not None:
        if not isinstance(cd, dict) or not isinstance(cd.get("options"), list) or not 2 <= len(cd["options"]) <= 4:
            errs.append(f"장면 {i}: card는 {{question, options: [2~4개], cta}} 형식")
    return errs


def smoke_test(ytxt: str) -> list[str]:
    """장면마다 아주 작은 프레임을 실제로 그려 본다. 렌더링 중 날 오류를 미리 잡아 자동 수정 단계로 보낸다."""
    from . import render as R
    errs: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check.yaml"
        path.write_text(ytxt, encoding="utf-8")
        try:
            spec, style, scenes = R.load_project(str(path), quiet=True)
            font = R.text.find_font(style.font)
        except Exception as e:
            return [f"장면 파일을 불러오다 오류: {type(e).__name__}: {e}"]
        vr = None
        try:
            from .voxel.renderer import VoxelRenderer
            vr = VoxelRenderer(36, 64, samples=0, shadow_size=256)
        except Exception:
            vr = None   # OpenGL을 못 쓰는 환경이면 3D 확인은 건너뜀
        comp = R.Composer(spec, style, font, (36, 64), vr)
        for i, sc in enumerate(scenes, 1):
            try:
                for t in (0.0, sc.duration * 0.5):
                    R.hud_label(sc.hud, t, sc.duration)
                if sc.world and vr is None:
                    continue
                prep, cam = R.prepare_scene(vr, sc, i - 1)
                for t in (0.0, sc.duration * 0.5):
                    comp.frame(sc, t, prep, cam)
                if prep is not None:
                    vr.release(prep)
            except Exception as e:
                errs.append(f"장면 {i} 렌더링 시험 중 오류: {type(e).__name__}: {e}")
        if vr is not None:
            vr.close()
    return errs


def contact_sheet(ytxt: str, cell=(180, 320), cols: int = 4) -> bytes | None:
    """장면마다 가운데 시점 한 장(자막 포함)을 번호와 함께 격자로 모은 PNG. OpenGL이 없으면 None."""
    import io
    from PIL import Image, ImageDraw, ImageFont
    from . import render as R
    try:
        from .voxel.renderer import VoxelRenderer
        vr = VoxelRenderer(cell[0], cell[1], samples=0, shadow_size=1024)
    except Exception:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.yaml"
        path.write_text(ytxt, encoding="utf-8")
        spec, style, scenes = R.load_project(str(path), quiet=True)
        font = R.text.find_font(style.font)
        tiles = []
        comp = R.Composer(spec, style, font, cell, vr)
        for i, sc in enumerate(scenes):
            t = sc.duration * (0.85 if sc.card else 0.55)
            prep, cam = R.prepare_scene(vr, sc, i)
            img = comp.frame(sc, t, prep, cam)
            if prep is not None:
                vr.release(prep)
            tiles.append(Image.fromarray(img).convert("RGB"))
    vr.close()
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell[0] + 6), rows * (cell[1] + 6)), (20, 18, 30))
    d = ImageDraw.Draw(sheet)
    num_font = ImageFont.truetype(R.text.find_font(), 20, index=0)
    for i, im in enumerate(tiles):
        x, y = (i % cols) * (cell[0] + 6), (i // cols) * (cell[1] + 6)
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + 34, y + 28], fill=(255, 220, 60))
        d.text((x + 6, y + 2), str(i + 1), font=num_font, fill=(0, 0, 0))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


REVIEW_PROMPT = """너는 쇼츠 영상의 감독이다. 첨부한 미리보기 시트(contact_sheet.png)에는 장면 번호(노란 칸)별로
실제 렌더링된 화면 한 장과 그 장면의 자막이 함께 보인다. 아래는 장면별 자막과 의도한 visual이다.

{scene_list}

각 장면을 보고 판정하라: 소리 없이 화면만 봐도 자막이 말하는 대상·동작이 보이는가?
- 대상이 화면 밖에 있거나, 너무 작거나, 다른 것에 가려졌거나, 자막과 상관없는 풍경이면 '불일치'.
- 불일치 장면은 world/shot/props/crowds/labels/vfx/big_text/markers/effects를 고쳐 화면이 자막을 직접 보여주게 한다.
  (카메라 거리·scale·at 좌표를 조정하거나, studio/tabletop + props로 대상을 직접 만든다.)
  caption·title·hud·facts_to_check·duration은 바꾸지 않는다.
- 모든 장면이 일치하면 다른 말 없이 "통과"라고만 답한다.
- 하나라도 고치면, 고친 장면이 포함된 전체 YAML을 ```yaml 블록 하나로 출력한다.

현재 YAML:
```yaml
{yaml}
```"""


def review(ytxt: str, spec: dict, backend_call_image, log=print) -> str | None:
    """감독 검수: 고쳐진 YAML 문자열 또는 None(통과/검수 불가)"""
    png = contact_sheet(ytxt)
    if png is None:
        log("  · (이 컴퓨터에서 미리보기를 만들 수 없어 감독 검수를 건너뜀)")
        return None
    lines = []
    for i, sc in enumerate(spec.get("scenes", []), 1):
        lines.append(f"{i}. 자막: {sc.get('caption', '')} | visual: {sc.get('visual', '(없음)')}")
    prompt = REVIEW_PROMPT.format(scene_list="\n".join(lines), yaml=ytxt)
    text = backend_call_image(prompt, png)
    if "```" not in text:
        return None
    return extract_yaml(text)


# ------------------------------------------------------------------ 전체 흐름
def _validated(ytxt: str, cat: dict, length, log) -> tuple[dict | None, list]:
    try:
        spec = yaml.safe_load(ytxt)
        errs = validate(spec, cat, length)
    except yaml.YAMLError as e:
        return None, [f"YAML 문법 오류: {e}"]
    if not errs:
        log("  · 장면별 시험 렌더링으로 오류 확인 중")
        errs = smoke_test(ytxt)
    return spec, errs


def plan(topic: str, length: int = 60, *, interactive: bool = True, web_search: bool = True,
         review_pass: bool = True, log=print) -> tuple[str, dict, list]:
    """주제 → 검증된 장면 YAML 텍스트, 파싱된 spec, 인용 목록"""
    backend = choose_backend(interactive)
    if backend == "api":
        key = require_key(interactive)
        call = lambda u, ws: ask(system, u, key, web_search=ws)
        call_image = lambda u, png: ask_with_image(system, u, png, key)
        log("  · Claude API 사용 (사용량만큼 별도 과금)")
    else:
        call = lambda u, ws: ask_cli(system, u, web_search=ws)

        def call_image(u, png):
            with tempfile.TemporaryDirectory() as wd:
                (Path(wd) / "contact_sheet.png").write_bytes(png)
                u2 = "먼저 Read 도구로 현재 폴더의 contact_sheet.png 를 열어 본 뒤 답하라.\n\n" + u
                return ask_cli(system, u2, tools="Read", workdir=wd)[0]
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
        spec, errs = _validated(ytxt, cat, length, log)
        if not errs:
            break
        log(f"▶ 장면 파일 자동 수정 중 ({attempt + 1}/3): " + "; ".join(errs[:3]) + (" …" if len(errs) > 3 else ""))
        fix = (f"아래 YAML을 렌더러가 읽을 수 있게 고쳐. 문제:\n- " + "\n- ".join(errs) +
               f"\n\n내용(대본·사실)은 유지하고 문제만 고쳐서 ```yaml 블록 하나로 출력해.\n\n```yaml\n{ytxt}```")
        text2, _ = call(fix, False)
        ytxt = extract_yaml(text2)
    else:
        raise RuntimeError("장면 파일을 자동으로 고치지 못했어요: " + "; ".join(errs))
    if review_pass:
        log("▶ 감독 검수: 장면 미리보기를 보며 화면과 자막이 맞는지 확인 중 (1~3분)")
        t1 = time.time()
        try:
            revised = review(ytxt, spec, call_image, log)
        except Exception as e:
            revised = None
            log(f"  · 검수 중 문제가 있어 원래 연출로 진행: {e}")
        if revised:
            spec2, errs2 = _validated(revised, cat, length, log)
            if not errs2:
                ytxt, spec = revised, spec2
                log(f"  · 화면이 안 맞는 장면을 고쳤어요 ({time.time() - t1:.0f}초)")
            else:
                log("  · 수정안에 오류가 있어 원래 연출을 유지: " + "; ".join(errs2[:2]))
        else:
            log(f"  · 통과 ({time.time() - t1:.0f}초)")
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
