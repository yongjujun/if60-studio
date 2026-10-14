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


def save_key(key: str, var: str = "ANTHROPIC_API_KEY") -> None:
    """.env에 KEY=값 저장 (같은 이름이 있으면 바꿈). .env는 GitHub에 올라가지 않는다."""
    env = ROOT / ".env"
    lines = []
    if env.exists():
        lines = [l for l in env.read_text(encoding="utf-8").splitlines() if not l.startswith(f"{var}=")]
    lines.append(f"{var}={key.strip()}")
    os.environ[var] = key.strip()
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
    ex0 = ROOT / "scenes" / "juvenile_law_abolished.yaml"
    ex1 = ROOT / "scenes" / "lightning_power.yaml"
    ex2 = ROOT / "scenes" / "nitrogen_snack.yaml"
    example_txt = "\n".join(
        f"### {label}\n```yaml\n{p.read_text(encoding='utf-8')}```" for label, p in
        (("예시 1 — 사회 풍자(satire) + 4단 구성 + keyframes 9장 (사회·제도 주제, 찬반 균형)", ex0),
         ("예시 2 — 방송형 연출 (콜드 오픈·대시보드·VFX·마무리 카드)", ex1),
         ("예시 3 — 사물을 화면에 직접 만들어 보여주기", ex2)) if p.exists())
    return f"""너는 쇼츠 채널 '만물제작소'(유튜브 'IF 60')의 작가 겸 연출가다. 주제는 무엇이든 올 수 있다(사물, 음식, 몸, 동물, 우주, 사회, 제도, 상상).
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

## 장르별 톤 — 먼저 주제의 장르를 정해 최상위 genre: 에 적는다
- genre: satire (사회 풍자) → **통쾌하게.** 시청자가 평소 당하기만 하던 쪽(사기·갑질·허세·특권)이
  가상 제도 앞에서 꼼짝 못 하게 되는 '사이다' 순간을 전개 구간에 2~3번 만든다. 웃음의 방향은 언제나 위로(힘 있는 쪽, 제도, 위선).
  단, 통쾌함 다음 반전에서 그 제도의 허점·부작용도 정직하게 보여준다(그래서 결말 질문이 살아난다).
- genre: science (과학) → **흥미롭게.** 실제 수치로 직접 계산한 의외의 결과(예: 피부로 하는 광합성 = 하루 몇 kcal)를
  big_text·대시보드로 보여주고, "그래서 실제로는?"을 매 장면 한 단계씩 밝혀 간다. 과장 없이 숫자가 놀라게 한다.
- genre: absurd (황당한 가정) → **진지하게.** 말도 안 되는 가정을 정부 브리핑·다큐·공식 통계 발표처럼 정색하고 설명한다.
  실제 연구·통계를 근거로 "만약 그렇다면 이렇게 된다"를 논리적으로 밀고 나가며, 진지할수록 웃기다.

## 공통 톤: 정색한 블랙코미디
요즘 시청자는 정보만 나열하는 영상보다, 사실을 정확히 짚으면서 그 사실이 만드는 '부조리'에 피식 웃게 되는 영상에 끌린다.
- 내레이터는 진지한 다큐 톤(정색)을 유지한 채 황당한 결론을 담담하게 말한다. 과장된 리액션·유행어 남발 금지.
- 웃음 장치 (영상마다 4~6개 섞기):
  · 담담한 반전: 그럴듯한 설명 → 마지막 한 마디로 뒤집기 ("…그래서 1년 안에 끝나긴 합니다. 주말을 반납하면요.")
  · 정색한 공식 발표체: 뉴스 자막띠(chyron)·도장(stamp)·대시보드로 황당한 상황을 '공식'처럼 보여주기
  · 숫자로 웃기기: 실제 수치를 계산해 나온 어이없는 결과 (1인당 843건, 하루 2.3건)
  · 점점 커지는 황당함: 상상 속 결과를 한 단계씩 키워 가기
  · 콜백: 초반의 소품·대사를 끝에서 다시 꺼내 마무리
  · 펀치라인 직후 beat(0.5~1초 정적) + punch(줌인) 또는 stamp — 웃을 틈을 준다
- 풍자의 과녁은 제도·관료주의·모순·우리 모두의 습관이다. 피해자·약자·특정 집단·실존 인물은 절대 조롱하지 않는다.
- 사회·정치적으로 의견이 갈리는 주제(법·제도·정책)는 한쪽 편을 들지 않는다:
  찬성 쪽이 기대하는 효과와 반대 쪽이 걱정하는 문제를 둘 다 사실로 보여주고, 양쪽의 부조리를 고르게 비튼다.
  현재 실제 상황(법·정부안이 어디까지 왔는지)을 한 장면으로 정확히 밝히고, 결론은 마지막 카드에서 시청자에게 묻는다.
- "~라면?" 상상 장면은 상상임이 분명하게 표현한다(자막띠 tag: "가상 상황" 등). 실제 사실과 상상을 섞어 오해하게 만들지 않는다.
- 범죄·사고를 다룰 때 수법·잔혹한 묘사는 넣지 않는다. 아이·피해자를 웃음거리로 쓰지 않는다.

## 콘텐츠 공식 (조회수가 검증된 방송형 구조 — 채널 '만물제작소')
- 길이는 60~120초(기본 약 90초, 12~18장면). 장면마다 act: 를 붙여 4단 구성을 완성한다 (순서대로, 넷 다 필수):
  ①act: 도입 — 3초 훅(콜드 오픈) + 되감기 + 실제 사실·수치로 설정 (전체의 약 25%)
  ②act: 전개 — 가정이 현실이 되면 벌어지는 일, 단계마다 커지는 황당함·통쾌함 (약 35%)
  ③act: 반전 — 예상을 뒤집는 사실·부작용·의외의 비교 (약 25%)
  ④act: 결말 — 현실 확인 + 기억에 남는 한 줄(payoff) + 질문 카드 (약 15%)
- 화면 틀: 최상위 layout으로 상단에 두 줄 제목(첫 줄 흰색, 둘째 줄 노란색 — 각 10자 안팎)이 영상 내내 고정된다.
  그래서 장면별 title은 쓰지 않는다(필요하면 짧게 한 번만). 채널명은 레이아웃이 알아서 작게 넣는다.
- ★콜드 오픈: 1번 장면(2~3초)은 영상에서 가장 센 순간(절정·결과)을 먼저 보여준다 — big_text로 핵심 숫자,
  vfx·shake·whiteout(hold: true)로 강하게 끝낸다. 2번 장면은 effects에 rewind {{at: 0}}를 넣고
  "어떻게 이렇게 됐을까요? 처음부터 볼게요." 같은 caption으로 시작점부터 다시 보여준다.
- 본편: 한 장면 = 한 가지 사실. 4~6초씩. 숫자가 나오면 panel(대시보드)의 카운터가 0부터 굴러 올라가게,
  핵심 수치는 big_text로 크게 빛나게, 가리키는 지점은 markers로 표시한다. 비유는 한국인에게 익숙한 것으로.
- 마지막 장면은 card(질문 + 번호 선택지 3개 + "댓글로 번호를 남겨주세요").
- HUD(우측 상단 배지)는 시계·짧은 문구에만: 시계 {{from: "05:00", to: "05:30", step: 1}}, 고정 문구 "공기 = 질소 78%"
- 말투: 친근한 존댓말. caption은 한 장면 45자 이내, 숫자는 구체적으로.
- 사실은 검색으로 확인된 것만. 불확실하면 쓰지 않는다. 출처를 facts_to_check에 URL과 함께.
- 위험한 행동을 따라 하게 만드는 내용, 실존 인물·상표·저작권 캐릭터는 넣지 않는다.

## ★시청 유지 설계: 3초 훅 · 20초 반전 · 기억에 남는 끝
쇼츠는 세 지점에서 시청자를 잃는다. 각 지점을 맡는 장면에 `role:`을 붙인다 (렌더링에는 영향 없음, 구조 검사용).
- role: hook (1번 장면) — 스크롤을 멈추게 한다. 3초 안에 끝난다: caption(narration) 20자 안팎.
  · 설명으로 시작하지 않는다. "궁금해서 못 넘기는 한 문장" — 결과를 먼저 던지고 이유는 숨긴다
    (예: "만 10세, 지우개 절도. 판결은…", "이 캔 하나로 서울이 멈춥니다.").
  · 첫 프레임부터 화면이 움직이고(vfx·punch·grow), 핵심 숫자나 도장이 big_text로 크게 뜬다.
- role: twist (영상 시작 후 약 15~30초에 시작하는 장면, 보통 4~6번째) — 이탈을 막는 중간 반전.
  · 지금까지 쌓은 예상을 한 번 뒤집는다: "그런데…", "문제는…", 예상과 반대인 숫자, 상황의 판이 커짐.
  · 반전 직후 beat 또는 punch로 한 박자 준다. 후반(끝에서 2~4번째 장면)에도 반전을 한 번 더 둘 수 있다.
- role: payoff (card 바로 앞, 끝에서 2~3번째 안) — 기억에 남을 한 줄. 영상이 끝나도 머리에 남는 문장이다.
  · 첫 장면(hook)의 소품·숫자·대사를 다시 꺼내는 콜백이면 가장 좋다(처음 장면으로 되돌아가게 → 반복 재생).
  · 짧고 단정하게, 펀치라인 뒤 beat 0.8초.
- 마지막 card의 질문은 hook에서 던진 궁금증과 이어지게 한다.
- 장면 시작 시각은 앞 장면들의 대사 길이(글자 수 ÷ 6.5 + 0.6초)를 더해 계산한다.

## ★웃음 설계: 1절로 끝내지 말고 2절·3절, 삼천포까지 갔다가 돌아온다
정보를 한 번 말하고 끝내면 재미가 없다. 가정의 결과 → 그 결과의 결과 → 또 그 결과로 꼬리를 물며 점점 엉뚱한 곳까지 간 뒤,
정색하고 본론으로 돌아와 결론을 낸다. 장면에 tangent: 1|2|3 과 role: snapback 을 붙인다 (전개~반전 구간, 각 절 1~3장면).
- tangent: 1 — 1절: 누구나 예상하는 직접적인 결과. (예: 머리 위 전과 표시 → 보이스피싱범이 출근길 지하철에서 바로 들통)
- tangent: 2 — 2절: 그 결과가 낳은 엉뚱한 부작용. 이미 살짝 옆길. (→ 사기꾼들이 모자를 쓰기 시작, 여름에도 털모자 매출 폭증)
- tangent: 3 — 3절(삼천포): 주제에서 한참 벗어났지만 논리적으로는 이어지는 황당한 결과. 가장 진지하게, 실제 숫자까지 동원해서.
  (→ 모자 쓴 사람 = 수상한 사람이 되자, 모자 업계가 '투명 모자'를 출시. 실제 국내 모자 시장 규모 ○○억 원이 위기)
- role: snapback — 정색하고 복귀하는 한 문장. "…아무튼." "모자 얘기는 여기까지 하고요." 바로 뒤에 반전이 온다.
- 3절의 소재를 결말 payoff에서 콜백으로 다시 꺼내면 최고 (예: 마지막에 투명 모자를 쓴 진짜 범인).
- 장면마다 gag: 에 그 장면의 웃음 장치를 하나 적는다 (아래 이름 중 하나). 웃음 없는 장면은 gag를 쓰지 않는다.
  장면의 절반 이상에 gag가 있어야 하고, 종류는 3가지 이상 섞는다:
  정색 반전 · 공식 발표체 · 숫자 개그 · 과잉 진지 · 삼천포 · 콜백 · 의외의 비교 · 3의 법칙 · 현실 공감 · 한 박자 늦은 펀치
- 웃긴 대사 쓰는 법:
  · 펀치라인은 문장 맨 끝 단어에 둔다. ("1년이면 끝나긴 합니다. 주말을 반납하면요.")
  · 웃음을 설명하지 않는다. "웃기게도", "황당하게도" 같은 말 금지. 감탄사·유행어·물음표 남발 금지.
  · 추상어보다 구체적인 사물 하나가 웃기다 ("불편"보다 "젤리 세 알", "혼란"보다 "털모자 매출").
  · 3의 법칙: 평범한 것 둘 + 비튼 것 하나. ("회의 줄고, 야근 줄고, 이혼은 늘었습니다.")
  · 내레이터는 끝까지 정색. 상황이 황당할수록 목소리는 더 차분하게.

## ★AI 이미지 6~10장으로 시각화 가능하게 (keyframes)
이 대본은 3D 복셀로도 렌더링되지만, AI 이미지 6~10장만으로도 영상을 만들 수 있어야 한다.
- 최상위 keyframes: 에 핵심 그림 6~10장을 정의한다. 각 장면은 image: 번호 로 그중 하나를 가리킨다.
  같은 그림을 이어지는 2~3장면이 함께 쓴다(줌·자막·도장·대시보드만 달라짐). 장면마다 새 그림을 만들지 않는다.
- 그림 하나 = 한 장소 + 한 주인공 + 한 동작. 글자 없이도 무슨 상황인지 보이게, 세로 9:16 구도.
- keyframes 형식: {{id: 1, desc: "한국어 한 줄 설명", prompt: "영어 이미지 생성 프롬프트 (장소, 주인공, 동작, 구도, 조명, vertical 9:16, no text)"}}
  prompt에는 실존 인물·상표·글자를 넣지 않고, 모든 그림이 같은 화풍이 되도록 끝에 같은 화풍 문구를 붙인다.
- 장면의 visual(복셀 연출)은 그 장면이 가리키는 keyframe 그림과 같은 내용이어야 한다.

## 렌더러가 지원하는 것 (이 목록 밖의 이름을 쓰면 렌더링이 실패한다)
템플릿(world.template):
{chr(10).join(tpl_lines)}

모델(props의 model, crowds의 agent에 공통 사용):
{chr(10).join(model_lines)}

장면 필드:
- duration(초), act(도입|전개|반전|결말, 필수), image(keyframes 번호, 필수), role(선택: hook|twist|payoff|snapback),
  tangent(선택: 1|2|3 — 웃음 설계의 절), gag(선택: 웃음 장치 이름), visual(화면 설명 한 줄, 필수), caption(자막이자 내레이션), narration(선택: 읽는 방식이 다를 때), title(선택), hud(선택), time(선택: 조명 시각 "14:00")
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
- 풍자 장치:
    big_text에 style: stamp → 빨간 도장이 '쾅' 찍힘 (예: {{text: "기각", style: stamp, from: 2.0, angle: -8}}) — 판결·승인·검토 중 같은 공식 느낌
    chyron: [{{tag: "속보"|"가상 상황"|"실제 상황", text: "정부, 촉법소년 제도 전면 폐지", from: 초}}] — 뉴스 하단 자막띠
    effects에 punch {{at: 초, length: 1.0, zoom: 1.3, center: [0.5, 0.45]}} — 순간 줌인 후 정지 (어이없는 순간)
    beat: 0.8 — 이 장면 대사가 끝난 뒤 0.8초 정적 (펀치라인 다음에)
- card (마무리 장면 전용, world 없이 써도 됨 — 앞 장면을 흐리게 깔아 줌):
    {{question: "당신의 선택은?", options: ["선택지1", "선택지2", "선택지3"], cta: "댓글로 번호를 남겨주세요"}}
    card가 있는 장면은 caption이 화면에 표시되지 않는다(질문·선택지가 자막 역할)
- camera: {{fade_out: 0.6}}

## 출력 형식
설명 없이 ```yaml 코드블록 하나만 출력한다. 문자열에 콜론(:)이 있으면 큰따옴표로 감싼다. 최상위 키:
title, genre: satire|science|absurd, style: voxel,
layout: {{title: ["첫 줄(흰색)", "둘째 줄(노란색)"], series: "시리즈명 N편"(선택)}},
look: voxel (기본, 픽셀 복셀) 또는 smooth (매끈한 질감 — 우주·과학 실험처럼 깔끔한 느낌이 어울릴 때),
facts_to_check: ["사실 한 줄 — 출처 URL", ...],
keyframes: [{{id, desc, prompt}}, ...] (6~10개),
scenes: [...]
배경음악·효과음은 만들지 않는다. 대신 caption을 AI 목소리(TTS)가 그대로 읽는다:
- caption은 소리 내어 읽었을 때 자연스러운 구어체 한두 문장으로 쓴다.
- caption에 기호·영문 단위·특이한 표기가 있으면 장면에 narration을 따로 써서 읽는 그대로 적는다.
  예: caption "약 27,700℃, 태양 표면의 5배!" → narration "약 이만 칠천 도, 태양 표면의 다섯 배!"
  예: caption "3억 V, 3만 A" → narration "삼억 볼트, 삼만 암페어"
- 장면 길이는 목소리 길이에 맞춰 자동으로 늘거나 줄어든다(연출 시각도 같은 비율로 이동). duration은 대략 잡으면 된다.
  한국어 내레이션은 1초에 약 6~7글자이므로 caption 글자 수 ÷ 6.5 + 0.6초 정도로 잡는다.
  영상 전체 길이는 사실상 대사 분량이 정한다: 90초면 narration(없으면 caption) 글자 수 합계 약 550~620자.

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


# ------------------------------------------------------------------ 시청 유지 구조
HOOK_MAX_SEC = 3.5          # 1번 장면(훅)은 이 안에 끝나야 함
TWIST_WINDOW = (15.0, 30.0)  # 중간 반전 장면이 시작해야 하는 구간(초)
ROLES = ("hook", "twist", "payoff", "snapback")
GAGS = ("정색 반전", "공식 발표체", "숫자 개그", "과잉 진지", "삼천포", "콜백", "의외의 비교", "3의 법칙",
        "현실 공감", "한 박자 늦은 펀치")
ACTS = ("도입", "전개", "반전", "결말")
LENGTH_RANGE = (60.0, 120.0)   # 영상 전체 길이(초)
KEYFRAME_RANGE = (6, 10)       # AI 이미지로 만들 때 필요한 그림 수
GENRES = {"satire": "사회 풍자 — 통쾌하게", "science": "과학 — 흥미롭게", "absurd": "황당한 가정 — 진지하게"}


def spoken_seconds(scene: dict) -> float:
    """장면 길이 추정 — 실제 영상은 목소리 길이에 맞춰지므로 대사 글자 수로 계산 (card는 duration)"""
    if isinstance(scene.get("card"), dict):
        try:
            return float(scene.get("duration", 5))
        except (TypeError, ValueError):
            return 5.0
    text = str(scene.get("narration") or scene.get("caption") or "")
    if not text.strip():
        try:
            return float(scene.get("duration", 4))
        except (TypeError, ValueError):
            return 4.0
    return len(re.sub(r"\s", "", text)) / 6.5 + 0.6


def retention_issues(spec: dict) -> list[str]:
    """3초 훅 · 20초 반전 · 기억에 남는 끝 구조 검사. 렌더링 오류가 아니라 '더 붙잡는 영상'을 위한 권고."""
    scenes = [s for s in (spec or {}).get("scenes") or [] if isinstance(s, dict)]
    if len(scenes) < 4:
        return []   # 아주 짧은 시험용 영상은 검사하지 않음
    issues, starts, t = [], [], 0.0
    for s in scenes:
        starts.append(t)
        t += spoken_seconds(s)
    for i, s in enumerate(scenes, 1):
        if s.get("role") is not None and s.get("role") not in ROLES:
            issues.append(f"장면 {i}: role '{s.get('role')}'는 쓸 수 없음 (hook/twist/payoff 중 하나)")
    first = scenes[0]
    if first.get("role") != "hook":
        issues.append("1번 장면에 role: hook이 없음 — 첫 장면을 3초 훅으로 만들 것")
    hook_len = spoken_seconds(first)
    if hook_len > HOOK_MAX_SEC:
        issues.append(f"훅(1번 장면) 대사가 약 {hook_len:.1f}초 — {HOOK_MAX_SEC:.0f}초 안에 끝나게 20자 안팎으로 줄일 것")
    lo, hi = TWIST_WINDOW
    twists = [(i, starts[i - 1]) for i, s in enumerate(scenes, 1) if s.get("role") == "twist"]
    if not twists:
        issues.append(f"중간 반전(role: twist)이 없음 — 시작 후 {lo:.0f}~{hi:.0f}초에 예상을 뒤집는 장면을 둘 것")
    elif not any(lo <= st <= hi for _, st in twists):
        where = ", ".join(f"{i}번 장면 {st:.0f}초" for i, st in twists)
        issues.append(f"중간 반전이 {lo:.0f}~{hi:.0f}초 구간에 없음 (지금: {where}) — 앞 장면 대사를 줄이거나 늘려서 맞출 것")
    tail = scenes[-3:] if isinstance(scenes[-1].get("card"), dict) else scenes[-2:]
    if not any(s.get("role") == "payoff" for s in tail):
        issues.append("끝에 role: payoff 장면이 없음 — 마지막 card 바로 앞에 기억에 남는 한 줄(가능하면 첫 장면 콜백)을 둘 것")
    issues += structure_issues(spec, scenes, t)
    issues += humor_issues(scenes)
    return issues


def humor_issues(scenes: list) -> list[str]:
    """웃음 설계 검사: 1절→2절→3절(삼천포)→복귀, 장면 절반 이상에 웃음 장치"""
    issues = []
    tang = []
    for i, s in enumerate(scenes, 1):
        v = s.get("tangent")
        if v is None:
            continue
        try:
            tang.append((i, int(v)))
        except (TypeError, ValueError):
            issues.append(f"장면 {i}: tangent는 1, 2, 3 중 하나")
    levels = [v for _, v in tang]
    missing = [n for n in (1, 2, 3) if n not in levels]
    if missing:
        issues.append(f"웃음 설계 {', '.join(f'{n}절' for n in missing)}이 없음 — 결과의 결과로 꼬리를 물어 "
                      "1절(직접 결과) → 2절(엉뚱한 부작용) → 3절(삼천포)까지 갈 것")
    elif levels != sorted(levels):
        issues.append("tangent 순서가 뒤섞임 — 1절 → 2절 → 3절 순서로 점점 멀리")
    snaps = [i for i, s in enumerate(scenes, 1) if s.get("role") == "snapback"]
    last3 = max((i for i, v in tang if v == 3), default=None)
    if not snaps:
        issues.append("role: snapback이 없음 — 3절(삼천포) 뒤에 정색하고 본론으로 돌아오는 한 문장을 둘 것")
    elif last3 and not any(i > last3 for i in snaps):
        issues.append("snapback이 3절보다 앞에 있음 — 삼천포 장면들이 끝난 직후에 둘 것")
    end = [i for i, s in enumerate(scenes, 1) if s.get("act") == "결말"]
    if tang and end and max(i for i, _ in tang) >= min(end):
        issues.append("tangent 장면이 결말 구간에 있음 — 1~3절은 전개·반전 구간 안에서 끝낼 것")
    talk = [s for s in scenes if not isinstance(s.get("card"), dict)]
    gags = [s.get("gag") for s in talk if s.get("gag")]
    bad = sorted({g for g in gags if g not in GAGS})
    if bad:
        issues.append(f"gag 이름 '{', '.join(map(str, bad))}'는 목록에 없음 ({' / '.join(GAGS)})")
    good = [g for g in gags if g in GAGS]
    need = (len(talk) + 1) // 2
    if len(good) < need:
        issues.append(f"웃음 장치(gag)가 있는 장면이 {len(good)}개 — 장면 절반 이상({need}개)에 웃음 장치를 넣을 것")
    elif len(set(good)) < 3:
        issues.append("웃음 장치가 한두 종류뿐 — 3가지 이상 섞을 것")
    return issues


# ------------------------------------------------------------------ 개그 작가 다듬기
PUNCHUP_SYSTEM = """너는 한국 예능·쇼츠 개그 작가다. 정색한 다큐 내레이션으로 사람을 웃기는 쇼츠 대본의 대사만 더 웃기게 다듬는다.
규칙:
- 장면의 뜻·사실·순서는 그대로. 아라비아 숫자는 한 글자도 바꾸거나 빼지 않는다. 화면(visual)과 어긋나는 말을 넣지 않는다.
- 펀치라인은 문장 맨 끝 단어에. 웃음을 설명하지 않는다("웃기게도" 금지). 유행어·감탄사·이모지·과한 물음표 금지.
- 추상어 대신 구체적인 사물·숫자, 3의 법칙(평범 둘 + 비튼 하나), 한 박자 늦은 펀치, 앞 장면 콜백을 활용한다.
- 내레이터는 끝까지 정색. 3절(삼천포)일수록 더 진지하게. snapback은 짧고 건조하게 ("…아무튼.").
- 웃음의 과녁은 제도·위선·힘 있는 쪽·우리 모두의 습관. 실존 인물·상표·특정 집단·피해자·외모를 조롱하지 않는다.
- 길이: 각 caption 45자 이내, 원래 대사 길이의 ±30% 안. 1번 장면(hook)은 20자 안팎.
- caption에 숫자·기호·영문이 있으면 읽는 그대로 한글로 쓴 narration도 함께 준다.
- 이미 충분히 웃기면 그 장면은 빼도 된다.
출력: 설명 없이 JSON 배열 하나만. [{"n": 장면번호, "caption": "...", "narration": "...(필요할 때)"}]"""

_NEEDS_READING = re.compile(r"[0-9A-Za-z%℃°㎡㎞㎏~·/+×]")


def _digits(s: str) -> list[str]:
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", s or ""))


def punchup_prompt(spec: dict) -> str:
    lines = [f"제목: {spec.get('title', '')}  장르: {GENRES.get(spec.get('genre'), spec.get('genre', ''))}", ""]
    for i, s in enumerate(spec.get("scenes") or [], 1):
        if not isinstance(s, dict) or isinstance(s.get("card"), dict):
            continue
        tags = [str(s.get(k)) if k != "tangent" else f"{s.get(k)}절" for k in ("act", "role", "tangent", "gag") if s.get(k)]
        lines.append(f"{i}. [{' · '.join(tags)}] 화면: {s.get('visual', '')}")
        lines.append(f"   caption: {s.get('caption', '')}")
        if s.get("narration"):
            lines.append(f"   narration: {s['narration']}")
    return "\n".join(lines)


def apply_punchup(spec: dict, text: str) -> tuple[dict, int]:
    """개그 작가의 JSON 답을 안전하게 반영. (새 spec, 바뀐 장면 수)"""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return spec, 0
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return spec, 0
    import copy
    new = copy.deepcopy(spec)
    scenes = new.get("scenes") or []
    changed = 0
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            n = int(it.get("n"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= len(scenes) or not isinstance(scenes[n - 1], dict) or isinstance(scenes[n - 1].get("card"), dict):
            continue
        s = scenes[n - 1]
        old, cap = str(s.get("caption") or ""), str(it.get("caption") or "").strip()
        nar = str(it.get("narration") or "").strip()
        if not cap or cap == old:
            continue
        if _digits(cap) != _digits(old):                       # 숫자(사실)가 바뀌면 버림
            continue
        if len(cap) > max(50, int(len(old) * 1.3)):
            continue
        if _NEEDS_READING.search(cap) and not nar:            # 읽는 방식이 필요한데 없으면 버림
            continue
        s["caption"] = cap
        if nar:
            s["narration"] = nar
        else:
            s.pop("narration", None)
        changed += 1
    return new, changed


def dump_yaml(spec: dict) -> str:
    return yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=1000, default_flow_style=None)


def structure_issues(spec: dict, scenes: list, total: float) -> list[str]:
    """만물제작소 제작 조건: 60~120초 · 도입→전개→반전→결말 · AI 이미지 6~10장 · 장르"""
    issues = []
    lo, hi = LENGTH_RANGE
    if not lo <= total <= hi:
        issues.append(f"대사 기준 전체 길이 약 {total:.0f}초 — {lo:.0f}~{hi:.0f}초 안에 들게 대사를 늘리거나 줄일 것")
    if spec.get("genre") not in GENRES:
        issues.append("최상위 genre가 없거나 잘못됨 (satire | science | absurd 중 하나)")
    acts = [s.get("act") for s in scenes]
    bad = [i for i, a in enumerate(acts, 1) if a not in ACTS]
    if bad:
        issues.append(f"act가 없거나 잘못된 장면: {', '.join(map(str, bad[:6]))} (도입/전개/반전/결말 중 하나)")
    else:
        order = [ACTS.index(a) for a in acts]
        if order != sorted(order):
            issues.append("act 순서가 뒤섞임 — 도입 → 전개 → 반전 → 결말 순서로")
        missing = [a for a in ACTS if a not in acts]
        if missing:
            issues.append(f"{', '.join(missing)} 구간이 없음 — 도입·전개·반전·결말을 모두 갖출 것")
    kfs = spec.get("keyframes")
    ids = set()
    if not isinstance(kfs, list):
        issues.append(f"keyframes가 없음 — AI 이미지 {KEYFRAME_RANGE[0]}~{KEYFRAME_RANGE[1]}장 목록을 만들 것")
    else:
        for k in kfs:
            if isinstance(k, dict) and k.get("id") is not None and str(k.get("prompt") or "").strip():
                ids.add(str(k["id"]))
        if len(ids) != len(kfs):
            issues.append("keyframes 항목은 모두 {id, desc, prompt} 형식이고 id가 겹치지 않아야 함")
        if not KEYFRAME_RANGE[0] <= len(kfs) <= KEYFRAME_RANGE[1]:
            issues.append(f"keyframes가 {len(kfs)}장 — {KEYFRAME_RANGE[0]}~{KEYFRAME_RANGE[1]}장으로 맞출 것 (이어지는 장면끼리 그림을 공유)")
        used = {str(s.get("image")) for s in scenes if s.get("image") is not None}
        nopic = [i for i, s in enumerate(scenes, 1)
                 if not isinstance(s.get("card"), dict) and str(s.get("image")) not in ids]
        if nopic:
            issues.append(f"image가 없거나 keyframes에 없는 번호인 장면: {', '.join(map(str, nopic[:6]))}")
        unused = sorted(ids - used)
        if unused:
            issues.append(f"어느 장면도 쓰지 않는 keyframe: {', '.join(unused)}")
    return issues


EFFECTS = ("shake", "flash", "tint", "whiteout", "rewind", "punch")
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
    chs = s.get("chyron") or []
    for j, c in enumerate([chs] if isinstance(chs, dict) else chs, 1):
        if not isinstance(c, dict) or not c.get("text"):
            errs.append(f"장면 {i} chyron {j}: {{tag: 속보, text: \"...\", from: 초}} 형식")
    if "beat" in s and not isinstance(s.get("beat"), (int, float)):
        errs.append(f"장면 {i}: beat는 초(숫자)")
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


def plan(topic: str, length: int = 90, *, interactive: bool = True, web_search: bool = True,
         review_pass: bool = True, log=print, genre: str | None = None,
         angle: str | None = None, punchup: bool = True, tangents: str | None = None) -> tuple[str, dict, list]:
    """주제 → 검증된 장면 YAML 텍스트, 파싱된 spec, 인용 목록"""
    backend = choose_backend(interactive)
    if backend == "api":
        key = require_key(interactive)
        call = lambda u, ws: ask(system, u, key, web_search=ws)
        call_light = lambda sysm, u: ask(sysm, u, key, web_search=False, max_tokens=6000)[0]
        call_image = lambda u, png: ask_with_image(system, u, png, key)
        log("  · Claude API 사용 (사용량만큼 별도 과금)")
    else:
        call = lambda u, ws: ask_cli(system, u, web_search=ws)
        call_light = lambda sysm, u: ask_cli(sysm, u, web_search=False)[0]

        def call_image(u, png):
            with tempfile.TemporaryDirectory() as wd:
                (Path(wd) / "contact_sheet.png").write_bytes(png)
                u2 = "먼저 Read 도구로 현재 폴더의 contact_sheet.png 를 열어 본 뒤 답하라.\n\n" + u
                return ask_cli(system, u2, tools="Read", workdir=wd)[0]
        log("  · Claude Code + 내 Claude 구독 사용 (추가 비용 없음, 구독 사용량에서 차감)")
    cat = catalog()
    system = system_prompt(cat)
    user = f"주제: {topic}\n목표 길이: 약 {length}초 (60~120초 안)\n"
    if genre in GENRES:
        user += f"장르: {genre} ({GENRES[genre]})\n"
    if angle:
        user += f"방향 힌트: {angle}\n"
    if tangents:
        user += f"삼천포 아이디어(참고용, 더 웃긴 게 떠오르면 바꿔도 됨): {tangents}\n"
    user += ("조건: 첫 3초 훅 · 도입→전개→반전→결말 완성 · 1절→2절→3절(삼천포)→복귀로 웃기기 · "
             "AI 이미지 6~10장(keyframes)으로 시각화 가능.\n"
             "먼저 웹 검색으로 핵심 사실과 수치를 확인한 뒤, 지정한 형식의 YAML만 출력해.")
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
    for attempt in range(2):
        issues = retention_issues(spec)
        if not issues:
            break
        log(f"▶ 제작 조건 맞추는 중 ({attempt + 1}/2 — 3초 훅 · 4단 구성 · 삼천포 웃음 · 길이 · 이미지): "
            + "; ".join(issues[:2]) + (" …" if len(issues) > 2 else ""))
        fix = ("아래 YAML이 제작 조건을 어겼어. 문제:\n- " + "\n- ".join(issues) +
               "\n\n시스템 프롬프트의 '시청 유지 설계'·'4단 구성'·'웃음 설계'·'keyframes' 규칙을 따른다. 확인된 사실·수치·출처는 바꾸지 말고, "
               "장면 순서·장면 추가(삼천포 2·3절)·대사·role·act·tangent·gag·image·keyframes만 손봐. ```yaml 블록 하나로 출력해.\n\n```yaml\n" + ytxt + "```")
        try:
            ytxt2 = extract_yaml(call(fix, False)[0])
            spec2, errs2 = _validated(ytxt2, cat, length, log)
            if not errs2 and len(retention_issues(spec2)) < len(issues):
                ytxt, spec = ytxt2, spec2
                log("  · 개선됨")
            else:
                log("  · 수정안이 더 낫지 않아 이전 대본 유지" + (": " + "; ".join(errs2[:2]) if errs2 else ""))
        except Exception as e:
            log(f"  · 다듬기 중 문제가 있어 이전 대본으로 진행: {e}")
            break
    left = retention_issues(spec)
    if left:
        log("  ⚠ 아직 안 맞는 조건 (장면 파일에서 직접 고칠 수 있어요): " + "; ".join(left))
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
            if not errs2 and len(retention_issues(spec2)) > len(retention_issues(spec)):
                errs2 = ["감독 수정안이 제작 조건(구성·이미지 목록·웃음 설계)을 깨뜨림"]
            if not errs2:
                ytxt, spec = revised, spec2
                log(f"  · 화면이 안 맞는 장면을 고쳤어요 ({time.time() - t1:.0f}초)")
            else:
                log("  · 수정안에 오류가 있어 원래 연출을 유지: " + "; ".join(errs2[:2]))
        else:
            log(f"  · 통과 ({time.time() - t1:.0f}초)")
    if punchup:
        log("▶ 개그 작가 다듬기: 대사의 펀치라인을 더 세게 (짧은 지시문이라 사용량 적음)")
        try:
            before = len(retention_issues(spec))
            spec2, n = apply_punchup(spec, call_light(PUNCHUP_SYSTEM, punchup_prompt(spec)))
            if n and len(retention_issues(spec2)) <= before:
                ytxt, spec = dump_yaml(spec2), spec2
                log(f"  · 대사 {n}개를 다듬었어요")
            else:
                log("  · 바꿀 만한 대사가 없어 그대로 둠")
        except Exception as e:
            log(f"  · 다듬기 중 문제가 있어 그대로 진행: {e}")
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


def write_image_prompts(spec: dict, out: Path) -> Path | None:
    """keyframes를 AI 이미지 생성기(ChatGPT·미드저니 등)에 붙여넣기 좋은 텍스트로 저장"""
    kfs = [k for k in spec.get("keyframes") or [] if isinstance(k, dict) and k.get("prompt")]
    if not kfs:
        return None
    lines = [f"# {spec.get('title', '')} — AI 이미지 {len(kfs)}장 (세로 9:16)", ""]
    for k in kfs:
        scenes = [str(i) for i, s in enumerate(spec.get("scenes") or [], 1)
                  if isinstance(s, dict) and str(s.get("image")) == str(k.get("id"))]
        lines += [f"[{k.get('id')}] {k.get('desc', '')}  (장면 {', '.join(scenes) or '-'})", str(k["prompt"]).strip(), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ------------------------------------------------------------------ 주제 목록 (topics.yaml)
def load_topics() -> list[dict]:
    path = ROOT / "topics.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [t for t in data.get("topics") or [] if isinstance(t, dict) and t.get("title")]


def pick_topics(text: str) -> list[dict] | None:
    """'3', '1,4', '1-3', '전체'면 목록에서 고른 주제들, 아니면 None(직접 입력한 주제)"""
    s = text.strip().replace(" ", "")
    if s in ("전체", "all"):
        return load_topics()
    if not re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", s):
        return None
    by_id = {int(t["id"]): t for t in load_topics() if str(t.get("id", "")).isdigit()}
    picked = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        for n in range(int(a), int(b or a) + 1):
            if n not in by_id:
                raise ValueError(f"주제 목록에 {n}번이 없어요 (1~{max(by_id) if by_id else 0}번)")
            picked.append(by_id[n])
    return picked
