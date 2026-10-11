"""주제 → 장면 YAML 초안 (Claude API).

자동화 정책 리스크를 줄이기 위해 '초안'만 만든다. 사람이 대본·사실관계를 고친 뒤 render 하세요.
ANTHROPIC_API_KEY가 필요하고, 모델은 IF60_PLANNER_MODEL로 바꿀 수 있다.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .procedural import KINDS

SYSTEM = f"""너는 과학 상상 쇼츠 채널 'IF 60'의 구성 작가다.
주어진 '만약 ~라면?' 주제로 9:16 도트 애니메이션 쇼츠의 장면 구성을 YAML로만 출력한다.

규칙
- 총 길이는 요청 초수에 맞춘다. 장면은 5~9개, 각 4~10초.
- 첫 장면은 title(짧은 질문형)과 강한 훅 caption. 마지막 장면은 여운/반전 한 줄.
- caption은 한 장면당 40자 이내, 사실은 정확하게. 확실하지 않은 수치는 쓰지 않는다.
- 레이어는 source: procedural (kind: {', '.join(KINDS)}) 또는 source: ai (prompt: 영어 묘사)만 사용.
- 재난/충격 장면에는 effects: [{{shake: {{at: 0.5, length: 1.2, amp: 4}}}}] 또는 flash 를 쓴다.
- camera.pan 은 [x, y] 저해상도 픽셀 이동량(-40~40).
- music: {{mood: wonder|tense|calm|epic, bpm: 90~140, seed: 정수}}

출력 형식 (설명 없이 YAML만):
title: ...
style: cinematic_pixel
music: {{...}}
facts_to_check: [대본에 쓴 수치와 그 근거를 사람이 확인할 수 있게 나열]
scenes:
  - duration: 4
    title: ...
    caption: ...
    layers: [...]
"""


def plan(topic: str, length: int = 60) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY 환경변수가 없어요. 예제 YAML을 복사해 직접 써도 됩니다.")
    body = {
        "model": os.environ.get("IF60_PLANNER_MODEL", "claude-sonnet-5-5"),
        "max_tokens": 4000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": f"주제: {topic}\n길이: {length}초"}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    out = "".join(b.get("text", "") for b in data["content"])
    m = re.search(r"```(?:yaml)?\n(.*?)```", out, re.S)
    return (m.group(1) if m else out).strip() + "\n"
