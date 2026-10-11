"""AI 생성 API 연결부.

- MockImageProvider: API 키 없이 동작하는 자리표시자 (절차적 우주 배경을 그림)
- OpenAIImageProvider: OpenAI 이미지 API (OPENAI_API_KEY 필요, 모델은 IF60_IMAGE_MODEL로 변경)

AI 영상(Kling, Veo 등)은 아직 직접 호출하지 않는다. 각 서비스에서 만든 클립을
scenes YAML의 `source: video`로 넣으면 TemporalPixelator가 도트 후처리를 한다.
새 제공자를 붙이려면 ImageProvider를 상속해 generate()만 구현하면 된다.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from . import procedural

CACHE = Path(os.environ.get("IF60_CACHE", ".cache/ai"))


class ImageProvider:
    name = "base"

    def generate(self, prompt: str, size: tuple[int, int]) -> Image.Image:
        raise NotImplementedError

    def cached(self, prompt: str, size: tuple[int, int]) -> Image.Image:
        """같은 프롬프트는 다시 돈 내고 만들지 않도록 캐시."""
        key = hashlib.sha1(f"{self.name}|{prompt}|{size}".encode()).hexdigest()[:16]
        path = CACHE / f"{key}.png"
        if path.exists():
            return Image.open(path).convert("RGBA")
        img = self.generate(prompt, size)
        CACHE.mkdir(parents=True, exist_ok=True)
        img.save(path)
        (CACHE / f"{key}.txt").write_text(prompt, encoding="utf-8")
        return img


class MockImageProvider(ImageProvider):
    """키 없이 파이프라인을 시험하기 위한 가짜 생성기. 프롬프트 해시로 시드를 정한다."""
    name = "mock"

    def generate(self, prompt: str, size: tuple[int, int]) -> Image.Image:
        w, h = size
        seed = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
        canvas = np.zeros((h, w, 4), np.uint8)
        for layer in (
            procedural.sky(w, h, ["#0b0820", "#1b1446", "#3a2a78", "#7a4fa8"]),
            procedural.starfield(w, h, seed=seed, twinkle=False),
            procedural.mountains(w, h, color="#2a1f55", top=0.7, seed=seed + 1),
            procedural.mountains(w, h, color="#140f30", top=0.82, seed=seed + 2),
        ):
            procedural.paste(canvas, layer.frame(0), 0, 0)
        return Image.fromarray(canvas)


class OpenAIImageProvider(ImageProvider):
    """OpenAI Images API. 실제 키로는 아직 검증 전이니 첫 호출 결과를 꼭 확인하세요."""
    name = "openai"

    def __init__(self):
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 없어요.")
        self.model = os.environ.get("IF60_IMAGE_MODEL", "gpt-image-1")

    def generate(self, prompt: str, size: tuple[int, int]) -> Image.Image:
        # 세로 영상용 해상도 요청 → 이후 도트 후처리에서 축소하므로 품질은 'medium'이면 충분
        body = {"model": self.model, "prompt": prompt, "size": "1024x1536", "n": 1}
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)["data"][0]
        if "b64_json" in data:
            raw = base64.b64decode(data["b64_json"])
        else:
            raw = urllib.request.urlopen(data["url"], timeout=120).read()
        return Image.open(io.BytesIO(raw)).convert("RGBA")


PROVIDERS = {"mock": MockImageProvider, "openai": OpenAIImageProvider}


def get_image_provider(name: str | None = None) -> ImageProvider:
    name = name or os.environ.get("IF60_IMAGE_PROVIDER", "mock")
    if name not in PROVIDERS:
        raise ValueError(f"알 수 없는 이미지 제공자: {name} (가능: {', '.join(PROVIDERS)})")
    return PROVIDERS[name]()
