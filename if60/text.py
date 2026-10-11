"""도트 한글 자막.

글자를 저해상도 캔버스(기본 360x640, 최종 3배 확대)에 그린 뒤 알파를 0/255로 잘라
안티앨리어싱 없는 또렷한 픽셀 글자를 만든다. 비트맵 폰트(갈무리체 등)를 fonts/에 넣으면
그 폰트를 우선 사용한다.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

FONT_CANDIDATES = [
    # 1순위: 사용자가 넣은 비트맵 폰트 (예: 갈무리체 Galmuri11.ttf — SIL OFL)
    *sorted(str(p) for p in (ROOT / "fonts").glob("*.[ot]t[fc]")),
    # 2순위: OS 기본 한글 폰트
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
]


def find_font(explicit: str | None = None) -> str:
    for p in ([explicit] if explicit else []) + FONT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "한글 폰트를 찾지 못했어요. fonts/ 폴더에 갈무리체(Galmuri11.ttf 등)를 넣거나 "
        "스타일 설정의 font 경로를 지정해 주세요.")


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    # .ttc는 index 0이 보통 한국어/공통 글리프를 포함
    return ImageFont.truetype(path, size=size, index=0)


def wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            trial = f"{cur} {word}".strip()
            if font.getlength(trial) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def safe_text(s: str) -> str:
    """대부분의 한글 폰트에 없는 아래·위첨자 숫자(N₂, m²)를 일반 숫자로 바꿔 □가 안 뜨게 한다."""
    return str(s).translate(_SUBSCRIPTS)


def chunk_caption(text: str, max_chars: int = 11) -> list[str]:
    """내레이션 자막을 2~5단어 덩어리로 자른다 (쇼츠에서 가장 읽기 쉬운 단위).
    문장부호(. ! ? ,) 뒤에서는 항상 끊는다."""
    chunks, cur = [], ""
    for word in safe_text(text).replace("\n", " ").split():
        trial = f"{cur} {word}".strip()
        if cur and len(trial) > max_chars:
            chunks.append(cur)
            cur = word
        else:
            cur = trial
        if cur and cur[-1] in ".!?,":
            chunks.append(cur.rstrip(","))
            cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def render_badge(canvas: tuple[int, int], label: str, *, font_path: str, size: int = 16,
                 x: float = 0.94, y: float = 0.075, fg=(255, 255, 255),
                 bg=(214, 40, 72)) -> np.ndarray:
    """화면 구석의 HUD 배지 (시계, 카운터, '+1년' 등). x,y는 배지 오른쪽 위 기준 비율."""
    w, h = canvas
    img = Image.new("RGBA", canvas, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(font_path, size)
    label = safe_text(label)
    tw = int(font.getlength(label))
    pad = max(3, size // 4)
    x1, y0 = int(w * x), int(h * y)
    x0 = x1 - tw - pad * 2
    d.rectangle((x0, y0, x1, y0 + size + pad), fill=(*bg, 255))
    d.text((x0 + pad, y0 + pad // 2 - 1), label, font=font, fill=(*fg, 255))
    arr = np.asarray(img).copy()
    a = arr[..., 3]
    arr[..., 3] = np.where(a > 110, 255, 0).astype(np.uint8)
    on = arr[..., 3] == 255
    rgb = arr[..., :3].astype(np.int32)
    near_fg = ((rgb - np.array(fg)) ** 2).sum(-1) < ((rgb - np.array(bg)) ** 2).sum(-1)
    arr[on & near_fg, :3] = fg
    arr[on & ~near_fg, :3] = bg
    return arr


def render_text(canvas: tuple[int, int], text: str, *, font_path: str, size: int = 22,
                y: float = 0.72, color=(255, 255, 255), outline=(10, 8, 24),
                box: tuple[int, int, int] | None = None, align: str = "center",
                visible_chars: int | None = None) -> np.ndarray:
    """canvas 크기의 RGBA(0/255 알파) 자막 레이어를 돌려준다.

    visible_chars: 타자기 효과용. 앞에서부터 이 개수만 보인다.
    """
    w, h = canvas
    img = Image.new("RGBA", canvas, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(font_path, size)
    text = safe_text(text)
    lines = wrap(text, font, int(w * 0.86))
    lh = int(size * 1.35)
    total_h = lh * len(lines)
    top = int(h * y - total_h / 2)

    if box is not None:
        bw = int(max(font.getlength(l) for l in lines)) + 16
        bx0 = (w - bw) // 2
        d.rectangle((bx0, top - 6, bx0 + bw, top + total_h + 4), fill=(*box, 255),
                    outline=(*color, 255), width=2)

    shown = 0
    for i, line in enumerate(lines):
        if visible_chars is not None:
            line = line[: max(0, visible_chars - shown)]
            shown += len(lines[i])
        if not line:
            continue
        tw = font.getlength(lines[i])
        x = (w - tw) / 2 if align == "center" else w * 0.07
        d.text((x, top + i * lh), line, font=font, fill=(*color, 255),
               stroke_width=0 if box else 2, stroke_fill=(*outline, 255))

    arr = np.asarray(img).copy()
    a = arr[..., 3]
    arr[..., 3] = np.where(a > 110, 255, 0).astype(np.uint8)
    # 안티앨리어싱 중간색 제거: 글자색/외곽선색 중 가까운 쪽으로 스냅
    on = arr[..., 3] == 255
    rgb = arr[..., :3].astype(np.int32)
    c, o = np.array(color), np.array(outline)
    near_c = ((rgb - c) ** 2).sum(-1) < ((rgb - o) ** 2).sum(-1)
    arr[on & near_c, :3] = c
    arr[on & ~near_c, :3] = o if box is None else (*box,)
    return arr
