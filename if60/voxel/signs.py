"""3D 공간 속 글자 소품 — 간판·현수막·가격표·매물 카드·모니터 화면·종이 공지.

화면 위에 뜨는 자막이 아니라 세상 속 물건에 적힌 글자라서, 카메라가 움직이면 같이 원근이 생긴다.
상황과 수치를 세상 속 물건으로 보여준다: "모기 물림 진료 · 대기 40분", "1층 · 개울 앞 · 모기 많음".

장면 YAML:
  props:
    - {shape: sign, style: card, title: "1층", text: "개울 앞\\n**모기 많음**\\n★★★",
       size: [0.8, 1.0], at: [x, y, z](판 아랫변 가운데), yaw: 0, mount: pole|wall|none, color: "#d7262e"}
  text의 줄바꿈은 \\n, **글자** 는 강조색 줄. 판 앞면은 yaw 0일 때 -z(카메라 쪽)를 본다.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STYLES = {
    # 이름: 배경, 글자, 강조(제목 띠·**강조 줄**), 테두리, 정렬, 스스로 빛남
    "board": {"bg": "#f4f1e8", "fg": "#1e1e26", "accent": "#d7262e", "frame": "#3a3a44", "align": "center", "glow": False},
    "card": {"bg": "#ffffff", "fg": "#26262e", "accent": "#d7262e", "frame": "#d7262e", "align": "center", "glow": False},
    "banner": {"bg": "#d7262e", "fg": "#ffffff", "accent": "#ffe14a", "frame": "#f2f2f2", "align": "center", "glow": False},
    "shop": {"bg": "#24263c", "fg": "#ffd23f", "accent": "#ffffff", "frame": "#111118", "align": "center", "glow": False},
    "price": {"bg": "#1f2b24", "fg": "#f2f1e6", "accent": "#ffd23f", "frame": "#6b4a2e", "align": "left", "glow": False},
    "screen": {"bg": "#0d2a5c", "fg": "#ffffff", "accent": "#ffd23f", "frame": "#15151a", "align": "center", "glow": True},
    "neon": {"bg": "#141019", "fg": "#ff5fa8", "accent": "#5fe3ff", "frame": "#0c0c10", "align": "center", "glow": True},
    "paper": {"bg": "#fbfbf7", "fg": "#202020", "accent": "#d7262e", "frame": "#fbfbf7", "align": "center", "glow": False},
}


def _hex(c) -> tuple[int, int, int]:
    c = str(c).lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) if len(c) == 6 else (128, 128, 128)


@lru_cache(maxsize=64)
def _font(path: str, size: int):
    return ImageFont.truetype(path, size=max(6, int(size)), index=0)


def _style(pr: dict) -> dict:
    st = dict(STYLES.get(pr.get("style", "board"), STYLES["board"]))
    for k_yaml, k in (("bg", "bg"), ("fg", "fg"), ("color", "accent"), ("frame", "frame")):
        if pr.get(k_yaml):
            st[k] = pr[k_yaml]
    if pr.get("style") == "banner" and pr.get("color"):      # 현수막은 color가 바탕색
        st["bg"], st["accent"] = pr["color"], STYLES["banner"]["accent"]
    if "glow" in pr:
        st["glow"] = bool(pr["glow"])
    return st


def sign_size(pr: dict) -> tuple[float, float]:
    s = pr.get("size", [1.0, 0.6])
    w, h = (float(s[0]), float(s[1])) if isinstance(s, (list, tuple)) and len(s) >= 2 else (1.0, 0.6)
    k = float(pr.get("scale", 1.0))
    return max(0.05, w * k), max(0.05, h * k)


def sign_texture(pr: dict, font_path: str) -> np.ndarray:
    """간판 그림(RGBA). 판 크기 비율대로, 글자는 판을 꽉 채우게 자동 크기."""
    from ..text import safe_text
    st = _style(pr)
    w_m, h_m = sign_size(pr)
    H = int(min(384, max(48, round(h_m * 170))))
    W = int(min(1024, max(48, round(H * w_m / h_m))))
    bg, fg, accent = _hex(st["bg"]), _hex(st["fg"]), _hex(st["accent"])
    im = Image.new("RGBA", (W, H), (*bg, 255))
    d = ImageDraw.Draw(im)
    pad = max(2, int(min(W, H) * 0.07))
    top = 0
    title = safe_text(str(pr.get("title", "") or ""))
    if title:                                                   # 제목 띠 (매물 카드 "1층", 가게 이름)
        th = int(H * (0.3 if pr.get("text") else 1.0))
        d.rectangle([0, 0, W, th], fill=(*accent, 255))
        tc = (255, 255, 255) if sum(accent) < 600 else (20, 20, 24)
        f = _fit(font_path, [title], W - pad * 2, th - pad)
        tw = f.getlength(title)
        d.text(((W - tw) / 2, (th - f.size) / 2 - f.size * 0.12), title, font=f, fill=(*tc, 255),
               stroke_width=max(1, f.size // 24), stroke_fill=(*tc, 255))
        top = th
    raw = str(pr.get("text", "") or "")
    lines = [safe_text(l) for l in raw.replace("\\n", "\n").split("\n")] if raw else []
    if lines:
        plain = [re.sub(r"\*\*(.+?)\*\*", r"\1", l) for l in lines]
        avail_h = H - top - pad * 2
        f = _fit(font_path, plain, W - pad * 2, avail_h / (len(lines) * 1.22))
        lh = f.size * 1.22
        y = top + pad + (avail_h - lh * len(lines)) / 2
        for l, p in zip(lines, plain):
            col = accent if l.startswith("**") and l.endswith("**") else fg
            tw = f.getlength(p)
            x = pad if st["align"] == "left" else (W - tw) / 2
            d.text((x, y), p, font=f, fill=(*col, 255), stroke_width=max(1, f.size // 22), stroke_fill=(*col, 255))
            y += lh
    fr = _hex(st["frame"])
    bw = max(1, int(min(W, H) * 0.025))
    if st["frame"] != st["bg"]:
        d.rectangle([0, 0, W - 1, H - 1], outline=(*fr, 255), width=bw)
    return np.asarray(im)


def _fit(path: str, lines: list[str], max_w: float, max_h: float):
    s = int(max(6, max_h))
    while s > 6:
        f = _font(path, s)
        if max(f.getlength(l) for l in lines) <= max_w:
            return f
        s = int(s * 0.9)
    return _font(path, 6)


def _rot(yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    return lambda p: np.stack([c * p[:, 0] + s * p[:, 2], p[:, 1], -s * p[:, 0] + c * p[:, 2]], 1)


def sign_quad(pr: dict) -> tuple[np.ndarray, np.ndarray]:
    """판 앞면 사각형: (6, 5) [x y z u v] — 아랫변 가운데 기준 상대 좌표(회전 적용), 법선"""
    w, h = sign_size(pr)
    yaw = math.radians(float(pr.get("yaw", 0)))
    z = -0.006
    p = np.array([[-w / 2, 0, z], [w / 2, 0, z], [w / 2, h, z], [-w / 2, 0, z], [w / 2, h, z], [-w / 2, h, z]],
                 np.float32)
    uv = np.array([[1, 1], [0, 1], [0, 0], [1, 1], [0, 0], [1, 0]], np.float32)   # 앞(-z)에서 보면 좌우가 뒤집힘
    uv[:, 0] = 1 - uv[:, 0]
    rot = _rot(yaw)
    n = rot(np.array([[0, 0, -1]], np.float32))[0]
    return np.hstack([rot(p), uv]).astype(np.float32), n.astype(np.float32)


def sign_backing(pr: dict):
    """판 뒤 받침(두께) + 기둥/다리 — 복셀 모델로 돌려준다 (그림자도 생김)"""
    from .geometry import Model
    st = _style(pr)
    w, h = sign_size(pr)
    m = Model("sign", [st["frame"]], [st["frame"]])
    t = float(pr.get("depth", 0.04))
    m.part((-w / 2 - 0.015, -0.015, 0.0), (w / 2 + 0.015, h + 0.015, t), st["frame"])
    mount = pr.get("mount", "none")
    at_y = float((pr.get("at") or [0, 0, 0])[1])
    if mount == "pole" and at_y > 0.05:                         # 땅에서 판까지 기둥 하나 (입간판·도로 표지)
        r = max(0.02, min(w, 1.0) * 0.04)
        m.part((-r, -at_y, t * 0.5 - r), (r, 0, t * 0.5 + r), "#8a8a92")
    elif mount == "stand" and at_y > 0.05:                      # 다리 두 개 (세움 간판·가격표 받침)
        for sx in (-w * 0.38, w * 0.38):
            m.part((sx - 0.02, -at_y, t * 0.3), (sx + 0.02, 0, t * 0.7), "#6b6b72")
    return m


SIGN_VS = """
#version 330
uniform mat4 u_vp;
uniform vec3 u_origin;
uniform float u_g;
in vec3 in_pos;
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_world;
void main() {
    vec3 w = u_origin + in_pos * u_g;
    v_uv = in_uv;
    v_world = w;
    gl_Position = u_vp * vec4(w, 1.0);
}
"""

SIGN_FS = """
#version 330
in vec2 v_uv;
in vec3 v_world;
uniform sampler2D u_tex;
uniform vec3 u_n;
uniform vec3 u_cam;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_amb_sky;
uniform vec3 u_amb_ground;
uniform vec3 u_fog_col;
uniform float u_fog_density;
uniform float u_glow;
out vec4 f_color;
void main() {
    vec4 t = texture(u_tex, v_uv);
    vec3 amb = mix(u_amb_ground, u_amb_sky, 0.6);
    vec3 col = t.rgb * (amb + u_sun_col * max(dot(u_n, u_sun_dir), 0.0) * 0.8);
    col = mix(col, t.rgb * 1.05, u_glow);
    float d = length(v_world - u_cam);
    float fog = 1.0 - exp(-pow(d * u_fog_density, 1.4));
    col = mix(col, u_fog_col, clamp(fog, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
"""
