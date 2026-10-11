"""IF 60 Studio CLI

  python -m if60 render examples/earth_rotation_stop.yaml -o out/earth.mp4
  python -m if60 render examples/earth_rotation_stop.yaml --style retro_dot --length 30 --preview
  python -m if60 pixelate my_ai_clip.mp4 -o out/clip_dot.mp4 --style retro_dot
  python -m if60 plan "달이 사라진다면?" -o scenes/moon.yaml
  python -m if60 styles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="if60", description="IF 60 Studio — 도트 애니메이션 쇼츠 제작 도구")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="장면 YAML을 9:16 쇼츠 mp4로 렌더링")
    r.add_argument("scenes")
    r.add_argument("-o", "--out", default=None)
    r.add_argument("--style", help="스타일 프리셋 (YAML의 style 대신)")
    r.add_argument("--length", type=float, help="총 길이(초)로 장면 길이를 비례 조정 (예: 30, 60, 120)")
    r.add_argument("--provider", help="이미지 생성 제공자: mock | openai")
    r.add_argument("--preview", action="store_true", help="1/3 해상도로 빠르게 확인")
    r.add_argument("--stills", help="장면별 대표 프레임 PNG를 저장할 폴더")
    r.add_argument("--fps", type=int, help="초당 프레임 수 (기본: 스타일 설정, 3D 미리보기는 15)")

    x = sub.add_parser("pixelate", help="AI 이미지/영상 하나를 도트로 후처리")
    x.add_argument("src")
    x.add_argument("-o", "--out", required=True)
    x.add_argument("--style", default="cinematic_pixel")
    x.add_argument("--preview", action="store_true")

    pl = sub.add_parser("plan", help="주제로 장면 YAML 초안 만들기 (Claude API)")
    pl.add_argument("topic")
    pl.add_argument("-o", "--out", required=True)
    pl.add_argument("--length", type=int, default=60)

    sub.add_parser("styles", help="스타일 프리셋 목록")

    a = p.parse_args(argv)

    if a.cmd == "render":
        from .render import render
        out = a.out or f"out/{Path(a.scenes).stem}{'_preview' if a.preview else ''}.mp4"
        render(a.scenes, out, style_override=a.style, length=a.length, provider=a.provider,
               preview=a.preview, stills_dir=a.stills, fps=a.fps)
    elif a.cmd == "pixelate":
        from .render import pixelate_file
        pixelate_file(a.src, a.out, a.style, a.preview)
    elif a.cmd == "plan":
        from .planner import plan
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(plan(a.topic, a.length), encoding="utf-8")
        print(f"✔ 초안 저장: {a.out}  — facts_to_check 항목을 꼭 확인한 뒤 렌더링하세요.")
    elif a.cmd == "styles":
        from .render import load_styles
        for k, v in load_styles().items():
            w, h = v["resolution"]
            print(f"{k:18s} {v.get('label', '')}  ({w}x{h} ×{v['scale']}, {v['fps']}fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
