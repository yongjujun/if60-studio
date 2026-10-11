"""IF 60 Studio CLI

  python -m if60 render examples/earth_rotation_stop.yaml -o out/earth.mp4
  python -m if60 render examples/earth_rotation_stop.yaml --style retro_dot --length 30 --preview
  python -m if60 pixelate my_ai_clip.mp4 -o out/clip_dot.mp4 --style retro_dot
  python -m if60 make "파리는 어떻게 그렇게 잘 도망칠까?"     # 주제 → 영상 한 번에
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

    mk = sub.add_parser("make", help="주제만 입력하면 조사 → 대본 → 렌더링까지 한 번에")
    mk.add_argument("topic", nargs="?", help="주제 (비우면 입력창이 뜸)")
    mk.add_argument("--length", type=int, default=60, help="목표 길이(초), 기본 60")
    mk.add_argument("--preview", action="store_true", help="저화질 미리보기로만 빠르게")
    mk.add_argument("--no-open", action="store_true", help="완성 후 영상을 자동으로 열지 않기")
    mk.add_argument("--no-search", action="store_true", help="웹 검색 없이 작성 (비추천)")

    sub.add_parser("key", help="Anthropic API 키 저장/변경")

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
        from .planner import plan, write_scene_file
        ytxt, spec, cites = plan(a.topic, a.length)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(ytxt, encoding="utf-8")
        print(f"✔ 초안 저장: {a.out}  — facts_to_check 항목을 꼭 확인한 뒤 렌더링하세요.")
    elif a.cmd == "key":
        import getpass
        from .planner import save_key
        k = getpass.getpass("새 Anthropic API 키 (화면에 표시되지 않음): ").strip()
        if not k.startswith("sk-"):
            print("API 키 형식이 아니에요 (sk-ant-…).")
            return 1
        save_key(k)
        print("✔ 저장했어요 (.env — GitHub에는 올라가지 않아요)")
    elif a.cmd == "make":
        return make(a)
    elif a.cmd == "styles":
        from .render import load_styles
        for k, v in load_styles().items():
            w, h = v["resolution"]
            print(f"{k:18s} {v.get('label', '')}  ({w}x{h} ×{v['scale']}, {v['fps']}fps)")
    return 0


def make(a) -> int:
    from .planner import plan, write_scene_file
    from .render import render
    topic = (a.topic or "").strip()
    if not topic:
        topic = input("\n🎬 영상 주제를 입력하세요: ").strip()
    if not topic:
        print("주제가 비어 있어요.")
        return 1
    print(f"\n주제: {topic}  (목표 약 {a.length}초)\n")
    try:
        ytxt, spec, cites = plan(topic, a.length, web_search=not a.no_search)
    except RuntimeError as e:
        print(f"\n✖ {e}")
        return 1
    scene_path = write_scene_file(topic, ytxt, spec, cites)
    print(f"  · 장면 파일: {scene_path.relative_to(Path.cwd()) if scene_path.is_relative_to(Path.cwd()) else scene_path}")
    print(f"  · 제목: {spec.get('title', topic)}  ({len(spec['scenes'])}장면)")
    print("\n▶ 2/3 영상 렌더링 중")
    out = Path("out") / f"{scene_path.stem}{'_preview' if a.preview else ''}.mp4"
    render(str(scene_path), str(out), preview=a.preview)
    print("\n▶ 3/3 확인할 사실 (올리기 전에 꼭 확인!)")
    for f in spec.get("facts_to_check") or []:
        print(f"  - {f}")
    print(f"\n🎉 완성: {out.resolve()}")
    print("   대본을 고치고 싶으면 장면 파일을 수정한 뒤:")
    print(f"   python -m if60 render {scene_path.relative_to(Path.cwd()) if scene_path.is_relative_to(Path.cwd()) else scene_path}")
    if not a.no_open:
        open_file(out)
    return 0


def open_file(path: Path) -> None:
    """완성된 영상을 기본 플레이어로 연다 (Mac: QuickTime, Windows: 영화 및 TV)."""
    import platform
    import subprocess
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        elif platform.system() == "Windows":
            import os
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
