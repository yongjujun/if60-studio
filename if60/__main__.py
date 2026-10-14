"""IF 60 Studio CLI

  python -m if60 render examples/earth_rotation_stop.yaml -o out/earth.mp4
  python -m if60 render examples/earth_rotation_stop.yaml --style retro_dot --length 30 --preview
  python -m if60 pixelate my_ai_clip.mp4 -o out/clip_dot.mp4 --style retro_dot
  python -m if60 make "파리는 어떻게 그렇게 잘 도망칠까?"     # 주제 → 영상 한 번에
  python -m if60 topics                       # 저장된 주제 목록 (topics.yaml)
  python -m if60 make 3                       # 목록 3번 주제로 제작 (1,4 / 1-3 / 전체 도 가능)
  python -m if60 plan "달이 사라진다면?" -o scenes/moon.yaml
  python -m if60 styles
  python -m if60 voice list supertonic        # 목소리 목록 (supertonic | typecast | google)
  python -m if60 voice test typecast:tc_xxx   # 목소리 들어보기
  python -m if60 voice use typecast:tc_xxx    # 기본 목소리로 정하기
  python -m if60 key typecast                 # API 키 저장 (anthropic | typecast | google)
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
    r.add_argument("--voice", help="목소리 (예: supertonic:F3, typecast:tc_xxx, google:Charon)")
    r.add_argument("--no-voice", action="store_true", help="내레이션 없이 무음으로")

    x = sub.add_parser("pixelate", help="AI 이미지/영상 하나를 도트로 후처리")
    x.add_argument("src")
    x.add_argument("-o", "--out", required=True)
    x.add_argument("--style", default="cinematic_pixel")
    x.add_argument("--preview", action="store_true")

    mk = sub.add_parser("make", help="주제만 입력하면 조사 → 대본 → 렌더링까지 한 번에")
    mk.add_argument("topic", nargs="?", help="주제 또는 목록 번호(3, 1-3, 1,4, 전체) — 비우면 목록과 입력창이 뜸")
    mk.add_argument("--length", type=int, default=90, help="목표 길이(초), 기본 90")
    mk.add_argument("--preview", action="store_true", help="저화질 미리보기로만 빠르게")
    mk.add_argument("--no-open", action="store_true", help="완성 후 영상을 자동으로 열지 않기")
    mk.add_argument("--no-search", action="store_true", help="웹 검색 없이 작성 (비추천)")
    mk.add_argument("--review", action="store_true",
                    help="감독 검수 켜기 (미리보기를 보며 화면·자막 일치 확인, 사용량 더 씀 — 기본은 꺼짐)")
    mk.add_argument("--no-punchup", action="store_true", help="개그 작가 다듬기(대사 펀치라인 강화) 건너뛰기")
    mk.add_argument("--voice", help="이번 영상의 목소리 (기본: voice use로 정한 것)")
    mk.add_argument("--no-voice", action="store_true", help="내레이션 없이 무음으로")

    k = sub.add_parser("key", help="API 키 저장/변경 (anthropic | typecast | google)")
    k.add_argument("service", nargs="?", default="anthropic", choices=["anthropic", "typecast", "google"])

    vo = sub.add_parser("voice", help="내레이션 목소리: list(목록) / test(들어보기) / use(기본으로 정하기)")
    vo.add_argument("action", choices=["list", "test", "use"])
    vo.add_argument("target", nargs="?", help="list: supertonic|typecast|google / test·use: 제공자:목소리")
    vo.add_argument("text", nargs="?", default="안녕하세요, 만물제작소입니다. 오늘은 번개 한 방이 얼마나 센지 알아볼게요.")

    pl = sub.add_parser("plan", help="주제로 장면 YAML 초안 만들기 (Claude API)")
    pl.add_argument("topic")
    pl.add_argument("-o", "--out", required=True)
    pl.add_argument("--length", type=int, default=90)

    sub.add_parser("styles", help="스타일 프리셋 목록")
    sub.add_parser("topics", help="저장된 주제 목록 (topics.yaml) — make 번호로 바로 제작")

    a = p.parse_args(argv)

    if a.cmd == "render":
        from .render import render
        out = a.out or f"out/{Path(a.scenes).stem}{'_preview' if a.preview else ''}.mp4"
        render(a.scenes, out, style_override=a.style, length=a.length, provider=a.provider,
               preview=a.preview, stills_dir=a.stills, fps=a.fps, voice=a.voice, narrate=not a.no_voice)
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
        var, label = {"anthropic": ("ANTHROPIC_API_KEY", "Anthropic"), "typecast": ("TYPECAST_API_KEY", "타입캐스트"),
                      "google": ("GOOGLE_TTS_API_KEY", "구글 Cloud")}[a.service]
        k = getpass.getpass(f"새 {label} API 키를 붙여넣고 Enter (화면에 표시되지 않음): ").strip()
        if len(k) < 16 or (a.service == "anthropic" and not k.startswith("sk-")):
            print("API 키 형식이 아닌 것 같아요. 다시 복사해서 붙여넣어 주세요.")
            return 1
        save_key(k, var)
        print("✔ 저장했어요 (.env — GitHub에는 올라가지 않아요)")
    elif a.cmd == "voice":
        return voice_cmd(a)
    elif a.cmd == "make":
        return make(a)
    elif a.cmd == "topics":
        print_topics()
    elif a.cmd == "styles":
        from .render import load_styles
        for k, v in load_styles().items():
            w, h = v["resolution"]
            print(f"{k:18s} {v.get('label', '')}  ({w}x{h} ×{v['scale']}, {v['fps']}fps)")
    return 0


GENRE_LABEL = {"satire": "풍자", "science": "과학", "absurd": "가정"}


def print_topics() -> None:
    from .planner import load_topics
    items = load_topics()
    if not items:
        print("topics.yaml에 주제가 없어요.")
        return
    print("\n📋 주제 목록 (topics.yaml)")
    for t in items:
        print(f"  {str(t.get('id')):>2}. [{GENRE_LABEL.get(t.get('genre'), '?')}] {t['title']}")


def make(a) -> int:
    from .planner import pick_topics
    raw = (a.topic or "").strip()
    if not raw:
        print_topics()
        raw = input("\n🎬 번호(예: 3, 1-3, 전체) 또는 새 주제를 입력하세요: ").strip()
    if not raw:
        print("주제가 비어 있어요.")
        return 1
    try:
        picked = pick_topics(raw)
    except ValueError as e:
        print(f"✖ {e}")
        return 1
    jobs = picked if picked is not None else [{"title": raw}]
    if len(jobs) > 1:
        print(f"\n주제 {len(jobs)}개를 차례로 만들어요. 하나에 보통 5~15분 걸려요.")
    failed = []
    for n, job in enumerate(jobs, 1):
        if len(jobs) > 1:
            print(f"\n━━━━━━━━ {n}/{len(jobs)} ━━━━━━━━")
        try:
            rc = make_one(a, job)
        except Exception as e:
            print(f"\n✖ {type(e).__name__}: {e}")
            rc = 1
            if len(jobs) == 1:
                raise
        if rc:
            failed.append(job["title"])
    if len(jobs) > 1:
        print(f"\n완료 {len(jobs) - len(failed)}개 / 실패 {len(failed)}개" + "".join(f"\n  - {f}" for f in failed))
    return 1 if failed else 0


def make_one(a, job: dict) -> int:
    from .planner import plan, write_image_prompts, write_scene_file
    from .render import render
    topic = job["title"]
    genre = job.get("genre")
    print(f"\n주제: {topic}  (목표 약 {a.length}초" + (f", {GENRE_LABEL.get(genre, genre)}" if genre else "") + ")\n")
    try:
        ytxt, spec, cites = plan(topic, a.length, web_search=not a.no_search, review_pass=a.review,
                                 genre=genre, angle=job.get("angle"), punchup=not a.no_punchup,
                                 tangents=job.get("tangents"))
    except RuntimeError as e:
        print(f"\n✖ {e}")
        return 1
    scene_path = write_scene_file(topic, ytxt, spec, cites)
    img_path = write_image_prompts(spec, Path("out") / f"{scene_path.stem}_AI이미지.txt")
    if img_path:
        print(f"  · AI 이미지 {len(spec.get('keyframes') or [])}장 프롬프트: {img_path}")
    print(f"  · 장면 파일: {scene_path.relative_to(Path.cwd()) if scene_path.is_relative_to(Path.cwd()) else scene_path}")
    print(f"  · 제목: {spec.get('title', topic)}  ({len(spec['scenes'])}장면)")
    print("\n▶ 2/3 영상 렌더링 중")
    out = Path("out") / f"{scene_path.stem}{'_preview' if a.preview else ''}.mp4"
    try:
        render(str(scene_path), str(out), preview=a.preview, voice=a.voice, narrate=not a.no_voice)
    except Exception as e:
        rel = scene_path.relative_to(Path.cwd()) if scene_path.is_relative_to(Path.cwd()) else scene_path
        print(f"\n✖ 렌더링 중 오류: {type(e).__name__}: {e}")
        print(f"   대본(장면 파일)은 저장돼 있어요: {rel}")
        print(f"   문제를 고친 뒤 다시 조사할 필요 없이:  python -m if60 render {rel}")
        raise
    print("\n▶ 3/3 확인할 사실 (올리기 전에 꼭 확인!)")
    for f in spec.get("facts_to_check") or []:
        print(f"  - {f}")
    print(f"\n🎉 완성: {out.resolve()}")
    print("   대본을 고치고 싶으면 장면 파일을 수정한 뒤:")
    print(f"   python -m if60 render {scene_path.relative_to(Path.cwd()) if scene_path.is_relative_to(Path.cwd()) else scene_path}")
    if not a.no_open:
        open_file(out)
    return 0


def voice_cmd(a) -> int:
    from . import tts
    from .planner import save_key
    try:
        if a.action == "list":
            prov = (a.target or "supertonic").split(":")[0]
            voices = tts.list_voices(prov)
            if prov == "typecast":
                voices = [v for v in voices if v["name"]]
            print(f"\n{prov} 목소리 {len(voices)}개  (들어보기: python -m if60 voice test {prov}:<ID>)\n")
            for v in voices:
                print(f"  {prov}:{v['id']:<28} {v['name']:<12} {v.get('info', '')}")
            if prov == "typecast":
                print("\n  목소리 미리듣기는 타입캐스트 웹사이트(typecast.ai)에서 캐릭터 이름으로 찾아보면 편해요.")
            return 0
        v = tts.parse_voice(a.target or tts.default_voice())
        if a.action == "test":
            x = tts.synthesize(a.text, v)
            out = Path("out") / f"voice_test_{v.provider}_{v.name}.wav"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(tts.encode_wav(x, tts.SR))
            print(f"✔ {v.label} ({len(x) / tts.SR:.1f}초): {out}")
            open_file(out)
            return 0
        save_key(v.label, "IF60_VOICE")
        print(f"✔ 기본 목소리를 {v.label}(으)로 정했어요. 이제 make·render가 이 목소리로 읽어요.")
        if v.provider == "typecast":
            print("  타입캐스트 무료 플랜: 월 15,000자 · 상업적 이용 불가 · 출처 표기 필수 (렌더링 후 문구를 알려드려요)")
        return 0
    except (RuntimeError, ValueError) as e:
        print(f"✖ {e}")
        return 1


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
