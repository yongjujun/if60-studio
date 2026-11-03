"""ffmpeg 입출력 — 영상 프레임 읽기와 raw 프레임을 mp4로 인코딩."""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

import numpy as np


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg가 필요해요. macOS: brew install ffmpeg / Windows: winget install ffmpeg")


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
        check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den), "duration": float(d["format"]["duration"])}


def read_frames(path: str, fps: float, max_w: int = 540) -> Iterator[np.ndarray]:
    """영상을 fps로 다시 샘플링해 RGB 프레임을 하나씩 돌려준다. (메모리 절약을 위해 max_w로 미리 축소)"""
    info = probe(path)
    w = min(max_w, info["width"])
    h = int(round(info["height"] * w / info["width"] / 2) * 2)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={w}:{h}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    assert proc.stdout
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def upload_check(path: str) -> list[str]:
    """완성 영상이 인스타 릴스·쇼츠 업로드 규격(1080x1920 9:16, H.264, 화소 비율 1:1, 30fps, AAC)인지 확인 → 문제 목록"""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,sample_aspect_ratio,r_frame_rate,channels",
                        "-of", "json", path], capture_output=True, text=True)
    try:
        streams = json.loads(r.stdout).get("streams", [])
    except json.JSONDecodeError:
        return ["영상 정보를 읽지 못함"]
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        return ["영상 트랙 없음"]
    issues = []
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    if (w, h) != (1080, 1920):
        issues.append(f"해상도 {w}x{h} — 업로드용은 1080x1920 (미리보기 파일은 올리지 마세요)")
    if w * 16 != h * 9:
        issues.append("화면 비율이 9:16이 아님 — 인스타가 테두리를 흐리게 채움")
    if v.get("sample_aspect_ratio") not in (None, "1:1"):
        issues.append(f"화소 비율 {v.get('sample_aspect_ratio')} — 1:1이어야 함")
    if v.get("codec_name") != "h264":
        issues.append(f"코덱 {v.get('codec_name')} — H.264 권장")
    if v.get("r_frame_rate") not in ("30/1", "60/1"):
        issues.append(f"프레임 {v.get('r_frame_rate')} — 30fps 권장")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if a and a.get("codec_name") != "aac":
        issues.append(f"오디오 {a.get('codec_name')} — AAC 권장")
    return issues


class Encoder:
    """raw RGB 프레임을 받아 H.264 mp4로 저장. 도트 영상은 저fps(10~12)로 만들고
    업로드 호환성을 위해 30fps로 프레임을 복제해 내보낸다."""

    def __init__(self, path: str, width: int, height: int, fps: float,
                 audio: str | None = None, out_fps: int = 30, crf: int = 16):
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio:   # 인스타 권장: AAC 스테레오 48kHz
            cmd += ["-i", audio, "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "48000", "-shortest"]
        # 인스타 릴스·쇼츠 권장 규격: H.264 High, 9:16, 화소 비율 1:1(setsar=1 — 없으면 앱이 비율을 잘못 읽어
        # 테두리를 흐리게 채우는 경우가 있음), 30fps 고정, 비트레이트 상한 12Mbps(앱 재압축으로 뭉개지는 것 방지)
        cmd += ["-vf", f"fps={out_fps},setsar=1", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-maxrate", "12M", "-bufsize", "24M", "-profile:v", "high", "-level", "4.1",
                "-g", str(out_fps * 2), "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self.proc.stdin
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg 인코딩 실패")
