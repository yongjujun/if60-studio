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


class Encoder:
    """raw RGB 프레임을 받아 H.264 mp4로 저장. 도트 영상은 저fps(10~12)로 만들고
    업로드 호환성을 위해 30fps로 프레임을 복제해 내보낸다."""

    def __init__(self, path: str, width: int, height: int, fps: float,
                 audio: str | None = None, out_fps: int = 30, crf: int = 16):
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio:
            cmd += ["-i", audio, "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-vf", f"fps={out_fps}", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self.proc.stdin
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg 인코딩 실패")
