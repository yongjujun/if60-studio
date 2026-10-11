"""칩튠 BGM/효과음 생성기 — 저작권 걱정 없는 8비트 사운드를 코드로 만든다.

사각파(멜로디) + 삼각파(베이스) + 노이즈(하이햇/킥)로 구성.
mood에 따라 스케일과 템포 느낌이 달라지고, seed가 같으면 항상 같은 곡이 나온다.
"""
from __future__ import annotations

import wave

import numpy as np

SR = 44100

MOODS = {
    #  스케일(반음),             루트 MIDI, 진행(스케일 도수)
    "wonder": ([0, 2, 4, 7, 9], 57, [0, 3, 4, 2]),       # 장조 펜타토닉 — 신비/호기심
    "tense": ([0, 2, 3, 7, 8], 52, [0, 0, 4, 3]),        # 단조 계열 — 긴장/재난
    "calm": ([0, 2, 4, 7, 9], 60, [0, 4, 3, 4]),
    "epic": ([0, 3, 5, 7, 10], 50, [0, 4, 3, 2]),
}


def _freq(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _square(f: float, n: int, duty: float = 0.5) -> np.ndarray:
    t = np.arange(n) / SR
    return np.where((t * f) % 1.0 < duty, 1.0, -1.0)


def _triangle(f: float, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    return 2 * np.abs(2 * ((t * f) % 1.0) - 1) - 1


def _env(n: int, attack=0.005, release=0.06) -> np.ndarray:
    e = np.ones(n)
    a, r = int(SR * attack), int(SR * release)
    if a:
        e[:a] = np.linspace(0, 1, a)
    if r and r < n:
        e[-r:] = np.linspace(1, 0, r)
    return e


def bgm(duration: float, mood: str = "wonder", bpm: int = 112, seed: int = 0,
        volume: float = 0.22) -> np.ndarray:
    scale, root, prog = MOODS.get(mood, MOODS["wonder"])
    rng = np.random.default_rng(seed)
    total = int(duration * SR) + SR
    out = np.zeros(total)
    step = 60 / bpm / 2  # 8분음표
    n_step = int(step * SR)
    deg = 0
    for i in range(int(duration / step) + 2):
        s = i * n_step
        if s >= total:
            break
        bar = (i // 8) % len(prog)
        chord = prog[bar]
        # 멜로디: 스케일 위 랜덤 워크, 가끔 쉼표
        if rng.random() > 0.18:
            deg = int(np.clip(deg + rng.choice([-2, -1, 0, 1, 1, 2]), -2, 9))
            midi = root + 12 + scale[deg % 5] + 12 * (deg // 5) + scale[chord % 5]
            length = n_step * (2 if rng.random() < 0.25 else 1)
            n = min(length, total - s)
            out[s:s + n] += _square(_freq(midi), n, duty=0.25) * _env(n) * 0.35
        # 베이스: 박마다 근음
        if i % 2 == 0:
            n = min(n_step * 2, total - s)
            out[s:s + n] += _triangle(_freq(root - 12 + scale[chord % 5]), n) * _env(n, release=0.02) * 0.55
        # 하이햇 / 킥
        if i % 2 == 1:
            n = min(int(0.03 * SR), total - s)
            out[s:s + n] += rng.uniform(-1, 1, n) * np.linspace(0.25, 0, n)
        if i % 8 in (0, 4):
            n = min(int(0.12 * SR), total - s)
            t = np.arange(n) / SR
            out[s:s + n] += np.sin(2 * np.pi * (110 - 400 * t) * t) * np.linspace(0.8, 0, n)
    out = out[: int(duration * SR)]
    fade = int(SR * 1.0)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out / (np.abs(out).max() + 1e-9) * volume


def sfx_boom(duration: float = 0.9, seed: int = 1) -> np.ndarray:
    """흔들림(shake) 효과에 맞춰 쓰는 8비트 폭발음."""
    n = int(duration * SR)
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1, 1, n)
    # 샘플 홀드로 '비트 크러시' 느낌
    hold = 12
    noise = np.repeat(noise[::hold], hold)[:n]
    return noise * np.linspace(1, 0, n) ** 2 * 0.5


def sfx_blip(freq: float = 880, duration: float = 0.06) -> np.ndarray:
    n = int(duration * SR)
    return _square(freq, n, 0.5) * _env(n, release=0.03) * 0.15


def mix(base: np.ndarray, clip: np.ndarray, at: float) -> None:
    s = int(at * SR)
    if s >= len(base):
        return
    n = min(len(clip), len(base) - s)
    base[s:s + n] += clip[:n]


def write_wav(path: str, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
