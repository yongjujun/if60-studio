"""내레이션 음성(TTS) — 장면마다 목소리를 먼저 만들고, 그 길이에 맞춰 장면·자막·연출 타이밍을 맞춘다.

목소리 지정: "제공자:목소리"
  supertonic:M1      내 컴퓨터에서 직접 생성 (무료·무제한, 키 필요 없음) — 기본값
  typecast:tc_...    타입캐스트 API (무료 플랜 월 15,000자 · 상업 이용 불가 · 출처 표기 필수)
  google:Charon      구글 Chirp 3 HD (월 100만 자 무료 · 결제 계정 등록 필요)

우선순위: 명령의 --voice > 장면 파일의 voice > .env의 IF60_VOICE > supertonic:M1
같은 문장·목소리는 .cache/tts에 저장해 두고 다시 만들지 않는다 (렌더링을 여러 번 해도 사용량이 늘지 않음).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "tts"
SR = 48000                       # 최종 음성 트랙 샘플레이트
DEFAULT_VOICE = "supertonic:M1"
PROVIDERS = ("supertonic", "typecast", "google")
TYPECAST_CREDIT = "이 콘텐츠는 인공지능 가상 연기자 서비스, 타입캐스트를 활용하여 제작되었습니다."


@dataclass
class Voice:
    provider: str
    name: str
    speed: float = 1.0
    options: dict = field(default_factory=dict)   # emotion, pitch 등 제공자별 옵션

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.name}"


def parse_voice(v) -> Voice:
    """'typecast:tc_123' 또는 {provider, name, speed, emotion, pitch} → Voice"""
    if isinstance(v, Voice):
        return v
    if isinstance(v, dict):
        d = dict(v)
        prov = d.pop("provider", "supertonic")
        name = str(d.pop("name", d.pop("voice", "")))
        speed = float(d.pop("speed", 1.0))
        return _check(Voice(prov, name, speed, d))
    s = str(v or DEFAULT_VOICE).strip()
    prov, _, name = s.partition(":")
    return _check(Voice(prov.strip().lower(), name.strip()))


def _check(v: Voice) -> Voice:
    if v.provider not in PROVIDERS:
        raise ValueError(f"목소리 제공자 '{v.provider}'는 지원하지 않아요 (가능: {', '.join(PROVIDERS)})")
    if not v.name:
        v.name = {"supertonic": "M1", "google": "Charon"}.get(v.provider, "")
        if not v.name:
            raise ValueError("타입캐스트는 목소리 ID가 필요해요. 예: typecast:tc_xxxx  (목록: python -m if60 voice list typecast)")
    return v


def default_voice() -> str:
    from .planner import load_env
    load_env()
    return os.environ.get("IF60_VOICE", "").strip() or DEFAULT_VOICE


# ------------------------------------------------------------------ 읽기 좋은 문장
_REPL = [("℃", "도"), ("°C", "도"), ("%", " 퍼센트"), ("×", " "), ("…", "."), ("~", "에서 "), ("→", ", "),
         ("·", ", "), ("“", ""), ("”", ""), ('"', ""), ("'", ""), ("‘", ""), ("’", "")]


def speech_text(text: str) -> str:
    """자막용 기호를 소리 내 읽기 좋은 형태로 (℃→도, %→퍼센트 등). 정확한 읽기는 장면의 narration으로 직접 쓰는 게 가장 좋다."""
    from .text import safe_text
    s = safe_text(str(text))
    for a, b in _REPL:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ------------------------------------------------------------------ 오디오 유틸
def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise RuntimeError(f"지원하지 않는 WAV 형식 ({sw * 8}bit)")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def encode_wav(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def resample(x: np.ndarray, sr: int, to: int = SR) -> np.ndarray:
    if sr == to or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * to / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, thr: float = 0.012, keep: float = 0.04) -> np.ndarray:
    """앞뒤 무음을 잘라 자막·화면과 딱 맞게 (말소리 앞뒤 40ms만 남김)"""
    if len(x) == 0:
        return x
    win = max(1, int(sr * 0.01))
    env = np.convolve(np.abs(x), np.ones(win) / win, mode="same")
    idx = np.nonzero(env > thr)[0]
    if len(idx) == 0:
        return x[:0]
    k = int(sr * keep)
    return x[max(0, idx[0] - k): min(len(x), idx[-1] + k)]


def normalize(x: np.ndarray, peak: float = 0.89) -> np.ndarray:
    m = float(np.abs(x).max()) if len(x) else 0.0
    return x if m < 1e-6 else (x * (peak / m)).astype(np.float32)


# ------------------------------------------------------------------ 제공자
_supertonic = None


def _synth_supertonic(text: str, v: Voice, context: dict) -> tuple[np.ndarray, int]:
    global _supertonic
    try:
        from supertonic import TTS
    except ImportError:
        raise RuntimeError("Supertonic이 설치되어 있지 않아요. 터미널에서:  pip install supertonic") from None
    if _supertonic is None:
        print("  · Supertonic 음성 모델 준비 중 (처음 한 번은 내려받느라 1~2분 걸려요)")
        _supertonic = TTS(auto_download=True)
    tts = _supertonic
    names = list(getattr(tts, "voice_style_names", []) or [])
    if names and v.name not in names:
        raise ValueError(f"Supertonic 목소리 '{v.name}' 없음 (가능: {', '.join(names)})")
    style = tts.get_voice_style(v.name)
    wav, _ = tts.synthesize(text, voice_style=style, lang="ko", speed=1.05 * v.speed,
                            total_steps=int(v.options.get("steps", 10)))
    return np.asarray(wav, np.float32).squeeze(), int(tts.sample_rate)


def _key(var: str, provider: str) -> str:
    from .planner import load_env
    load_env()
    k = os.environ.get(var, "").strip()
    if not k:
        raise RuntimeError(f"{provider} API 키가 없어요. 터미널에서:  python -m if60 key {provider.lower()}")
    return k


def _http(req: urllib.request.Request, what: str) -> bytes:
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        hint = {401: "키가 잘못됐어요", 402: "이번 달 무료 사용량(크레딧)을 다 썼어요", 403: "권한이 없어요 (API 사용 설정 확인)",
                404: "목소리 ID를 찾을 수 없어요", 429: "요청이 너무 많아요 — 잠시 후 다시"}.get(e.code, "")
        raise RuntimeError(f"{what} 오류 {e.code} {hint}: {detail}") from None


def _synth_typecast(text: str, v: Voice, context: dict) -> tuple[np.ndarray, int]:
    key = _key("TYPECAST_API_KEY", "Typecast")
    emotion = v.options.get("emotion", "smart")
    if emotion == "smart":     # 앞뒤 문장을 보고 감정을 알아서 정함
        prompt = {"emotion_type": "smart", "previous_text": context.get("prev", "")[:2000],
                  "next_text": context.get("next", "")[:2000]}
    else:
        prompt = {"emotion_type": "preset", "emotion_preset": emotion,
                  "emotion_intensity": float(v.options.get("intensity", 1.0))}
    body = {"voice_id": v.name, "text": text, "model": v.options.get("model", "ssfm-v30"), "language": "kor",
            "prompt": prompt,
            "output": {"audio_format": "wav", "audio_tempo": max(0.5, min(2.0, v.speed)),
                       "audio_pitch": int(v.options.get("pitch", 0)), "target_lufs": -14}}
    req = urllib.request.Request("https://api.typecast.ai/v1/text-to-speech", data=json.dumps(body).encode(),
                                 headers={"X-API-KEY": key, "Content-Type": "application/json"})
    return decode_wav(_http(req, "타입캐스트"))


def _synth_google(text: str, v: Voice, context: dict) -> tuple[np.ndarray, int]:
    key = _key("GOOGLE_TTS_API_KEY", "Google")
    name = v.name if v.name.startswith("ko-KR-") else f"ko-KR-Chirp3-HD-{v.name}"
    body = {"input": {"text": text}, "voice": {"languageCode": "ko-KR", "name": name},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000,
                            "speakingRate": max(0.25, min(2.0, v.speed))}}
    req = urllib.request.Request("https://texttospeech.googleapis.com/v1/text:synthesize",
                                 data=json.dumps(body).encode(),
                                 headers={"X-Goog-Api-Key": key, "Content-Type": "application/json"})
    data = json.loads(_http(req, "구글 TTS"))
    return decode_wav(base64.b64decode(data["audioContent"]))


SYNTH = {"supertonic": _synth_supertonic, "typecast": _synth_typecast, "google": _synth_google}


def synthesize(text: str, voice, context: dict | None = None, cache: bool = True) -> np.ndarray:
    """문장 → 48kHz 모노 float32 (앞뒤 무음 제거). 결과는 캐시에 저장."""
    v = parse_voice(voice)
    text = speech_text(text)
    if not text:
        return np.zeros(0, np.float32)
    ctx = context or {}
    sig = json.dumps([v.provider, v.name, v.speed, v.options, text,
                      ctx.get("prev", "") if v.provider == "typecast" else "",
                      ctx.get("next", "") if v.provider == "typecast" else ""], ensure_ascii=False)
    path = CACHE / f"{hashlib.sha1(sig.encode()).hexdigest()[:20]}.wav"
    if cache and path.exists():
        x, sr = decode_wav(path.read_bytes())
        return resample(x, sr)
    x, sr = SYNTH[v.provider](text, v, ctx)
    x = normalize(trim_silence(resample(x, sr), SR))
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode_wav(x, SR))
    return x


# ------------------------------------------------------------------ 목소리 목록
def list_voices(provider: str) -> list[dict]:
    provider = provider.lower()
    if provider == "supertonic":
        try:
            from supertonic import TTS
        except ImportError:
            raise RuntimeError("Supertonic이 설치되어 있지 않아요:  pip install supertonic") from None
        global _supertonic
        if _supertonic is None:
            _supertonic = TTS(auto_download=True)
        desc = {"M": "남성", "F": "여성"}
        return [{"id": n, "name": f"{desc.get(n[:1], '')} {n}".strip(), "info": ""}
                for n in sorted(_supertonic.voice_style_names)]
    if provider == "typecast":
        key = _key("TYPECAST_API_KEY", "Typecast")
        req = urllib.request.Request("https://api.typecast.ai/v3/voices?model=ssfm-v30", headers={"X-API-KEY": key})
        out = []
        for vv in json.loads(_http(req, "타입캐스트")):
            names = vv.get("voice_name") or {}
            nm = (names.get("kor") or names.get("eng") or "") if isinstance(names, dict) else str(names)
            info = " · ".join(str(x) for x in [vv.get("gender"), vv.get("age"), ", ".join(vv.get("use_cases") or [])] if x)
            out.append({"id": vv["voice_id"], "name": nm, "info": info, "preview": vv.get("preview_url")})
        _save_typecast_names(out)
        return out
    if provider == "google":
        key = _key("GOOGLE_TTS_API_KEY", "Google")
        req = urllib.request.Request("https://texttospeech.googleapis.com/v1/voices?languageCode=ko-KR",
                                     headers={"X-Goog-Api-Key": key})
        out = []
        for vv in json.loads(_http(req, "구글 TTS")).get("voices", []):
            n = vv.get("name", "")
            if "Chirp3-HD" in n:
                out.append({"id": n.split("Chirp3-HD-")[-1], "name": n,
                            "info": {"MALE": "남성", "FEMALE": "여성"}.get(vv.get("ssmlGender"), "")})
        return out
    raise ValueError(f"제공자 '{provider}' 없음 (가능: {', '.join(PROVIDERS)})")


def _save_typecast_names(voices: list[dict]) -> None:
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "typecast_voices.json").write_text(
            json.dumps({v["id"]: v["name"] for v in voices}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def typecast_name(voice_id: str) -> str:
    try:
        return json.loads((CACHE / "typecast_voices.json").read_text(encoding="utf-8")).get(voice_id, voice_id)
    except (OSError, ValueError):
        return voice_id


def credit_text(v: Voice) -> str | None:
    """게시물 설명란에 넣을 표기 문구.
    타입캐스트 무료 플랜: 출처 표기 필수. Supertonic: 라이선스상 AI 생성물임을 밝혀야 함."""
    if v.provider == "typecast":
        return f"{TYPECAST_CREDIT}\n출연진: {typecast_name(v.name)}\nhttps://typecast.ai/"
    if v.provider == "supertonic":
        return "이 영상의 내레이션은 AI 음성(Supertonic)으로 만들었습니다."
    return None
