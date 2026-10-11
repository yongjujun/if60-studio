"""내레이션(TTS): 목소리 길이에 맞춘 장면·자막·연출 싱크, 제공자 요청 형식, 출처 표기."""
import base64
import io
import json
import shutil
import subprocess

import numpy as np
import pytest

from if60 import render as R
from if60 import tts


def fake_audio(text, v, ctx):
    """글자 수에 비례하는 길이(1글자 0.1초)의 가짜 목소리 — 앞뒤에 무음 0.3초"""
    sr = 24000
    n = int(len(text) * 0.1 * sr)
    t = np.arange(n) / sr
    x = np.concatenate([np.zeros(int(0.3 * sr)), 0.4 * np.sin(2 * np.pi * 220 * t), np.zeros(int(0.3 * sr))])
    return x.astype(np.float32), sr


@pytest.fixture
def fake_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "CACHE", tmp_path / "cache")
    calls = []

    def synth(text, v, ctx):
        calls.append(text)
        return fake_audio(text, v, ctx)
    monkeypatch.setitem(tts.SYNTH, "supertonic", synth)
    return calls


def test_parse_voice_and_speech_text():
    v = tts.parse_voice("typecast:tc_abc")
    assert (v.provider, v.name) == ("typecast", "tc_abc")
    assert tts.parse_voice("google").name == "Charon"
    assert tts.parse_voice({"provider": "supertonic", "name": "F3", "speed": 1.1}).speed == 1.1
    with pytest.raises(ValueError):
        tts.parse_voice("typecast")              # 타입캐스트는 ID 필수
    with pytest.raises(ValueError):
        tts.parse_voice("capcut:abc")
    assert tts.speech_text("약 27,700℃, 78%…") == "약 27,700도, 78 퍼센트."


def test_scenes_follow_voice_length_and_events_move_with_them(fake_voice):
    sc = R.Scene(duration=6.0, layers=[], caption="번개는 아주 뜨거워요.",
                 effects=[("flash", {"at": 3.0, "length": 0.2})],
                 big_text=[{"text": "×5", "from": 3.0}], vfx=[{"type": "orb", "from": 1.5, "to": 4.5}],
                 panel={"value": {"from": 0, "to": 100, "at": 1.5, "length": 3.0}})
    sc2 = R.Scene(duration=2.0, layers=[], caption="그런데 사실 번개의 온도는 태양 표면보다 다섯 배나 높아요.")
    track = R.apply_voice([sc, sc2], "supertonic:M1", log=lambda *a: None)
    assert sc.audio_len == pytest.approx(len(sc.caption) * 0.1 + 0.08, abs=0.03)   # 앞뒤 무음은 잘림
    assert sc.duration == pytest.approx(R.GAP_MIN if sc.audio_len + 0.43 < R.GAP_MIN else 0.08 + sc.audio_len + R.TAIL, abs=0.01)
    k = sc.duration / 6.0
    assert sc.effects[0][1]["at"] == pytest.approx(3.0 * k)            # 연출도 같은 비율로 이동
    assert sc.big_text[0]["from"] == pytest.approx(3.0 * k)
    assert sc.vfx[0]["to"] == pytest.approx(4.5 * k)
    assert sc.panel["value"]["length"] == pytest.approx(3.0 * k)
    assert sc2.duration > 2.0                                            # 긴 대사는 장면이 늘어남
    assert len(track) == int((sc.duration + sc2.duration) * tts.SR)
    # 두 번째 장면 대사가 시작하는 지점에 소리가 있다
    a = int((sc.duration + sc2.audio_start + 0.1) * tts.SR)
    assert np.abs(track[a:a + 2000]).max() > 0.1
    # 자막 덩어리는 말하는 구간 안에서만 넘어간다
    chunks = R.text.chunk_caption(sc2.caption)
    assert R.caption_at(sc2, sc2.audio_start + 0.01) == chunks[0]
    assert R.caption_at(sc2, sc2.audio_start + sc2.audio_len - 0.01) == chunks[-1]


def test_voice_cache_avoids_second_call(fake_voice):
    tts.synthesize("같은 문장", "supertonic:M1")
    tts.synthesize("같은 문장", "supertonic:M1")
    assert fake_voice == ["같은 문장"]


def test_card_scene_reads_question_and_keeps_time_for_options(fake_voice):
    sc = R.Scene(duration=6.0, layers=[], card={"question": "어디에 쓸까요?", "options": ["가", "나", "다"]})
    R.apply_voice([sc], "supertonic:M1", log=lambda *a: None)
    assert fake_voice == ["어디에 쓸까요?"]
    assert sc.duration >= 1.6 + 0.35 * 3


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _wav_bytes():
    x, sr = fake_audio("가나다", None, None)
    return tts.encode_wav(x, sr)


def test_typecast_request_and_credit(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "CACHE", tmp_path / "cache")
    monkeypatch.setenv("TYPECAST_API_KEY", "tc-test-key-1234567890")
    sent = []

    def fake_urlopen(req, timeout=0):
        sent.append(req)
        return FakeResp(_wav_bytes())
    monkeypatch.setattr(tts.urllib.request, "urlopen", fake_urlopen)
    x = tts.synthesize("번개는 뜨거워요.", "typecast:tc_abc", {"prev": "앞 문장", "next": "뒤 문장"})
    assert len(x) > 0
    req = sent[0]
    assert req.full_url == "https://api.typecast.ai/v1/text-to-speech"
    assert req.get_header("X-api-key") == "tc-test-key-1234567890"
    body = json.loads(req.data)
    assert body["voice_id"] == "tc_abc" and body["model"] == "ssfm-v30" and body["language"] == "kor"
    assert body["prompt"] == {"emotion_type": "smart", "previous_text": "앞 문장", "next_text": "뒤 문장"}
    assert "tc-test-key" not in req.full_url                     # 키는 주소가 아니라 헤더로
    credit = tts.credit_text(tts.parse_voice("typecast:tc_abc"))
    assert tts.TYPECAST_CREDIT in credit and "https://typecast.ai/" in credit


def test_google_request(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "CACHE", tmp_path / "cache")
    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "g-test-key-1234567890")
    sent = []

    def fake_urlopen(req, timeout=0):
        sent.append(req)
        return FakeResp(json.dumps({"audioContent": base64.b64encode(_wav_bytes()).decode()}).encode())
    monkeypatch.setattr(tts.urllib.request, "urlopen", fake_urlopen)
    assert len(tts.synthesize("안녕하세요", "google:Kore")) > 0
    body = json.loads(sent[0].data)
    assert body["voice"]["name"] == "ko-KR-Chirp3-HD-Kore"
    assert sent[0].get_header("X-goog-api-key") == "g-test-key-1234567890"


def test_missing_key_gives_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "CACHE", tmp_path / "cache")
    monkeypatch.delenv("TYPECAST_API_KEY", raising=False)
    from if60 import planner
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="python -m if60 key typecast"):
        tts.synthesize("안녕", "typecast:tc_abc")


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffmpeg 필요")
def test_render_with_voice_has_audio_and_matching_length(tmp_path, fake_voice):
    src = tmp_path / "s.yaml"
    src.write_text(
        "title: 목소리 테스트\nstyle: voxel\nvoice: supertonic:M1\nscenes:\n"
        "  - duration: 5\n    caption: 안녕하세요.\n    world: {template: studio, seed: 1}\n    shot: front\n"
        "  - duration: 1\n    caption: 두 번째 장면은 대사가 길어서 장면도 길어져요.\n"
        "    world: {template: studio, seed: 1}\n    shot: front\n", encoding="utf-8")
    out = tmp_path / "o.mp4"
    R.render(str(src), str(out), preview=True, fps=4)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                                      str(out)], capture_output=True, text=True).stdout)
    assert sorted(s["codec_type"] for s in info["streams"]) == ["audio", "video"]
    dur = float(info["format"]["duration"])
    expect = (0.08 + 0.6 + R.TAIL) + (R.LEAD + 2.5 + R.TAIL)
    assert abs(dur - max(expect, R.GAP_MIN)) < 0.6
    assert (tmp_path / "o_voice.wav").exists()
