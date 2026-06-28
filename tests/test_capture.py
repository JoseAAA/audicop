"""Tests for `app.adapters.capture`.

The `soundcard` library is replaced by a fake device so the suite never
touches real audio hardware; we verify our own logic: source selection,
threaded capture lifecycle, WAV mixing and PCM conversion.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.adapters import capture


# ---------------------------------------------------------------------------
# Fake soundcard backend
# ---------------------------------------------------------------------------
class _FakeRec:
    def __enter__(self) -> _FakeRec:
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def record(self, numframes: int) -> np.ndarray:
        return np.zeros((numframes, 1), dtype=np.float32)


class _FakeSource:
    def recorder(self, **_kw: object) -> _FakeRec:
        return _FakeRec()


class _FakeSC:
    def all_speakers(self) -> list[str]:
        return ["spk"]

    def all_microphones(self, include_loopback: bool = False) -> list[str]:
        return ["mic"]

    def default_speaker(self) -> SimpleNamespace:
        return SimpleNamespace(name="spk")

    def default_microphone(self) -> _FakeSource:
        return _FakeSource()

    def get_microphone(self, name: str, include_loopback: bool = False) -> _FakeSource:
        return _FakeSource()


def _write_wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.array(samples, dtype=np.int16).tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_float_block_to_int16_bytes_clips_and_scales() -> None:
    block = np.array([[0.0], [1.0], [-1.0], [0.5], [2.0]], dtype=np.float32)
    vals = np.frombuffer(capture._float_block_to_int16_bytes(block), dtype=np.int16)
    assert list(vals) == [0, 32767, -32767, 16383, 32767]  # 2.0 clipped to 1.0


def test_mix_wavs_averages(tmp_path: Path) -> None:
    a, b, out = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "m.wav"
    _write_wav(a, [1000, 2000, 3000])
    _write_wav(b, [3000, 2000, 1000])
    capture._mix_wavs([a, b], out)
    assert list(_read_wav(out)) == [2000, 2000, 2000]


def test_mix_wavs_aligns_to_shortest(tmp_path: Path) -> None:
    a, b, out = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "m.wav"
    _write_wav(a, [100, 200, 300, 400])
    _write_wav(b, [100, 200])
    capture._mix_wavs([a, b], out)
    assert len(_read_wav(out)) == 2


def test_wav_duration(tmp_path: Path) -> None:
    p = tmp_path / "d.wav"
    _write_wav(p, [0] * 16000)
    assert abs(capture._wav_duration_s(p) - 1.0) < 0.001


def test_wav_duration_bad_file(tmp_path: Path) -> None:
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not a wav")
    assert capture._wav_duration_s(p) == 0.0


# ---------------------------------------------------------------------------
# Availability + source selection
# ---------------------------------------------------------------------------
def test_is_capture_available_true() -> None:
    with patch.object(capture, "_soundcard", return_value=_FakeSC()):
        assert capture.is_capture_available() is True


def test_is_capture_available_false_when_no_devices() -> None:
    fake = _FakeSC()
    fake.all_speakers = lambda: []  # type: ignore[method-assign]
    fake.all_microphones = lambda include_loopback=False: []  # type: ignore[method-assign]
    with patch.object(capture, "_soundcard", return_value=fake):
        assert capture.is_capture_available() is False


def test_is_capture_available_false_on_error() -> None:
    with patch.object(capture, "_soundcard", side_effect=capture.CaptureError("no audio")):
        assert capture.is_capture_available() is False


def test_soundcard_unavailable_raises() -> None:
    """If soundcard failed to import at load time, _soundcard() raises clearly."""
    with patch.object(capture, "_SOUNDCARD", None), pytest.raises(capture.CaptureError):
        capture._soundcard()


def test_is_capture_available_false_when_module_missing() -> None:
    with patch.object(capture, "_SOUNDCARD", None):
        assert capture.is_capture_available() is False


def test_run_with_com_returns_value() -> None:
    """The COM-isolation helper runs the callable on its own thread and returns it."""
    assert capture._run_with_com(lambda: 21 * 2) == 42


def test_run_with_com_propagates_exception() -> None:
    """Exceptions raised inside the helper's worker thread reach the caller."""

    def boom() -> int:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        capture._run_with_com(boom)


def test_loopback_source_falls_back_to_scan() -> None:
    fake = _FakeSC()
    fake.get_microphone = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))  # type: ignore[method-assign]
    fake.all_microphones = lambda include_loopback=False: [  # type: ignore[method-assign]
        SimpleNamespace(isloopback=False, name="real-mic"),
        SimpleNamespace(isloopback=True, name="loopback"),
    ]
    src = capture._loopback_source(fake)
    assert getattr(src, "isloopback", False) is True


def test_loopback_source_not_found_raises() -> None:
    fake = _FakeSC()
    fake.get_microphone = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))  # type: ignore[method-assign]
    fake.all_microphones = lambda include_loopback=False: []  # type: ignore[method-assign]
    with pytest.raises(capture.CaptureError):
        capture._loopback_source(fake)


# ---------------------------------------------------------------------------
# Recorder lifecycle (threaded, with the fake backend)
# ---------------------------------------------------------------------------
def test_recorder_requires_a_source(tmp_path: Path) -> None:
    with pytest.raises(capture.CaptureError):
        capture.Recorder(include_mic=False, include_loopback=False, out_dir=tmp_path)


def test_recorder_meeting_produces_mixed_and_tracks(tmp_path: Path) -> None:
    with patch.object(capture, "_soundcard", return_value=_FakeSC()):
        r = capture.Recorder(include_mic=True, include_loopback=True, out_dir=tmp_path)
        r.start()
        time.sleep(0.15)
        res = r.stop()

    assert res.mixed_path.name == "mixed.wav"
    assert res.mixed_path.exists()
    assert res.mic_path is not None and res.mic_path.exists()
    assert res.others_path is not None and res.others_path.exists()
    assert res.duration_s > 0


def test_recorder_voice_single_track_no_mixing(tmp_path: Path) -> None:
    with patch.object(capture, "_soundcard", return_value=_FakeSC()):
        r = capture.Recorder(include_mic=True, include_loopback=False, out_dir=tmp_path)
        r.start()
        time.sleep(0.15)
        res = r.stop()

    assert res.mixed_path == res.mic_path  # single track is used as-is
    assert res.others_path is None
    assert res.mixed_path.exists()
