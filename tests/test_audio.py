"""Tests for `app.adapters.audio`.

The smoke test invokes the real bundled ffmpeg to round-trip a tiny
generated WAV through :func:`app.adapters.audio.to_wav_16k`. That keeps the
test cheap while exercising the full subprocess path.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.adapters import audio
from app.adapters.audio import AudioConversionError, cleanup, get_duration_seconds, to_wav_16k


def _write_sine_wav(path: Path, *, sample_rate: int = 22050, seconds: float = 0.25) -> None:
    """Generate a tiny stereo WAV file (so ffmpeg has work to do)."""
    n_frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # Cheap "audio": alternating samples, low amplitude.
        frames = b"".join(
            struct.pack("<hh", (i % 200) - 100, (i % 200) - 100) for i in range(n_frames)
        )
        wav.writeframes(frames)


def test_to_wav_16k_smoke(tmp_path: Path) -> None:
    """Real ffmpeg call: a tiny stereo WAV becomes mono 16 kHz."""
    src = tmp_path / "input.wav"
    _write_sine_wav(src)

    dest_dir = tmp_path / "out"
    out = to_wav_16k(src, dest_dir=dest_dir)

    assert out.exists()
    assert out.suffix == ".wav"
    assert out.parent == dest_dir

    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 16_000
        assert wav.getnchannels() == 1


def test_to_wav_16k_missing_source(tmp_path: Path) -> None:
    """Non-existent input raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        to_wav_16k(tmp_path / "missing.mp3")


def test_to_wav_16k_ffmpeg_failure(tmp_path: Path) -> None:
    """If ffmpeg exits non-zero we surface a friendly AudioConversionError."""
    src = tmp_path / "input.wav"
    _write_sine_wav(src)

    fake_run = MagicMock(
        return_value=MagicMock(returncode=1, stderr="boom: invalid stream", stdout="")
    )
    with (
        patch.object(audio.subprocess, "run", fake_run),
        pytest.raises(AudioConversionError) as exc_info,
    ):
        to_wav_16k(src, dest_dir=tmp_path / "out")

    assert "ffmpeg" in str(exc_info.value).lower()
    assert "boom" in str(exc_info.value)


def test_get_duration_seconds(tmp_path: Path) -> None:
    """Duration of a 0.25 s 22050 Hz WAV is ~0.25 s."""
    src = tmp_path / "input.wav"
    _write_sine_wav(src, sample_rate=22050, seconds=0.25)
    duration = get_duration_seconds(src)
    assert duration == pytest.approx(0.25, abs=0.05)


def test_get_duration_seconds_invalid_file(tmp_path: Path) -> None:
    """Invalid WAV → returns 0.0 instead of raising."""
    bogus = tmp_path / "not-a-wav.wav"
    bogus.write_bytes(b"definitely not a wav header")
    assert get_duration_seconds(bogus) == 0.0


def test_cleanup_directory(tmp_path: Path) -> None:
    """cleanup removes a directory tree without raising."""
    target = tmp_path / "scratch"
    target.mkdir()
    (target / "file.txt").write_text("hi")
    cleanup(target)
    assert not target.exists()


def test_cleanup_file(tmp_path: Path) -> None:
    """cleanup removes a single file without raising."""
    target = tmp_path / "f.txt"
    target.write_text("hi")
    cleanup(target)
    assert not target.exists()


def test_cleanup_missing_path(tmp_path: Path) -> None:
    """cleanup is a no-op for a non-existent path."""
    cleanup(tmp_path / "does-not-exist")  # must not raise
