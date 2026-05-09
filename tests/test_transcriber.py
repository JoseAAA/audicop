"""Tests for `audicop.transcriber`.

`WhisperModel` is mocked so the suite never has to download real model
weights. We only verify our wrapper's behavior: validation, lazy load,
segment streaming, and error mapping.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from audicop import transcriber
from audicop.transcriber import (
    Transcriber,
    TranscriptionError,
    TranscriptSegment,
)


def _wav_path(tmp_path: Path) -> Path:
    """Return a path to a non-empty file Pretending to be a WAV."""
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFF....WAVEfmt ")  # arbitrary; we mock decoding
    return p


@pytest.fixture
def fake_segments() -> list[SimpleNamespace]:
    """Three fake segments simulating faster-whisper output."""
    return [
        SimpleNamespace(start=0.0, end=1.5, text=" Hola mundo "),
        SimpleNamespace(start=1.5, end=3.2, text="esto es una prueba"),
        SimpleNamespace(start=3.2, end=4.0, text=""),  # silent segment
    ]


@pytest.fixture
def fake_info() -> SimpleNamespace:
    """Fake `TranscriptionInfo` returned by faster-whisper."""
    return SimpleNamespace(language="es", language_probability=0.97, duration=4.0)


def test_invalid_model_size_raises() -> None:
    with pytest.raises(ValueError, match="modelo no soportado"):
        Transcriber(model_size="huge", compute_type="int8", device="cpu")


def test_invalid_compute_type_raises() -> None:
    with pytest.raises(ValueError, match="compute_type no soportado"):
        Transcriber(model_size="tiny", compute_type="bfloat42", device="cpu")


def test_invalid_device_raises() -> None:
    with pytest.raises(ValueError, match="device debe ser"):
        Transcriber(model_size="tiny", compute_type="int8", device="tpu")


def test_load_caches_model() -> None:
    """`load()` builds a `WhisperModel` once and reuses it on subsequent calls."""
    fake_model = MagicMock()
    with patch.object(transcriber, "WhisperModel", return_value=fake_model) as ctor:
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        first = t.load()
        second = t.load()

    assert first is second is fake_model
    ctor.assert_called_once()


def test_load_failure_wraps_exception() -> None:
    """If `WhisperModel(...)` blows up, we surface a `TranscriptionError`."""
    with patch.object(transcriber, "WhisperModel", side_effect=RuntimeError("boom")):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        with pytest.raises(TranscriptionError, match="No se pudo cargar el modelo"):
            t.load()


def test_transcribe_streams_segments(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
    fake_info: SimpleNamespace,
) -> None:
    """`transcribe()` returns an iterator over `TranscriptSegment` and metadata."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), fake_info)

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        segs_iter, info = t.transcribe(_wav_path(tmp_path), language="es")

    segments = list(segs_iter)
    assert len(segments) == 3
    assert all(isinstance(s, TranscriptSegment) for s in segments)
    assert segments[0].text == "Hola mundo"  # stripped
    assert segments[1].start == 1.5
    assert info.language == "es"
    assert info.language_probability == pytest.approx(0.97)
    assert info.duration == pytest.approx(4.0)


def test_transcribe_passes_options(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
    fake_info: SimpleNamespace,
) -> None:
    """language/task/beam_size/vad_filter are forwarded to the underlying model."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), fake_info)

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        t.transcribe(
            _wav_path(tmp_path),
            language="en",
            task="translate",
            beam_size=3,
            vad_filter=False,
        )

    fake_model.transcribe.assert_called_once()
    kwargs = fake_model.transcribe.call_args.kwargs
    assert kwargs["language"] == "en"
    assert kwargs["task"] == "translate"
    assert kwargs["beam_size"] == 3
    assert kwargs["vad_filter"] is False


def test_transcribe_missing_file(tmp_path: Path) -> None:
    """Non-existent WAV raises FileNotFoundError before any model work."""
    fake_model = MagicMock()
    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        with pytest.raises(FileNotFoundError):
            t.transcribe(tmp_path / "missing.wav")
    fake_model.transcribe.assert_not_called()


def test_transcribe_invalid_task_raises(tmp_path: Path) -> None:
    """task must be 'transcribe' or 'translate'."""
    t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
    with pytest.raises(ValueError, match="task debe estar en"):
        t.transcribe(_wav_path(tmp_path), task="summarize")


def test_transcribe_decoding_failure(tmp_path: Path) -> None:
    """Underlying decode error becomes a TranscriptionError."""
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("CUDA OOM")

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        with pytest.raises(TranscriptionError, match="Fallo decodificando"):
            t.transcribe(_wav_path(tmp_path))


def test_transcribe_handles_missing_info_fields(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
) -> None:
    """If faster-whisper returns sparse info, we still build a TranscriptionInfo."""
    sparse_info = SimpleNamespace()  # no attributes at all
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), sparse_info)

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        _, info = t.transcribe(_wav_path(tmp_path))

    assert info.language == "unknown"
    assert info.language_probability == 0.0
    assert info.duration == 0.0
