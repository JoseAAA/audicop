"""Tests for `app.services.pipeline`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.adapters.transcriber import TranscriptSegment
from app.services import pipeline
from app.services.pipeline import TranscriptionSettings, get_transcriber, iter_transcription


def _settings() -> TranscriptionSettings:
    return TranscriptionSettings(
        model_size="tiny",
        compute_type="int8",
        device="cpu",
        language=None,
        task="transcribe",
        vad_filter=True,
    )


def test_get_transcriber_caches(monkeypatch) -> None:
    """get_transcriber builds once per (model, compute, device) and reuses."""
    pipeline._TRANSCRIBER_CACHE.clear()
    built = []

    class FakeT:
        def __init__(self, **kw: object) -> None:
            built.append(kw)

        def load(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "Transcriber", FakeT)
    a = get_transcriber("tiny", "int8", "cpu")
    b = get_transcriber("tiny", "int8", "cpu")
    c = get_transcriber("small", "int8", "cpu")

    assert a is b
    assert c is not a
    assert len(built) == 2  # tiny + small


def test_iter_transcription_emits_event_sequence(tmp_path: Path) -> None:
    """Pipeline yields status → meta → segment(s) → done, and cleans the wav."""
    src = tmp_path / "in.mp3"
    src.write_bytes(b"fake")
    wav = tmp_path / "work" / "in.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"fakewav")

    fake_segments = [
        TranscriptSegment(start=0.0, end=2.0, text="Hola"),
        TranscriptSegment(start=2.0, end=4.0, text="mundo"),
    ]
    fake_info = SimpleNamespace(language="es", language_probability=0.99, duration=4.0)
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.return_value = (iter(fake_segments), fake_info)

    cleanup_calls: list[Path] = []

    with (
        patch.object(pipeline, "to_wav_16k", return_value=wav),
        patch.object(pipeline, "get_duration_seconds", return_value=4.0),
        patch.object(pipeline, "get_transcriber", return_value=fake_transcriber),
        patch.object(pipeline, "cleanup", side_effect=cleanup_calls.append),
    ):
        events = list(iter_transcription(src, _settings()))

    types = [e["type"] for e in events]
    assert types[0] == "status"
    assert "meta" in types
    assert types.count("segment") == 2
    assert types[-1] == "done"

    meta = next(e for e in events if e["type"] == "meta")
    assert meta["duration"] == 4.0
    assert isinstance(meta["estimated_seconds"], float)

    first_seg = next(e for e in events if e["type"] == "segment")
    assert first_seg["text"] == "Hola"
    assert 0.0 <= float(first_seg["pct"]) <= 1.0

    done = events[-1]
    assert done["language"] == "es"
    assert done["duration"] == 4.0

    # the wav's parent dir was cleaned up
    assert cleanup_calls == [wav.parent]


def test_iter_transcription_cleans_up_on_error(tmp_path: Path) -> None:
    """If decoding raises, the wav is still cleaned up (finally)."""
    src = tmp_path / "in.mp3"
    src.write_bytes(b"fake")
    wav = tmp_path / "work" / "in.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"fakewav")

    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.side_effect = RuntimeError("boom")
    cleanup_calls: list[Path] = []

    with (
        patch.object(pipeline, "to_wav_16k", return_value=wav),
        patch.object(pipeline, "get_duration_seconds", return_value=4.0),
        patch.object(pipeline, "get_transcriber", return_value=fake_transcriber),
        patch.object(pipeline, "cleanup", side_effect=cleanup_calls.append),
    ):
        gen = iter_transcription(src, _settings())
        collected_error = False
        try:
            for _ in gen:
                pass
        except RuntimeError:
            collected_error = True

    assert collected_error
    assert cleanup_calls == [wav.parent]
