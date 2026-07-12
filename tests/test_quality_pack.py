"""Tests for the transcription/summary quality pack.

Covers the repetition collapser (Whisper hallucination loops), the
``<think>`` block stripper for reasoning models, and the per-device beam
width chosen by the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.adapters.local_llm import strip_think_blocks
from app.adapters.transcriber import TranscriptSegment
from app.core import config
from app.services import pipeline
from app.services.formatting import collapse_repetitions
from app.services.pipeline import TranscriptionSettings, iter_transcription


# ---------------------------------------------------------------------------
# collapse_repetitions
# ---------------------------------------------------------------------------
def test_collapses_word_runs_of_three_or_more() -> None:
    assert collapse_repetitions("gracias gracias gracias gracias por venir") == (
        "gracias por venir"
    )


def test_keeps_natural_double_words() -> None:
    text = "sí sí claro que vamos"
    assert collapse_repetitions(text) == text


def test_collapses_phrase_loops() -> None:
    loop = "gracias por ver el video " * 4
    assert collapse_repetitions(loop.strip()) == "gracias por ver el video"


def test_collapses_offset_phrase_loops() -> None:
    # The loop starts mid-sentence — the scanner must catch non-aligned repeats.
    text = "bueno entonces nos vemos mañana nos vemos mañana nos vemos mañana"
    assert collapse_repetitions(text) == "bueno entonces nos vemos mañana"


def test_normal_speech_untouched() -> None:
    text = "El equipo revisó el presupuesto y aprobó la compra de licencias"
    assert collapse_repetitions(text) == text


def test_case_insensitive_but_keeps_first_casing() -> None:
    assert collapse_repetitions("Hola hola HOLA todos") == "Hola todos"


def test_short_text_passthrough() -> None:
    assert collapse_repetitions("ok ok") == "ok ok"


# ---------------------------------------------------------------------------
# strip_think_blocks
# ---------------------------------------------------------------------------
def test_think_block_removed() -> None:
    out = "".join(strip_think_blocks(iter(["<think>plan secreto</think>Hola"])))
    assert out == "Hola"


def test_think_block_split_across_chunks() -> None:
    chunks = ["Antes <th", "ink>razona", "miento</thi", "nk> Después"]
    assert "".join(strip_think_blocks(iter(chunks))) == "Antes  Después"


def test_stream_without_think_untouched() -> None:
    chunks = ["[00:10] ", "Resumen ", "normal."]
    assert "".join(strip_think_blocks(iter(chunks))) == "[00:10] Resumen normal."


def test_unclosed_think_is_suppressed() -> None:
    out = "".join(strip_think_blocks(iter(["Visible <think>nunca cierra..."])))
    assert out == "Visible "


# ---------------------------------------------------------------------------
# Pipeline: per-device beam + segment cleaning
# ---------------------------------------------------------------------------
def _run_pipeline(tmp_path: Path, device: str, text: str) -> tuple[dict, MagicMock]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "in.mp3"
    src.write_bytes(b"fake")
    wav = tmp_path / "work" / "in.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"fakewav")

    fake_info = SimpleNamespace(language="es", language_probability=0.9, duration=4.0)
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.return_value = (
        iter([TranscriptSegment(start=0.0, end=2.0, text=text)]),
        fake_info,
    )
    settings = TranscriptionSettings(
        model_size="tiny",
        compute_type="int8",
        device=device,
        language=None,
        task="transcribe",
        vad_filter=True,
    )
    with (
        patch.object(pipeline, "to_wav_16k", return_value=wav),
        patch.object(pipeline, "get_duration_seconds", return_value=4.0),
        patch.object(pipeline, "get_transcriber", return_value=fake_transcriber),
        patch.object(pipeline, "cleanup"),
        patch.object(pipeline.local_llm, "unload_active"),
    ):
        events = list(iter_transcription(src, settings))
    segment = next(e for e in events if e["type"] == "segment")
    return segment, fake_transcriber


def test_pipeline_beam_cpu_vs_gpu(tmp_path: Path) -> None:
    _, cpu_t = _run_pipeline(tmp_path / "cpu", "cpu", "hola")
    assert cpu_t.transcribe.call_args.kwargs["beam_size"] == config.BEAM_SIZE_CPU
    _, gpu_t = _run_pipeline(tmp_path / "gpu", "cuda", "hola")
    assert gpu_t.transcribe.call_args.kwargs["beam_size"] == config.DEFAULT_BEAM_SIZE


def test_pipeline_collapses_segment_repetitions(tmp_path: Path) -> None:
    segment, _ = _run_pipeline(tmp_path, "cpu", "gracias por ver gracias por ver gracias por ver")
    assert segment["text"] == "gracias por ver"
