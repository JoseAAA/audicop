"""Transcription pipeline as a framework-agnostic event stream.

This is the heart that the API layer drives. :func:`iter_transcription`
runs the (blocking) decode and **yields plain event dicts** as it goes —
status, meta, segment (with progress), and done. It knows nothing about
FastAPI or asyncio, so it is fully unit-testable; the API runs it in a
thread and bridges the events to SSE.

Event schema (each is a ``dict`` with a ``"type"`` key):
    - ``{"type": "status", "label": str}`` — phase label for the UI.
    - ``{"type": "meta", "duration": float, "estimated_seconds": float}``
      — known after audio conversion, before decoding.
    - ``{"type": "segment", "start": float, "end": float, "text": str,
        "pct": float, "elapsed": float, "eta": float | None}``
    - ``{"type": "done", "language": str, "language_probability": float,
        "duration": float}``

Errors are not events: the generator raises (``AudioConversionError``,
``TranscriptionError``, ``FileNotFoundError``) and the caller decides how
to surface them.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.adapters import local_llm
from app.adapters.audio import cleanup, get_duration_seconds, to_wav_16k
from app.adapters.transcriber import Transcriber
from app.core import config
from app.services.formatting import collapse_repetitions

logger = logging.getLogger(__name__)

# Below this fraction / elapsed we don't show an ETA (too noisy to be useful).
_ETA_MIN_PCT = 0.02
_ETA_MIN_ELAPSED_S = 1.0


@dataclass(frozen=True, slots=True)
class TranscriptionSettings:
    """Resolved settings for one transcription run.

    Attributes:
        model_size: Whisper model size (e.g. ``"large-v3"``).
        compute_type: CTranslate2 compute type (e.g. ``"int8_float16"``).
        device: ``"cuda"`` or ``"cpu"``.
        language: Two-letter code, or ``None`` to autodetect.
        task: ``"transcribe"`` or ``"translate"``.
        vad_filter: Whether to apply the Silero VAD filter.
        initial_prompt: Optional vocabulary hint (names, jargon) to prime
            the decoder, or ``None``.
    """

    model_size: str
    compute_type: str
    device: str
    language: str | None
    task: str
    vad_filter: bool
    initial_prompt: str | None = None


_TRANSCRIBER_CACHE: dict[tuple[str, str, str], Transcriber] = {}


def get_transcriber(model_size: str, compute_type: str, device: str) -> Transcriber:
    """Return a cached :class:`Transcriber`, loading the model on first use.

    Caching by ``(model, compute, device)`` avoids reloading multi-GB
    weights on every request.
    """
    key = (model_size, compute_type, device)
    transcriber = _TRANSCRIBER_CACHE.get(key)
    if transcriber is None:
        transcriber = Transcriber(model_size=model_size, compute_type=compute_type, device=device)
        transcriber.load()
        _TRANSCRIBER_CACHE[key] = transcriber
    return transcriber


def release_transcribers() -> None:
    """Unload every cached Whisper model and clear the cache.

    Audicop's phases are sequential — transcribe first, analyze after — and
    both models are multi-GB, so they must never sit in memory together.
    Called when a transcription run ends: from that point the memory belongs
    to the local LLM. The next transcription reloads Whisper from the on-disk
    cache (a few seconds), a fair price for fitting on modest machines.
    """
    if not _TRANSCRIBER_CACHE:
        return
    for transcriber in _TRANSCRIBER_CACHE.values():
        transcriber.unload()
    _TRANSCRIBER_CACHE.clear()
    logger.info("Whisper liberado: la memoria queda disponible para el LLM local.")


def iter_transcription(
    media_path: Path, settings: TranscriptionSettings
) -> Iterator[dict[str, object]]:
    """Run the full pipeline for one media file, yielding progress events.

    Args:
        media_path: Path to the source audio/video on disk.
        settings: Resolved model/language/task settings.

    Yields:
        Event dicts (see module docstring).

    Raises:
        FileNotFoundError: If ``media_path`` does not exist.
        AudioConversionError: If ffmpeg fails.
        TranscriptionError: If the model fails to load or decode.
    """
    wav_path: Path | None = None
    try:
        yield {"type": "status", "label": "Extrayendo audio a 16 kHz mono…"}
        wav_path = to_wav_16k(media_path)
        duration = get_duration_seconds(wav_path) or 0.0
        estimated = config.estimate_processing_seconds(
            duration, settings.model_size, settings.device
        )
        yield {"type": "meta", "duration": duration, "estimated_seconds": estimated}

        yield {"type": "status", "label": "Cargando modelo…"}
        # Symmetric memory handoff: if the analysis LLM is loaded, free it
        # first — Whisper is about to need that memory (phases never overlap).
        local_llm.unload_active()
        transcriber = get_transcriber(settings.model_size, settings.compute_type, settings.device)

        yield {"type": "status", "label": "Transcribiendo…"}
        # Beam width per device: wide beam is ~free on GPU but multiplies CPU
        # wall time for little gain (see config.BEAM_SIZE_CPU).
        beam_size = config.DEFAULT_BEAM_SIZE if settings.device == "cuda" else config.BEAM_SIZE_CPU
        segments_iter, info = transcriber.transcribe(
            wav_path,
            language=settings.language,
            task=settings.task,
            beam_size=beam_size,
            vad_filter=settings.vad_filter,
            initial_prompt=settings.initial_prompt,
        )
        effective_duration = duration or info.duration or 0.0
        start = time.monotonic()

        for seg in segments_iter:
            elapsed = time.monotonic() - start
            pct = min(seg.end / effective_duration, 1.0) if effective_duration > 0 else 0.0
            eta: float | None = None
            if pct >= _ETA_MIN_PCT and elapsed > _ETA_MIN_ELAPSED_S:
                eta = max(0.0, elapsed / pct - elapsed)
            yield {
                "type": "segment",
                "start": seg.start,
                "end": seg.end,
                # Collapse Whisper's hallucinated repetition loops before the
                # text reaches the UI / store / LLM context.
                "text": collapse_repetitions(seg.text),
                "pct": pct,
                "elapsed": elapsed,
                "eta": eta,
            }

        yield {
            "type": "done",
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": effective_duration,
            # Internal: lets the API worker copy the audio into the meetings
            # library BEFORE the finally-cleanup deletes it. Stripped from the
            # event before it reaches the browser.
            "wav_path": str(wav_path),
        }
    finally:
        if wav_path is not None:
            cleanup(wav_path.parent)
        # Whisper's job is done (success, error or abandoned stream): release
        # it so the analysis phase gets the memory.
        release_transcribers()
