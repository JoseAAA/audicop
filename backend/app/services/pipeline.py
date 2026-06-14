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

from app.adapters.audio import cleanup, get_duration_seconds, to_wav_16k
from app.adapters.transcriber import Transcriber
from app.core import config

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
    """

    model_size: str
    compute_type: str
    device: str
    language: str | None
    task: str
    vad_filter: bool


_TRANSCRIBER_CACHE: dict[tuple[str, str, str], Transcriber] = {}


def get_transcriber(model_size: str, compute_type: str, device: str) -> Transcriber:
    """Return a cached :class:`Transcriber`, loading the model on first use.

    Caching by ``(model, compute, device)`` avoids reloading multi-GB
    weights on every request — the equivalent of Streamlit's
    ``st.cache_resource``.
    """
    key = (model_size, compute_type, device)
    transcriber = _TRANSCRIBER_CACHE.get(key)
    if transcriber is None:
        transcriber = Transcriber(model_size=model_size, compute_type=compute_type, device=device)
        transcriber.load()
        _TRANSCRIBER_CACHE[key] = transcriber
    return transcriber


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
        transcriber = get_transcriber(settings.model_size, settings.compute_type, settings.device)

        yield {"type": "status", "label": "Transcribiendo…"}
        segments_iter, info = transcriber.transcribe(
            wav_path,
            language=settings.language,
            task=settings.task,
            vad_filter=settings.vad_filter,
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
                "text": seg.text,
                "pct": pct,
                "elapsed": elapsed,
                "eta": eta,
            }

        yield {
            "type": "done",
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": effective_duration,
        }
    finally:
        if wav_path is not None:
            cleanup(wav_path.parent)
