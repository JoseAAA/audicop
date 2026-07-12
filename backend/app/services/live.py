"""Live transcription while a recording is in progress.

The capture threads feed raw PCM blocks into a :class:`LiveTranscriber`
(the *tap*). A worker thread mixes the tracks online, accumulates a few
seconds of audio, and decodes each window with Whisper, emitting draft
segments the UI streams as "transcripción en vivo".

It is a **preview**: windows are cut by time (a word can straddle a
boundary), so when the recording stops the normal full-file transcription
still runs and produces the definitive result. Decoding is injected as a
callable, keeping this module engine-agnostic and unit-testable, and every
failure here is swallowed after logging — live preview must never break the
recording itself.
"""

from __future__ import annotations

import logging
import threading
import wave
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from app.adapters.transcriber import TranscriptSegment
from app.core import config
from app.services.formatting import collapse_repetitions

logger = logging.getLogger(__name__)

_POLL_S: float = 0.2
"""How often the worker checks for newly fed audio."""


def _write_wav(samples: np.ndarray, path: Path) -> None:
    """Write mono int16 samples to a 16 kHz WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(config.TARGET_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(config.TARGET_SAMPLE_RATE)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _to_mono_int16(block: np.ndarray) -> np.ndarray:
    """Convert a float32 capture block (possibly stereo) to mono int16."""
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)


class LiveTranscriber:
    """Mixes tapped capture blocks and decodes them in near-real-time windows.

    Args:
        tracks: Track labels that will feed blocks (e.g. ``("others", "mic")``).
        work_dir: Directory for the per-window scratch WAV.
        decode: Callable decoding a WAV path into transcript segments.
        on_event: Callback receiving each event dict:
            ``{"start", "end", "text"}`` per draft segment, then
            ``{"done": True}`` exactly once when the live stream ends.
        window_s: Seconds of audio accumulated before each decode.
    """

    def __init__(
        self,
        *,
        tracks: Sequence[str],
        work_dir: Path,
        decode: Callable[[Path], Sequence[TranscriptSegment]],
        on_event: Callable[[dict[str, object]], None],
        window_s: float = config.LIVE_WINDOW_S,
    ) -> None:
        """Set up the tap buffers and worker state (call :meth:`start` next)."""
        self._queues: dict[str, deque[np.ndarray]] = {t: deque() for t in tracks}
        self._work_dir = work_dir
        self._decode = decode
        self._on_event = on_event
        self._window_samples = int(window_s * config.TARGET_SAMPLE_RATE)
        self._min_tail_samples = int(config.LIVE_MIN_TAIL_S * config.TARGET_SAMPLE_RATE)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: list[np.ndarray] = []
        self._pending_samples = 0
        self._offset_samples = 0
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Tap (called from capture threads — must be fast and never raise)
    # ------------------------------------------------------------------
    def feed(self, track: str, block: np.ndarray) -> None:
        """Receive one captured block for ``track`` (thread-safe, non-blocking)."""
        queue = self._queues.get(track)
        if queue is None or self._stop.is_set():
            return
        with self._lock:
            queue.append(_to_mono_int16(block))

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the mixing/decoding worker thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="audicop-live")
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker, flush the remaining tail, and emit the done event."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_queues()
                while self._pending_samples >= self._window_samples:
                    self._decode_pending(self._window_samples)
                self._stop.wait(_POLL_S)
            self._drain_queues()
            if self._pending_samples >= self._min_tail_samples:
                self._decode_pending(None)
        except Exception:
            logger.exception("Live transcription worker failed")
        finally:
            self._emit({"done": True})

    def _drain_queues(self) -> None:
        """Mix pairwise-aligned blocks from every track into the pending buffer."""
        with self._lock:
            ready = min(len(q) for q in self._queues.values())
            batches = [[q.popleft() for _ in range(ready)] for q in self._queues.values()]
        for blocks in zip(*batches, strict=True):  # per-track batches share `ready` length
            shortest = min(b.shape[0] for b in blocks)
            acc = np.zeros(shortest, dtype=np.int32)
            for b in blocks:
                acc += b[:shortest].astype(np.int32)
            mixed = (acc // len(blocks)).astype(np.int16)
            self._pending.append(mixed)
            self._pending_samples += mixed.shape[0]

    def _decode_pending(self, take_samples: int | None) -> None:
        """Decode one window (or everything, on flush) with global offsets.

        ``take_samples`` caps the window to an exact size so consecutive
        windows tile the timeline; the remainder stays pending for the next
        decode. ``None`` (final flush) takes it all.
        """
        samples = np.concatenate(self._pending)
        if take_samples is not None and samples.shape[0] > take_samples:
            remainder = samples[take_samples:]
            samples = samples[:take_samples]
            self._pending = [remainder]
            self._pending_samples = int(remainder.shape[0])
        else:
            self._pending = []
            self._pending_samples = 0
        base_s = self._offset_samples / config.TARGET_SAMPLE_RATE
        self._offset_samples += samples.shape[0]

        wav_path = self._work_dir / "live_window.wav"
        try:
            _write_wav(samples, wav_path)
            segments = self._decode(wav_path)
        except Exception:
            logger.exception("Live window decode failed")
            return
        for seg in segments:
            text = collapse_repetitions(seg.text.strip())
            if not text:
                continue
            self._emit({"start": base_s + seg.start, "end": base_s + seg.end, "text": text})

    def _emit(self, event: dict[str, object]) -> None:
        try:
            self._on_event(event)
        except Exception:
            logger.debug("live on_event failed", exc_info=True)
