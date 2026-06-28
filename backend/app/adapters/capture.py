"""Local audio capture for the "record" modes (own voice + meeting).

Records the microphone and/or the speaker *loopback* (what the other
participants say in a call) into 16 kHz mono WAV files — the exact format
the transcription pipeline already consumes, so recording reuses the rest
of the app untouched.

Backed by the ``soundcard`` library (BSD-3, pip-only via CFFI, no system
deps, no PyTorch). It is imported lazily so the rest of Audicop keeps
working on machines where the audio backend is unavailable (headless CI,
missing drivers): only the record endpoints degrade, with a clear message.

Each source is recorded in its own thread to its own WAV (on Windows the
WASAPI client must be opened in the thread that drains it). For a meeting
we keep the two tracks separate — mic = "you", loopback = "the others" —
which both lets us mix them for transcription and, later, attribute who
spoke without any extra dependency.

PRIVACY: capture only ever runs on explicit user action (the record
buttons) and the audio never leaves the machine; it feeds the same local
pipeline as an uploaded file. See AGENTS.md §3.
"""

from __future__ import annotations

import logging
import threading
import warnings
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core import config

logger = logging.getLogger(__name__)

_START_TIMEOUT_S: float = 4.0
"""How long to wait for a capture thread to open its device before erroring."""

_JOIN_TIMEOUT_S: float = 5.0
"""How long to wait for a capture thread to flush and exit on stop."""

_MIC_TRACK: str = "mic"
_OTHERS_TRACK: str = "others"


class CaptureError(RuntimeError):
    """Raised when audio capture cannot start or fails mid-recording."""


def _soundcard() -> Any:
    """Import and return the ``soundcard`` module, or raise :class:`CaptureError`."""
    try:
        import soundcard
    except Exception as exc:  # ImportError or a backend load failure
        raise CaptureError(
            "No se pudo cargar el sistema de audio para grabar. "
            "Reinstala dependencias con `uv sync` o revisa los drivers de audio."
        ) from exc
    return soundcard


def is_capture_available() -> bool:
    """Return ``True`` if this machine exposes any audio device we can record."""
    try:
        sc = _soundcard()
        return bool(sc.all_speakers()) or bool(sc.all_microphones())
    except Exception:
        logger.debug("Audio capture not available", exc_info=True)
        return False


def _loopback_source(sc: Any) -> Any:
    """Return a soundcard recorder source for the default speaker's loopback."""
    speaker = sc.default_speaker()
    try:
        return sc.get_microphone(str(speaker.name), include_loopback=True)
    except Exception:
        logger.debug("get_microphone loopback failed; scanning all microphones", exc_info=True)
        for mic in sc.all_microphones(include_loopback=True):
            if getattr(mic, "isloopback", False):
                return mic
    raise CaptureError("No se encontró un dispositivo de audio del sistema (loopback).")


def _mic_source(sc: Any) -> Any:
    """Return the default microphone as a soundcard recorder source."""
    mic = sc.default_microphone()
    if mic is None:
        raise CaptureError("No se encontró un micrófono predeterminado.")
    return mic


def _float_block_to_int16_bytes(block: Any) -> bytes:
    """Convert a float32 [-1, 1] capture block to little-endian int16 PCM bytes."""
    arr = np.asarray(block, dtype=np.float32)
    clipped = np.clip(arr, -1.0, 1.0)
    out = (clipped * 32767.0).astype(np.int16)
    return out.tobytes()


def _wav_duration_s(path: Path) -> float:
    """Return the duration in seconds of a PCM WAV file, or ``0.0`` on error."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate() or config.TARGET_SAMPLE_RATE
            return wf.getnframes() / float(rate)
    except (wave.Error, OSError):
        logger.debug("Could not read duration of %s", path, exc_info=True)
        return 0.0


def _mix_wavs(paths: list[Path], out: Path) -> None:
    """Mix several 16 kHz mono int16 WAVs into one, averaging to avoid clipping."""
    tracks: list[Any] = []
    for p in paths:
        with wave.open(str(p), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
        tracks.append(np.frombuffer(frames, dtype=np.int16).astype(np.int32))

    length = min((int(t.shape[0]) for t in tracks), default=0)
    if length == 0:
        raise CaptureError("Las pistas de audio están vacías.")

    mixed = np.zeros(length, dtype=np.int32)
    for t in tracks:
        mixed += t[:length]
    mixed //= len(tracks)
    out_i16 = np.clip(mixed, -32768, 32767).astype(np.int16)

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(config.TARGET_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(config.TARGET_SAMPLE_RATE)
        wf.writeframes(out_i16.tobytes())


@dataclass(frozen=True, slots=True)
class RecordingResult:
    """Outcome of a finished recording.

    Attributes:
        mixed_path: The WAV to transcribe (16 kHz mono — all sources mixed).
        mic_path: The isolated microphone track ("you"), or ``None``.
        others_path: The isolated loopback track ("the others"), or ``None``.
        duration_s: Duration of the recording in seconds.
    """

    mixed_path: Path
    mic_path: Path | None
    others_path: Path | None
    duration_s: float


class _SourceThread(threading.Thread):
    """Records one audio source to a WAV until signalled to stop."""

    def __init__(
        self,
        make_source: Callable[[], Any],
        wav_path: Path,
        stop_event: threading.Event,
        label: str,
    ) -> None:
        """Initialize the capture thread (daemonized so it never blocks exit)."""
        super().__init__(daemon=True, name=f"audicop-capture-{label}")
        # NOTE: not `self._stop` — that name shadows threading.Thread._stop(),
        # which join() calls internally (it would raise "Event not callable").
        self._make_source = make_source
        self._wav_path = wav_path
        self._stop_event = stop_event
        self.label = label
        self.error: str | None = None
        self._ready = threading.Event()

    def run(self) -> None:
        """Open the device in this thread and stream blocks to the WAV file."""
        try:
            source = self._make_source()
            with wave.open(str(self._wav_path), "wb") as wf:
                wf.setnchannels(config.TARGET_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(config.TARGET_SAMPLE_RATE)
                recorder = source.recorder(
                    samplerate=config.TARGET_SAMPLE_RATE,
                    channels=config.TARGET_CHANNELS,
                    blocksize=config.RECORD_BLOCK_FRAMES,
                )
                with recorder as rec, warnings.catch_warnings():
                    # soundcard warns on benign buffer gaps; don't spam logs.
                    warnings.filterwarnings("ignore", message="data discontinuity in recording")
                    self._ready.set()
                    while not self._stop_event.is_set():
                        block = rec.record(numframes=config.RECORD_BLOCK_FRAMES)
                        wf.writeframes(_float_block_to_int16_bytes(block))
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Capture thread %s failed", self.label)
        finally:
            self._ready.set()  # unblock the starter even if we errored early

    def wait_until_ready(self, timeout: float) -> None:
        """Block until the device is open (or the thread errored) or ``timeout``."""
        self._ready.wait(timeout)


class Recorder:
    """Captures one or two audio sources to disk for later transcription.

    Use :meth:`start` to begin and :meth:`stop` to finish; the instance is
    single-use. Recreate it for a new recording.
    """

    def __init__(self, *, include_mic: bool, include_loopback: bool, out_dir: Path) -> None:
        """Configure which sources to capture and where to write them.

        Args:
            include_mic: Capture the default microphone ("you").
            include_loopback: Capture the speaker loopback ("the others").
            out_dir: Directory for the per-source and mixed WAV files.

        Raises:
            CaptureError: If neither source is requested.
        """
        if not include_mic and not include_loopback:
            raise CaptureError("Hay que capturar al menos una fuente de audio.")
        self._include_mic = include_mic
        self._include_loopback = include_loopback
        self._out_dir = out_dir
        self._stop = threading.Event()
        self._threads: list[_SourceThread] = []
        self._mic_path: Path | None = None
        self._others_path: Path | None = None

    def start(self) -> None:
        """Open the devices and begin capturing in background threads.

        Raises:
            CaptureError: If the audio backend or a device cannot be opened.
        """
        sc = _soundcard()
        self._out_dir.mkdir(parents=True, exist_ok=True)

        if self._include_loopback:
            self._others_path = self._out_dir / "others.wav"
            self._threads.append(
                _SourceThread(
                    lambda: _loopback_source(sc), self._others_path, self._stop, _OTHERS_TRACK
                )
            )
        if self._include_mic:
            self._mic_path = self._out_dir / "mic.wav"
            self._threads.append(
                _SourceThread(lambda: _mic_source(sc), self._mic_path, self._stop, _MIC_TRACK)
            )

        for t in self._threads:
            t.start()
        for t in self._threads:
            t.wait_until_ready(_START_TIMEOUT_S)

        failed = next((t for t in self._threads if t.error), None)
        if failed is not None:
            self._stop.set()
            raise CaptureError(f"No se pudo iniciar la captura ({failed.label}): {failed.error}")

    def stop(self) -> RecordingResult:
        """Stop capturing, finalize the WAVs, and return the result.

        Raises:
            CaptureError: If a capture thread errored or produced no audio.
        """
        self._stop.set()
        for t in self._threads:
            t.join(timeout=_JOIN_TIMEOUT_S)

        errored = next((t for t in self._threads if t.error), None)
        if errored is not None:
            raise CaptureError(f"La grabación falló ({errored.label}): {errored.error}")

        paths = [p for p in (self._mic_path, self._others_path) if p is not None and p.exists()]
        if not paths:
            raise CaptureError("La grabación no produjo ningún archivo de audio.")

        if len(paths) == 1:
            mixed = paths[0]
        else:
            mixed = self._out_dir / "mixed.wav"
            _mix_wavs(paths, mixed)

        duration = _wav_duration_s(mixed)
        if duration <= 0:
            raise CaptureError("La grabación quedó vacía. ¿Estaba el micrófono o el audio activo?")

        return RecordingResult(
            mixed_path=mixed,
            mic_path=self._mic_path,
            others_path=self._others_path,
            duration_s=duration,
        )
