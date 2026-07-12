"""Recording endpoints: detect a meeting, start/stop local audio capture.

Single-user, single active recording: the app captures the microphone
and/or speaker loopback to a temp WAV and, on stop, returns its path so
the browser can feed it into the existing ``/api/transcribe`` flow. While
recording, a live tap decodes ~6 s windows and streams draft segments over
``GET /api/record/live`` (SSE). The audio never leaves the machine.

PRIVACY: capture runs only on these explicit actions; see AGENTS.md §3.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.adapters import local_llm
from app.adapters.capture import CaptureError, Recorder, is_capture_available
from app.adapters.hardware import detect_hardware
from app.adapters.meeting import detect_active_meeting
from app.adapters.transcriber import TranscriptSegment, is_model_cached
from app.core import config
from app.services.live import LiveTranscriber
from app.services.pipeline import get_transcriber
from app.services.recommender import recommend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/record", tags=["record"])

# Single-user local app: at most one recording at a time, guarded by a lock.
_lock = threading.Lock()
_active: Recorder | None = None
_live: LiveTranscriber | None = None
_live_queue: queue.Queue[dict[str, object]] | None = None


def _live_decode(wav_path: Path) -> Sequence[TranscriptSegment]:
    """Decode one live window with the recommended Whisper model.

    Reuses the pipeline's transcriber cache, so the model loads once at the
    first window and every later decode is warm. The LLM is released first —
    the sequential memory choreography applies during recording too.
    """
    local_llm.unload_active()
    choice = recommend(detect_hardware())
    transcriber = get_transcriber(choice.model_size, choice.compute_type, choice.device)
    beam = config.DEFAULT_BEAM_SIZE if choice.device == "cuda" else config.BEAM_SIZE_CPU
    segments, _info = transcriber.transcribe(wav_path, beam_size=beam, vad_filter=True)
    return list(segments)


def _start_live(recorder_tracks: Sequence[str], work_dir: Path) -> LiveTranscriber | None:
    """Create and start the live transcriber, or ``None`` when not viable.

    Live preview needs the Whisper model already on disk: downloading
    gigabytes mid-recording would stall the preview for its whole duration,
    so the first-ever recording simply runs without it.
    """
    global _live_queue
    choice = recommend(detect_hardware())
    if not is_model_cached(choice.model_size):
        _live_queue = None
        return None
    _live_queue = queue.Queue()
    live = LiveTranscriber(
        tracks=recorder_tracks,
        work_dir=work_dir,
        decode=_live_decode,
        on_event=_live_queue.put,
    )
    live.start()
    return live


def _recordings_root() -> Path:
    return Path(tempfile.gettempdir()) / config.RECORDINGS_DIR_NAME


def _fresh_recording_dir() -> Path:
    """Clean previous recordings and return a new empty directory for this one."""
    root = _recordings_root()
    if root.exists():
        for child in root.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not clean old recording %s", child, exc_info=True)
    out = root / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=True)
    return out


@router.get("/meeting")
def meeting_status() -> dict[str, object]:
    """Report whether a known meeting app is running and if capture is available."""
    app_name = detect_active_meeting()
    return {
        "detected": app_name is not None,
        "app": app_name,
        "capture_available": is_capture_available(),
    }


class StartRequest(BaseModel):
    """Body for POST /api/record/start."""

    mode: Literal["voice", "meeting"]
    include_mic: bool = True


@router.post("/start")
def start_recording(req: StartRequest) -> dict[str, object]:
    """Begin capturing audio for the chosen mode (with live preview if viable)."""
    global _active, _live
    with _lock:
        if _active is not None:
            raise HTTPException(status_code=409, detail="Ya hay una grabación en curso.")
        if not is_capture_available():
            raise HTTPException(
                status_code=400,
                detail="Este equipo no tiene captura de audio disponible.",
            )
        # Voice mode is always the mic; meeting mode adds the speaker loopback.
        include_loopback = req.mode == "meeting"
        include_mic = True if req.mode == "voice" else req.include_mic
        out_dir = _fresh_recording_dir()
        try:
            recorder = Recorder(
                include_mic=include_mic,
                include_loopback=include_loopback,
                out_dir=out_dir,
                live_sink=lambda track, block: _live.feed(track, block) if _live else None,
            )
            _live = _start_live(recorder.track_labels, out_dir)
            recorder.start()
        except CaptureError as exc:
            if _live is not None:
                _live.stop()
                _live = None
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _active = recorder
    return {"recording": True, "mode": req.mode, "live": _live is not None}


@router.post("/pause")
def pause_recording() -> dict[str, object]:
    """Pause the active recording (the paused span is dropped, not recorded)."""
    with _lock:
        if _active is None:
            raise HTTPException(status_code=409, detail="No hay ninguna grabación en curso.")
        _active.pause()
    return {"paused": True}


@router.post("/resume")
def resume_recording() -> dict[str, object]:
    """Resume a paused recording."""
    with _lock:
        if _active is None:
            raise HTTPException(status_code=409, detail="No hay ninguna grabación en curso.")
        _active.resume()
    return {"paused": False}


@router.post("/stop")
def stop_recording() -> dict[str, object]:
    """Stop the active recording and return the WAV path to transcribe."""
    global _active, _live
    with _lock:
        recorder = _active
        live = _live
        _active = None
        _live = None
    if recorder is None:
        raise HTTPException(status_code=409, detail="No hay ninguna grabación en curso.")
    if live is not None:
        live.stop()  # flush the tail + emit the done event to the SSE stream
    try:
        result = recorder.stop()
    except CaptureError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "path": str(result.mixed_path),
        "filename": result.mixed_path.name,
        "duration": result.duration_s,
    }


@router.get("/live")
def live_events() -> StreamingResponse:
    """Stream the live draft transcript of the active recording (SSE).

    Ends with a ``{"done": true}`` event when the recording stops (or right
    away when live preview is unavailable, e.g. first run without the model).
    """
    q = _live_queue
    if q is None:
        raise HTTPException(
            status_code=404,
            detail="No hay transcripción en vivo (primera grabación o modelo no descargado).",
        )

    def event_stream() -> Iterator[str]:
        while True:
            event = q.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("done"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
