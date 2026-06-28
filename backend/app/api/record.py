"""Recording endpoints: detect a meeting, start/stop local audio capture.

Single-user, single active recording: the app captures the microphone
and/or speaker loopback to a temp WAV and, on stop, returns its path so
the browser can feed it into the existing ``/api/transcribe`` flow. The
audio never leaves the machine.

PRIVACY: capture runs only on these explicit actions; see AGENTS.md §3.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.adapters.capture import CaptureError, Recorder, is_capture_available
from app.adapters.meeting import detect_active_meeting
from app.core import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/record", tags=["record"])

# Single-user local app: at most one recording at a time, guarded by a lock.
_lock = threading.Lock()
_active: Recorder | None = None


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
    """Begin capturing audio for the chosen mode."""
    global _active
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
        try:
            recorder = Recorder(
                include_mic=include_mic,
                include_loopback=include_loopback,
                out_dir=_fresh_recording_dir(),
            )
            recorder.start()
        except CaptureError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _active = recorder
    return {"recording": True, "mode": req.mode}


@router.post("/stop")
def stop_recording() -> dict[str, object]:
    """Stop the active recording and return the WAV path to transcribe."""
    global _active
    with _lock:
        recorder = _active
        _active = None
    if recorder is None:
        raise HTTPException(status_code=409, detail="No hay ninguna grabación en curso.")
    try:
        result = recorder.stop()
    except CaptureError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "path": str(result.mixed_path),
        "filename": result.mixed_path.name,
        "duration": result.duration_s,
    }
