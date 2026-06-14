"""Transcription endpoints: start a job (upload or local path) + stream events.

The decode is blocking and yields segments lazily (see
:func:`app.services.pipeline.iter_transcription`). We run it in a thread
and bridge each event into an :class:`asyncio.Queue`; the SSE endpoint
drains that queue to the browser. Jobs live in an in-memory registry —
fine for a single-user local app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.adapters.audio import cleanup
from app.adapters.transcriber import is_model_cached
from app.core import config
from app.services.pipeline import TranscriptionSettings, iter_transcription

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["transcribe"])

_DONE = object()  # sentinel: no more events for a job


@dataclass
class _Job:
    """One transcription job and its event channel."""

    source_path: Path
    settings: TranscriptionSettings
    is_temp: bool
    queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None


_JOBS: dict[str, _Job] = {}


def _validate_extension(name: str) -> None:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Extensión .{ext} no soportada. "
                f"Soportadas: {', '.join(config.SUPPORTED_EXTENSIONS)}."
            ),
        )


def _save_upload(upload: UploadFile) -> Path:
    """Stream an upload to a temp dir (no full read into RAM) and return the path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=config.TEMP_PREFIX))
    safe_name = Path(upload.filename or "audio").name
    dest = tmp_dir / safe_name
    with dest.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            out.write(chunk)
    return dest


def _resolve_local_path(path_str: str) -> Path:
    cleaned = path_str.strip().strip('"').strip("'")
    p = Path(cleaned).expanduser()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=400, detail=f"No se encontró el archivo: {p}")
    _validate_extension(p.name)
    return p.resolve()


async def _run_job(job_id: str, job: _Job) -> None:
    """Drive the blocking pipeline in a thread, pushing events to the queue."""
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            for event in iter_transcription(job.source_path, job.settings):
                loop.call_soon_threadsafe(job.queue.put_nowait, event)
        except FileNotFoundError as exc:
            loop.call_soon_threadsafe(
                job.queue.put_nowait, {"type": "error", "message": f"Archivo no encontrado: {exc}"}
            )
        except Exception as exc:  # AudioConversionError, TranscriptionError, etc.
            logger.exception("Transcription job %s failed", job_id)
            loop.call_soon_threadsafe(job.queue.put_nowait, {"type": "error", "message": str(exc)})
        finally:
            if job.is_temp:
                cleanup(job.source_path.parent)
            loop.call_soon_threadsafe(job.queue.put_nowait, _DONE)

    await loop.run_in_executor(None, worker)


@router.post("/transcribe")
async def start_transcription(
    file: UploadFile | None = File(default=None),
    path: str | None = Form(default=None),
    language: str = Form(default="auto"),
    task: str = Form(default="transcribe"),
    vad_filter: bool = Form(default=True),
    model_size: str = Form(...),
    compute_type: str = Form(...),
    device: str = Form(...),
) -> dict[str, object]:
    """Start a transcription job from an upload or a local path."""
    if file is not None and file.filename:
        _validate_extension(file.filename)
        source_path = _save_upload(file)
        is_temp = True
    elif path:
        source_path = _resolve_local_path(path)
        is_temp = False
    else:
        raise HTTPException(status_code=400, detail="Falta el archivo o la ruta local.")

    settings = TranscriptionSettings(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        language=None if language == "auto" else language,
        task=task,
        vad_filter=vad_filter,
    )

    job_id = uuid.uuid4().hex
    job = _Job(source_path=source_path, settings=settings, is_temp=is_temp)
    _JOBS[job_id] = job
    job.task = asyncio.create_task(_run_job(job_id, job))

    return {
        "job_id": job_id,
        "model_cached": is_model_cached(model_size),
        "filename": source_path.name,
    }


@router.get("/transcribe/{job_id}/events")
async def transcription_events(job_id: str) -> StreamingResponse:
    """Stream a job's events to the browser as Server-Sent Events."""
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado o ya finalizado.")

    async def event_stream() -> Any:
        try:
            while True:
                event = await job.queue.get()
                if event is _DONE:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            _JOBS.pop(job_id, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
