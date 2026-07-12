"""Meetings library endpoints: list, search, open, rename, notes, delete.

All data lives in local files on the user's machine (SQLite + per-meeting
audio copy — see :mod:`app.services.meeting_store`); these endpoints never
touch the network.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.adapters.local_llm import ChatMessage, LLMError
from app.api.chat import get_ready_model
from app.services import meeting_store
from app.services.formatting import to_timestamped_text

router = APIRouter(prefix="/api", tags=["meetings"])

_TITLE_CONTEXT_CHARS = 3000
"""How much transcript the title generator sees (the opening sets the topic)."""


class _TitleBody(BaseModel):
    title: str


class _NotesBody(BaseModel):
    notes: str


@router.get("/meetings")
def list_meetings(q: str = "") -> dict[str, object]:
    """List meetings (newest first), optionally filtered by `q`."""
    return {"meetings": [asdict(m) for m in meeting_store.list_meetings(q)]}


@router.get("/meetings/{meeting_id}")
def get_meeting(meeting_id: str) -> dict[str, object]:
    """Return one meeting with its full transcript, notes and audio flag."""
    meeting = meeting_store.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Reunión no encontrada.")
    return asdict(meeting) | {"has_audio": meeting_store.audio_path(meeting_id) is not None}


@router.get("/meetings/{meeting_id}/audio")
def get_meeting_audio(meeting_id: str) -> FileResponse:
    """Serve the meeting's local listening copy (m4a, or WAV fallback)."""
    path = meeting_store.audio_path(meeting_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Esta reunión no tiene audio guardado.")
    media_type = "audio/mp4" if path.suffix == ".m4a" else "audio/wav"
    return FileResponse(path, media_type=media_type)


@router.post("/meetings/{meeting_id}/autotitle")
def autotitle_meeting(meeting_id: str) -> dict[str, object]:
    """Generate a short title from the transcript with the local model.

    Same behavior as chat apps that name conversations automatically. Falls
    back with 503 when no local model fits the machine — the meeting keeps
    its filename-based title, which is always valid.
    """
    meeting = meeting_store.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="Reunión no encontrada.")
    excerpt = to_timestamped_text(meeting.segments)[:_TITLE_CONTEXT_CHARS]
    if not excerpt.strip():
        raise HTTPException(status_code=400, detail="La reunión no tiene transcripción.")
    try:
        model = get_ready_model()
        raw = "".join(
            model.stream_chat(
                system="Eres un asistente que titula transcripciones.",
                messages=[
                    ChatMessage(
                        role="user",
                        content=(
                            "Dame un título corto (máximo 6 palabras, sin comillas, sin "
                            "punto final) que describa el tema de esta transcripción. "
                            "Responde SOLO el título, en el idioma de la transcripción.\n\n"
                            f"{excerpt}"
                        ),
                    )
                ],
            )
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    title = " ".join(raw.strip().strip('"“”').splitlines()[0].split())[:80]
    if not title:
        raise HTTPException(status_code=502, detail="El modelo no devolvió un título.")
    meeting_store.rename_meeting(meeting_id, title)
    return {"title": title}


@router.patch("/meetings/{meeting_id}")
def rename_meeting(meeting_id: str, body: _TitleBody) -> dict[str, object]:
    """Rename a meeting."""
    if not meeting_store.rename_meeting(meeting_id, body.title):
        raise HTTPException(status_code=404, detail="Reunión no encontrada.")
    return {"ok": True}


@router.put("/meetings/{meeting_id}/notes")
def save_notes(meeting_id: str, body: _NotesBody) -> dict[str, object]:
    """Save the AI notes of a meeting."""
    if not meeting_store.save_notes(meeting_id, body.notes):
        raise HTTPException(status_code=404, detail="Reunión no encontrada.")
    return {"ok": True}


@router.delete("/meetings/{meeting_id}")
def delete_meeting(meeting_id: str) -> dict[str, object]:
    """Delete a meeting permanently from the local library."""
    if not meeting_store.delete_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="Reunión no encontrada.")
    return {"ok": True}
