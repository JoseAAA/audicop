"""Persistent, local-only library of past meetings (SQLite).

Gives Audicop a Meetily-style history: every finished transcription is saved
as a *meeting* the user can reopen, search, rename and delete. Everything
lives in a single SQLite file **on the user's disk** — nothing ever leaves
the machine — and deleting a meeting removes it permanently.

SQLite ships with Python (zero new dependencies) and a connection-per-call
keeps this safe across the threadpool FastAPI runs sync endpoints in.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core import config
from app.services.transcript_store import StoredSegment


def _db_path() -> Path:
    """Return the SQLite file path, creating its directory if needed."""
    path = Path.home() / config.MEETINGS_DB_SUBPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meetings (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            duration   REAL NOT NULL DEFAULT 0,
            language   TEXT NOT NULL DEFAULT 'unknown',
            segments   TEXT NOT NULL DEFAULT '[]',
            notes      TEXT NOT NULL DEFAULT '',
            condensed  TEXT NOT NULL DEFAULT ''
        )
        """
    )
    # Best-effort migration for databases created before `condensed` existed
    # (the ALTER fails harmlessly once the column is there).
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE meetings ADD COLUMN condensed TEXT NOT NULL DEFAULT ''")
    return conn


@dataclass(frozen=True, slots=True)
class MeetingSummary:
    """A meeting as shown in the sidebar list (no heavy fields)."""

    id: str
    title: str
    created_at: str
    duration: float
    language: str
    has_notes: bool


@dataclass(frozen=True, slots=True)
class Meeting:
    """A full meeting: metadata, transcript segments and saved notes."""

    id: str
    title: str
    created_at: str
    duration: float
    language: str
    segments: tuple[StoredSegment, ...]
    notes: str


def save_meeting(
    *,
    title: str,
    duration: float,
    language: str,
    segments: tuple[StoredSegment, ...],
) -> str:
    """Persist a finished transcription as a new meeting; return its id."""
    meeting_id = uuid.uuid4().hex
    payload = json.dumps(
        [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
        ensure_ascii=False,
    )
    with _connect() as conn:
        conn.execute(
            "INSERT INTO meetings (id, title, created_at, duration, language, segments)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                meeting_id,
                title.strip() or "Reunión sin título",
                datetime.now(timezone.utc).isoformat(),
                float(duration),
                language,
                payload,
            ),
        )
    return meeting_id


def list_meetings(query: str = "") -> list[MeetingSummary]:
    """Return meetings, newest first, optionally filtered by title/content."""
    sql = "SELECT id, title, created_at, duration, language, notes != '' AS has_notes FROM meetings"
    args: tuple[object, ...] = ()
    q = query.strip()
    if q:
        sql += " WHERE title LIKE ? OR segments LIKE ? OR notes LIKE ?"
        like = f"%{q}%"
        args = (like, like, like)
    sql += " ORDER BY created_at DESC"
    with _connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [
        MeetingSummary(
            id=r["id"],
            title=r["title"],
            created_at=r["created_at"],
            duration=r["duration"],
            language=r["language"],
            has_notes=bool(r["has_notes"]),
        )
        for r in rows
    ]


def get_meeting(meeting_id: str) -> Meeting | None:
    """Return one meeting with its full transcript, or ``None``."""
    with _connect() as conn:
        r = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    if r is None:
        return None
    segments = tuple(
        StoredSegment(start=s["start"], end=s["end"], text=s["text"])
        for s in json.loads(r["segments"])
    )
    return Meeting(
        id=r["id"],
        title=r["title"],
        created_at=r["created_at"],
        duration=r["duration"],
        language=r["language"],
        segments=segments,
        notes=r["notes"],
    )


def rename_meeting(meeting_id: str, title: str) -> bool:
    """Update a meeting's title; return ``False`` if it does not exist."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE meetings SET title = ? WHERE id = ?",
            (title.strip() or "Reunión sin título", meeting_id),
        )
    return cur.rowcount > 0


def get_condensed(meeting_id: str) -> str:
    """Return the cached map-reduce notes for a long meeting, or ``""``.

    Condensing a multi-hour transcript takes minutes of LLM work; it depends
    only on the transcript, so it is computed once and reused for every
    question afterwards (surviving app restarts).
    """
    with _connect() as conn:
        row = conn.execute("SELECT condensed FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    return str(row["condensed"]) if row is not None else ""


def save_condensed(meeting_id: str, condensed: str) -> bool:
    """Persist the condensed notes; return ``False`` if the meeting is gone."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE meetings SET condensed = ? WHERE id = ?", (condensed, meeting_id)
        )
    return cur.rowcount > 0


def save_notes(meeting_id: str, notes: str) -> bool:
    """Store the AI notes for a meeting; return ``False`` if it does not exist."""
    with _connect() as conn:
        cur = conn.execute("UPDATE meetings SET notes = ? WHERE id = ?", (notes, meeting_id))
    return cur.rowcount > 0


def _audio_dir() -> Path:
    """Return the directory holding each meeting's listening copy."""
    return Path.home() / config.MEETINGS_AUDIO_SUBPATH


def audio_target(meeting_id: str) -> Path:
    """Return the preferred (compressed) audio path for a meeting."""
    return _audio_dir() / f"{meeting_id}.m4a"


def audio_path(meeting_id: str) -> Path | None:
    """Return the stored audio file for a meeting, or ``None`` if absent.

    Checks the compressed copy first, then the raw-WAV fallback that
    :func:`app.adapters.audio.save_compressed_copy` writes when ffmpeg
    cannot encode.
    """
    for suffix in (".m4a", ".wav"):
        candidate = _audio_dir() / f"{meeting_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def delete_meeting(meeting_id: str) -> bool:
    """Delete a meeting (and its audio copy) permanently."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    for suffix in (".m4a", ".wav"):
        (_audio_dir() / f"{meeting_id}{suffix}").unlink(missing_ok=True)
    return cur.rowcount > 0
