"""Tests for `app.services.meeting_store` and the /api/meetings endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import meeting_store
from app.services.transcript_store import StoredSegment

SEGS = (
    StoredSegment(start=0.0, end=2.0, text="Hola equipo"),
    StoredSegment(start=2.0, end=5.0, text="Revisemos el presupuesto"),
)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the store (DB + audio dir) at throwaway paths for every test."""
    monkeypatch.setattr(meeting_store, "_db_path", lambda: tmp_path / "meetings.db")
    monkeypatch.setattr(meeting_store, "_audio_dir", lambda: tmp_path / "audio")


def _save(title: str = "Reunión de ventas") -> str:
    return meeting_store.save_meeting(title=title, duration=5.0, language="es", segments=SEGS)


def test_save_and_get_roundtrip() -> None:
    mid = _save()
    m = meeting_store.get_meeting(mid)
    assert m is not None
    assert m.title == "Reunión de ventas"
    assert m.language == "es"
    assert m.segments == SEGS
    assert m.notes == ""


def test_list_newest_first_and_search() -> None:
    _save("Planificación sprint")
    _save("Llamada con cliente")
    all_meetings = meeting_store.list_meetings()
    assert [m.title for m in all_meetings] == ["Llamada con cliente", "Planificación sprint"]
    # search by title
    assert [m.title for m in meeting_store.list_meetings("sprint")] == ["Planificación sprint"]
    # search by transcript content
    assert len(meeting_store.list_meetings("presupuesto")) == 2
    assert meeting_store.list_meetings("no-existe-xyz") == []


def test_rename_notes_delete() -> None:
    mid = _save()
    assert meeting_store.rename_meeting(mid, "Nuevo título") is True
    assert meeting_store.save_notes(mid, "[00:00] Nota guardada") is True
    m = meeting_store.get_meeting(mid)
    assert m is not None and m.title == "Nuevo título" and "Nota" in m.notes
    assert meeting_store.list_meetings()[0].has_notes is True
    assert meeting_store.delete_meeting(mid) is True
    assert meeting_store.get_meeting(mid) is None
    assert meeting_store.delete_meeting(mid) is False  # already gone


def test_empty_title_gets_default() -> None:
    mid = _save("   ")
    m = meeting_store.get_meeting(mid)
    assert m is not None and m.title == "Reunión sin título"


def test_audio_path_prefers_m4a_and_delete_removes_it() -> None:
    mid = _save()
    assert meeting_store.audio_path(mid) is None  # nothing stored yet
    target = meeting_store.audio_target(mid)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"aac")
    assert meeting_store.audio_path(mid) == target
    assert meeting_store.delete_meeting(mid) is True
    assert meeting_store.audio_path(mid) is None  # audio deleted with the meeting


def test_condensed_notes_roundtrip_and_migration(tmp_path: Path) -> None:
    """Condensed notes persist per meeting; old DBs get the column added."""
    import sqlite3

    # Simulate a pre-`condensed` database (created before the column existed).
    old_db = tmp_path / "meetings.db"
    with sqlite3.connect(old_db) as conn:
        conn.execute(
            "CREATE TABLE meetings (id TEXT PRIMARY KEY, title TEXT NOT NULL,"
            " created_at TEXT NOT NULL, duration REAL NOT NULL DEFAULT 0,"
            " language TEXT NOT NULL DEFAULT 'unknown',"
            " segments TEXT NOT NULL DEFAULT '[]', notes TEXT NOT NULL DEFAULT '')"
        )
    mid = _save()  # _connect() auto-migrates on first touch
    assert meeting_store.get_condensed(mid) == ""
    assert meeting_store.save_condensed(mid, "[00:00] notas condensadas") is True
    assert meeting_store.get_condensed(mid) == "[00:00] notas condensadas"
    assert meeting_store.get_condensed("nope") == ""


def test_audio_path_wav_fallback() -> None:
    mid = _save()
    wav = meeting_store.audio_target(mid).with_suffix(".wav")
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"wav")
    assert meeting_store.audio_path(mid) == wav


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
def test_api_meetings_crud() -> None:
    mid = _save("Demo con cliente")
    with TestClient(app) as client:
        listed = client.get("/api/meetings").json()["meetings"]
        assert listed[0]["title"] == "Demo con cliente"

        found = client.get("/api/meetings", params={"q": "cliente"}).json()["meetings"]
        assert len(found) == 1

        one = client.get(f"/api/meetings/{mid}").json()
        assert one["segments"][0]["text"] == "Hola equipo"

        assert client.patch(f"/api/meetings/{mid}", json={"title": "Demo v2"}).status_code == 200
        assert (
            client.put(f"/api/meetings/{mid}/notes", json={"notes": "resumen"}).status_code == 200
        )
        assert client.get(f"/api/meetings/{mid}").json()["title"] == "Demo v2"

        assert client.delete(f"/api/meetings/{mid}").status_code == 200
        assert client.get(f"/api/meetings/{mid}").status_code == 404


def test_api_meeting_not_found() -> None:
    with TestClient(app) as client:
        assert client.get("/api/meetings/nope").status_code == 404
        assert client.patch("/api/meetings/nope", json={"title": "x"}).status_code == 404
        assert client.delete("/api/meetings/nope").status_code == 404
        assert client.get("/api/meetings/nope/audio").status_code == 404


def test_api_meeting_audio_roundtrip() -> None:
    mid = _save()
    with TestClient(app) as client:
        assert client.get(f"/api/meetings/{mid}/audio").status_code == 404
        assert client.get(f"/api/meetings/{mid}").json()["has_audio"] is False
        target = meeting_store.audio_target(mid)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-aac")
        r = client.get(f"/api/meetings/{mid}/audio")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/mp4")
        assert client.get(f"/api/meetings/{mid}").json()["has_audio"] is True


def test_api_autotitle_renames_from_transcript() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch as mock_patch

    from app.api import meetings as meetings_mod

    mid = _save("clip")
    fake_model = SimpleNamespace(stream_chat=lambda **_k: iter(['"Revisión de presupuesto"\n']))
    with (
        mock_patch.object(meetings_mod, "get_ready_model", return_value=fake_model),
        TestClient(app) as client,
    ):
        r = client.post(f"/api/meetings/{mid}/autotitle")
        assert r.status_code == 200
        assert r.json()["title"] == "Revisión de presupuesto"  # quotes stripped
    stored = meeting_store.get_meeting(mid)
    assert stored is not None and stored.title == "Revisión de presupuesto"


def test_api_autotitle_unavailable_model_is_503() -> None:
    from unittest.mock import patch as mock_patch

    from app.adapters.local_llm import LLMError
    from app.api import meetings as meetings_mod

    mid = _save()
    with (
        mock_patch.object(meetings_mod, "get_ready_model", side_effect=LLMError("sin modelo")),
        TestClient(app) as client,
    ):
        assert client.post(f"/api/meetings/{mid}/autotitle").status_code == 503
