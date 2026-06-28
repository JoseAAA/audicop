"""API tests for the recording endpoints (TestClient, mocked capture)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.adapters.capture import RecordingResult
from app.api import record as rec_mod
from app.main import app


@pytest.fixture(autouse=True)
def _reset_active() -> Iterator[None]:
    """Ensure no recording leaks between tests."""
    rec_mod._active = None
    yield
    rec_mod._active = None


def test_meeting_status_reports_detection() -> None:
    with (
        patch.object(rec_mod, "detect_active_meeting", return_value="Zoom"),
        patch.object(rec_mod, "is_capture_available", return_value=True),
        TestClient(app) as client,
    ):
        r = client.get("/api/record/meeting")

    assert r.status_code == 200
    body = r.json()
    assert body["detected"] is True
    assert body["app"] == "Zoom"
    assert body["capture_available"] is True


def test_start_then_stop_returns_path(tmp_path: Path) -> None:
    fake_recorder = MagicMock()
    fake_recorder.stop.return_value = RecordingResult(
        mixed_path=tmp_path / "mixed.wav",
        mic_path=tmp_path / "mic.wav",
        others_path=tmp_path / "others.wav",
        duration_s=12.5,
    )
    with (
        patch.object(rec_mod, "is_capture_available", return_value=True),
        patch.object(rec_mod, "Recorder", return_value=fake_recorder),
        patch.object(rec_mod, "_fresh_recording_dir", return_value=tmp_path),
        TestClient(app) as client,
    ):
        start = client.post("/api/record/start", json={"mode": "meeting", "include_mic": True})
        assert start.status_code == 200
        assert start.json()["recording"] is True
        stop = client.post("/api/record/stop")

    assert stop.status_code == 200
    body = stop.json()
    assert body["duration"] == 12.5
    assert body["filename"] == "mixed.wav"
    assert body["path"].endswith("mixed.wav")
    fake_recorder.start.assert_called_once()
    fake_recorder.stop.assert_called_once()


def test_pause_resume(tmp_path: Path) -> None:
    fake_recorder = MagicMock()
    with (
        patch.object(rec_mod, "is_capture_available", return_value=True),
        patch.object(rec_mod, "Recorder", return_value=fake_recorder),
        patch.object(rec_mod, "_fresh_recording_dir", return_value=tmp_path),
        TestClient(app) as client,
    ):
        client.post("/api/record/start", json={"mode": "voice"})
        assert client.post("/api/record/pause").json() == {"paused": True}
        assert client.post("/api/record/resume").json() == {"paused": False}
    fake_recorder.pause.assert_called_once()
    fake_recorder.resume.assert_called_once()


def test_pause_without_recording_conflicts() -> None:
    with TestClient(app) as client:
        assert client.post("/api/record/pause").status_code == 409
        assert client.post("/api/record/resume").status_code == 409


def test_double_start_conflicts(tmp_path: Path) -> None:
    with (
        patch.object(rec_mod, "is_capture_available", return_value=True),
        patch.object(rec_mod, "Recorder", return_value=MagicMock()),
        patch.object(rec_mod, "_fresh_recording_dir", return_value=tmp_path),
        TestClient(app) as client,
    ):
        first = client.post("/api/record/start", json={"mode": "voice"})
        second = client.post("/api/record/start", json={"mode": "voice"})

    assert first.status_code == 200
    assert second.status_code == 409


def test_stop_without_recording_conflicts() -> None:
    with TestClient(app) as client:
        r = client.post("/api/record/stop")
    assert r.status_code == 409


def test_start_when_capture_unavailable() -> None:
    with (
        patch.object(rec_mod, "is_capture_available", return_value=False),
        TestClient(app) as client,
    ):
        r = client.post("/api/record/start", json={"mode": "voice"})
    assert r.status_code == 400
