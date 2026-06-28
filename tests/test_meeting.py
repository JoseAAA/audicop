"""Tests for `app.adapters.meeting` (conferencing-app detection)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from app.adapters import meeting


def _procs(*names: str) -> list[SimpleNamespace]:
    """Build fake psutil process objects exposing an ``info`` dict."""
    return [SimpleNamespace(info={"name": n}) for n in names]


def test_detects_known_meeting_app() -> None:
    """A running Teams process is reported by its friendly name."""
    with patch.object(
        meeting.psutil, "process_iter", return_value=_procs("ms-teams.exe", "chrome.exe")
    ):
        assert meeting.detect_active_meeting() == "Microsoft Teams"


def test_detects_zoom() -> None:
    with patch.object(meeting.psutil, "process_iter", return_value=_procs("Zoom.exe")):
        assert meeting.detect_active_meeting() == "Zoom"


def test_no_meeting_returns_none() -> None:
    """Only unrelated processes → no meeting detected."""
    with patch.object(
        meeting.psutil, "process_iter", return_value=_procs("explorer.exe", "code.exe")
    ):
        assert meeting.detect_active_meeting() is None


def test_scan_failure_returns_none() -> None:
    """If psutil blows up wholesale, detection degrades to None (never raises)."""
    with patch.object(meeting.psutil, "process_iter", side_effect=RuntimeError("boom")):
        assert meeting.detect_active_meeting() is None


def test_handles_processes_with_missing_name() -> None:
    """Processes with an empty/None name are skipped, not crashed on."""
    procs = [SimpleNamespace(info={"name": None}), SimpleNamespace(info={"name": "zoom.exe"})]
    with patch.object(meeting.psutil, "process_iter", return_value=procs):
        assert meeting.detect_active_meeting() == "Zoom"
