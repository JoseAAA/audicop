"""Tests for `app.adapters.meeting` (active-meeting detection by mic usage)."""

from __future__ import annotations

from unittest.mock import patch

from app.adapters import meeting


def test_detects_browser_meeting() -> None:
    """A browser using the mic (e.g. Google Meet) is reported as a meeting."""
    with patch.object(meeting, "_apps_using_microphone", return_value=["chrome.exe"]):
        assert meeting.detect_active_meeting() == "una reunión en el navegador"


def test_detects_packaged_teams() -> None:
    """The new (packaged) Teams using the mic is recognized by family name."""
    with patch.object(meeting, "_apps_using_microphone", return_value=["msteams_8wekyb3d8bbwe"]):
        assert meeting.detect_active_meeting() == "Microsoft Teams"


def test_no_meeting_when_mic_idle() -> None:
    with patch.object(meeting, "_apps_using_microphone", return_value=[]):
        assert meeting.detect_active_meeting() is None


def test_ignores_non_meeting_mic_user() -> None:
    """Recording in Audacity uses the mic but is not a meeting."""
    with patch.object(meeting, "_apps_using_microphone", return_value=["audacity.exe"]):
        assert meeting.detect_active_meeting() is None


def test_no_false_positive_from_steam_or_zoomit() -> None:
    """The old substring bug (steamservice→Teams, ZoomIt→Zoom) must not recur."""
    with patch.object(
        meeting, "_apps_using_microphone", return_value=["steamservice.exe", "zoomit.exe"]
    ):
        assert meeting.detect_active_meeting() is None


def test_scan_failure_returns_none() -> None:
    with patch.object(meeting, "_apps_using_microphone", side_effect=RuntimeError("boom")):
        assert meeting.detect_active_meeting() is None


def test_identifier_extracts_last_segment() -> None:
    chrome_key = r"C:#Program Files#Google#Chrome#Application#chrome.exe"
    assert meeting._identifier(chrome_key) == "chrome.exe"
    assert meeting._identifier("MSTeams_8wekyb3d8bbwe") == "msteams_8wekyb3d8bbwe"


def test_apps_using_microphone_does_not_raise() -> None:
    """The real registry scan returns a list (contents depend on the machine)."""
    assert isinstance(meeting._apps_using_microphone(), list)
