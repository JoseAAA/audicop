"""Detect whether a known video-conferencing app is currently running.

This powers the "record meeting" convenience: when an app like Microsoft
Teams or Zoom is detected, the UI can proactively offer one-click
recording. Detection is a read-only scan of process names via ``psutil``
(already a dependency) — it never inspects window titles, audio, or
network, and never raises.

Browser-based meetings (Google Meet in a tab) are deliberately not
detected: a Meet tab is indistinguishable from any other browser tab by
process name, so for those the user starts recording manually.
"""

from __future__ import annotations

import logging

import psutil

from app.core import config

logger = logging.getLogger(__name__)


def detect_active_meeting() -> str | None:
    """Return the friendly name of a running conferencing app, or ``None``.

    Scans running process names for the substrings in
    :data:`app.core.config.MEETING_APP_PROCESSES`. The first match wins.

    Returns:
        A user-facing app name (e.g. ``"Microsoft Teams"``) if a known
        meeting app is running, otherwise ``None``. Never raises: any
        failure scanning processes is logged and treated as "not detected".
    """
    try:
        running = {
            (proc.info.get("name") or "").lower()
            for proc in psutil.process_iter(["name"])
            if proc.info.get("name")
        }
    except Exception:  # pragma: no cover - defensive; psutil rarely fails wholesale
        logger.debug("Process scan failed during meeting detection", exc_info=True)
        return None

    for needle, friendly in config.MEETING_APP_PROCESSES.items():
        if any(needle in name for name in running):
            return friendly
    return None
