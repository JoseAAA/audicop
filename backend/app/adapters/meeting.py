"""Detect whether the user is *actively in a meeting/call* (Windows).

Checking whether a conferencing app is installed or running gives false
positives — Teams idles in the tray, Steam's service name even contains
"teams", and browser meetings (Google Meet) are not processes at all. So
instead we ask Windows which application is **using the microphone right
now**: that is the reliable "you're in a call" signal. It catches Meet
(chrome.exe), Teams, Zoom, etc., and ignores background apps.

We read the per-user CapabilityAccessManager consent store via the stdlib
``winreg`` (no extra dependency). On non-Windows we return ``None`` and the
user records manually. The function never raises.
"""

from __future__ import annotations

import logging
import sys

from app.core import config

logger = logging.getLogger(__name__)

# Per-user store Windows updates whenever an app starts/stops using the mic.
# Inside it, an entry whose ``LastUsedTimeStop`` is 0 is using the mic *now*.
_MIC_CONSENT_BASE = (
    r"Software\Microsoft\Windows\CurrentVersion"
    r"\CapabilityAccessManager\ConsentStore\microphone"
)
_IN_USE_SENTINEL = 0


def _identifier(subkey: str) -> str:
    r"""Reduce a consent-store subkey to a comparable app id (lowercased).

    NonPackaged subkeys are executable paths with ``#`` replacing ``\``
    (e.g. ``C:#...#chrome.exe``); packaged subkeys are package family names
    (e.g. ``MSTeams_8wekyb3d8bbwe``). We keep the last path segment.
    """
    return subkey.rsplit("#", 1)[-1].lower()


def _apps_using_microphone() -> list[str]:
    """Return identifiers of apps currently using the microphone (Windows only)."""
    if sys.platform != "win32":
        return []
    import winreg

    active: list[str] = []
    for suffix in ("", r"\NonPackaged"):
        try:
            root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _MIC_CONSENT_BASE + suffix)
        except OSError:
            continue
        try:
            index = 0
            while True:
                try:
                    name = winreg.EnumKey(root, index)
                except OSError:
                    break  # no more subkeys
                index += 1
                if name.lower() == "nonpackaged":
                    continue  # handled by the explicit \NonPackaged pass
                try:
                    with winreg.OpenKey(root, name) as entry:
                        stop, _ = winreg.QueryValueEx(entry, "LastUsedTimeStop")
                except OSError:
                    continue  # entry without the expected value
                if stop == _IN_USE_SENTINEL:
                    active.append(_identifier(name))
        finally:
            winreg.CloseKey(root)
    return active


def detect_active_meeting() -> str | None:
    """Return a friendly app name if the user seems to be in a meeting, else ``None``.

    Inferred from a known conferencing/browser app using the microphone
    right now (see :data:`app.core.config.MEETING_APP_HINTS`). Never raises;
    any failure degrades to ``None`` (the user can still record manually).
    """
    try:
        active = _apps_using_microphone()
    except Exception:  # pragma: no cover - defensive against registry quirks
        logger.debug("Microphone-usage scan failed", exc_info=True)
        return None
    for ident in active:
        for needle, friendly in config.MEETING_APP_HINTS.items():
            if needle in ident:
                return friendly
    return None
