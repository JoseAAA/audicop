"""Tests for `app.services.formatting`."""

from __future__ import annotations

import pytest

from app.adapters.transcriber import TranscriptSegment
from app.services.formatting import _format_timestamp, to_timestamped_text


def _segs() -> list[TranscriptSegment]:
    """Three segments, one of them empty (should be skipped everywhere)."""
    return [
        TranscriptSegment(start=0.0, end=2.5, text="Hola mundo"),
        TranscriptSegment(start=2.5, end=5.0, text="esto es una prueba"),
        TranscriptSegment(start=5.0, end=6.0, text="   "),  # whitespace only
    ]


# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00"),
        (5.0, "00:05"),
        (65.0, "01:05"),
        (3599.0, "59:59"),
        (3600.0, "01:00:00"),
        (3661.0, "01:01:01"),
    ],
)
def test_format_timestamp_compact(seconds: float, expected: str) -> None:
    assert _format_timestamp(seconds) == expected


def test_format_timestamp_negative_clamped() -> None:
    assert _format_timestamp(-3.0) == "00:00"


# ---------------------------------------------------------------------------
# to_timestamped_text
# ---------------------------------------------------------------------------
def test_to_timestamped_text() -> None:
    out = to_timestamped_text(_segs())
    assert out == "[00:00] Hola mundo\n[00:02] esto es una prueba"


def test_to_timestamped_text_switches_to_hours() -> None:
    segs = [TranscriptSegment(start=3700.0, end=3705.0, text="tarde")]
    assert to_timestamped_text(segs) == "[01:01:40] tarde"
