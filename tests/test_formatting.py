"""Tests for `audicop.formatting`."""

from __future__ import annotations

import pytest

from audicop.formatting import (
    _format_timestamp,
    to_plain_text,
    to_srt,
    to_timestamped_text,
    to_vtt,
)
from audicop.transcriber import TranscriptSegment


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


def test_format_timestamp_srt_uses_comma() -> None:
    assert _format_timestamp(3661.5, with_millis=True, comma=True) == "01:01:01,500"


def test_format_timestamp_vtt_uses_dot() -> None:
    assert _format_timestamp(3661.5, with_millis=True, comma=False) == "01:01:01.500"


# ---------------------------------------------------------------------------
# to_plain_text
# ---------------------------------------------------------------------------
def test_to_plain_text_skips_empty() -> None:
    out = to_plain_text(_segs())
    assert out == "Hola mundo\nesto es una prueba"


def test_to_plain_text_empty_input() -> None:
    assert to_plain_text([]) == ""


# ---------------------------------------------------------------------------
# to_timestamped_text
# ---------------------------------------------------------------------------
def test_to_timestamped_text() -> None:
    out = to_timestamped_text(_segs())
    assert out == "[00:00] Hola mundo\n[00:02] esto es una prueba"


def test_to_timestamped_text_switches_to_hours() -> None:
    segs = [TranscriptSegment(start=3700.0, end=3705.0, text="tarde")]
    assert to_timestamped_text(segs) == "[01:01:40] tarde"


# ---------------------------------------------------------------------------
# to_srt
# ---------------------------------------------------------------------------
def test_to_srt_structure() -> None:
    out = to_srt(_segs())
    assert out.startswith("1\n00:00:00,000 --> 00:00:02,500\nHola mundo")
    assert "2\n00:00:02,500 --> 00:00:05,000\nesto es una prueba" in out
    assert "," in out  # SRT uses comma for millis
    assert out.endswith("\n")


def test_to_srt_empty() -> None:
    assert to_srt([]) == ""


# ---------------------------------------------------------------------------
# to_vtt
# ---------------------------------------------------------------------------
def test_to_vtt_structure() -> None:
    out = to_vtt(_segs())
    assert out.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.500\nHola mundo" in out
    assert "00:00:02.500 --> 00:00:05.000\nesto es una prueba" in out
    assert ".000" in out  # VTT uses dot for millis


def test_to_vtt_empty_has_header() -> None:
    assert to_vtt([]).startswith("WEBVTT")
