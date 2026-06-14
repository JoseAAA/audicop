"""Render decoded segments into the output formats Audicop offers.

Pure, side-effect-free functions over ``list[TranscriptSegment]`` (the
dataclass produced by :mod:`audicop.transcriber`). Keeping them here makes
them trivially unit-testable and reusable by both the UI download buttons
and the AI context builder.

Formats:
    - Plain text: just the spoken words, newline per segment.
    - Timestamped text: ``[MM:SS] words`` — YouTube-style.
    - SRT: SubRip subtitles (``HH:MM:SS,mmm``).
    - VTT: WebVTT subtitles (``HH:MM:SS.mmm``).
"""

from __future__ import annotations

from collections.abc import Sequence

from audicop.transcriber import TranscriptSegment


def _format_timestamp(seconds: float, *, with_millis: bool = False, comma: bool = False) -> str:
    """Format a number of seconds as a human/subtitle timestamp.

    Args:
        seconds: Offset in seconds (negative values are clamped to 0).
        with_millis: If ``True``, include milliseconds (subtitle formats).
        comma: If ``True``, use ``,`` as the millisecond separator (SRT);
            otherwise ``.`` (VTT). Ignored when ``with_millis`` is ``False``.

    Returns:
        ``MM:SS`` (compact) or ``HH:MM:SS`` / ``HH:MM:SS,mmm`` / ``HH:MM:SS.mmm``.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1000)

    if with_millis:
        sep = "," if comma else "."
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def to_plain_text(segments: Sequence[TranscriptSegment]) -> str:
    """Return the spoken text only, one segment per line.

    Empty segments (no decoded words) are skipped.
    """
    lines = [seg.text.strip() for seg in segments if seg.text.strip()]
    return "\n".join(lines)


def to_timestamped_text(segments: Sequence[TranscriptSegment]) -> str:
    """Return ``[MM:SS] text`` lines, YouTube-transcript style.

    Empty segments are skipped. Timestamps switch to ``[HH:MM:SS]``
    automatically once the audio passes the one-hour mark.
    """
    lines: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append(f"[{_format_timestamp(seg.start)}] {text}")
    return "\n".join(lines)


def to_srt(segments: Sequence[TranscriptSegment]) -> str:
    """Return the segments as an SRT (SubRip) subtitle document."""
    blocks: list[str] = []
    index = 1
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        start = _format_timestamp(seg.start, with_millis=True, comma=True)
        end = _format_timestamp(seg.end, with_millis=True, comma=True)
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
        index += 1
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(segments: Sequence[TranscriptSegment]) -> str:
    """Return the segments as a WebVTT subtitle document."""
    blocks: list[str] = ["WEBVTT", ""]
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        start = _format_timestamp(seg.start, with_millis=True, comma=False)
        end = _format_timestamp(seg.end, with_millis=True, comma=False)
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)
