"""Server-side text shaping for transcripts.

Pure, side-effect-free helpers over ``list[TranscriptSegment]`` (the
dataclass produced by :mod:`app.adapters.transcriber`): the ``[MM:SS]``
timestamped rendering that feeds the AI context and the meetings store, plus
the anti-hallucination repetition collapser. Plain/SRT/VTT exports are
generated client-side in ``frontend/app.js``.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.adapters.transcriber import TranscriptSegment


def _format_timestamp(seconds: float) -> str:
    """Format seconds as ``MM:SS`` (or ``HH:MM:SS`` past the hour mark).

    Negative values are clamped to ``00:00``.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs = remainder_ms // 1000
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def collapse_repetitions(text: str) -> str:
    """Collapse Whisper's hallucinated repetition loops in a segment.

    Whisper's classic failure on noise/silence is stuttering output —
    ``"gracias por ver gracias por ver gracias por ver"`` — as either one
    word or a short phrase repeated back-to-back. Two conservative passes
    (ported from meetily's cleanup):

    1. Runs of the SAME word 3+ times collapse to one (3+, not 2+, so
       natural doubling like "sí, sí" survives).
    2. A 2-5 word phrase immediately repeated collapses to one occurrence.

    Comparison is case-insensitive; the first occurrence's casing is kept.
    """
    words = text.split()
    if len(words) < 3:
        return text

    # Pass 1 — word runs: 3+ consecutive copies collapse to ONE (hallucination);
    # exactly 2 stay (natural doubling like "sí sí").
    deduped: list[str] = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].lower() == words[i].lower():
            j += 1
        count = j - i
        deduped.extend(words[i : i + (1 if count >= 3 else count)])
        i = j

    for size in range(2, 6):
        out: list[str] = []
        i = 0
        while i < len(deduped):
            window = deduped[i : i + size]
            if len(window) == size:
                phrase = [w.lower() for w in window]
                repeats = 1
                while [
                    w.lower() for w in deduped[i + repeats * size : i + (repeats + 1) * size]
                ] == phrase:
                    repeats += 1
                if repeats > 1:  # phrase loop found: keep one occurrence
                    out.extend(window)
                    i += repeats * size
                    continue
            out.append(deduped[i])
            i += 1
        deduped = out

    return " ".join(deduped)


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
