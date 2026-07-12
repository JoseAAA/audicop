"""In-memory store of the last finished transcript (single-user local app).

Audicop's phases are sequential: Whisper transcribes and is unloaded, and the
local LLM analyzes *afterwards*. This store is the bridge between them — the
transcript survives on the server independently of the browser tab, so:

- the analysis phase can always retrieve it (even if the tab reloaded and the
  request arrives without a transcript), and
- a reloaded page can restore the last result instead of losing the work.

Held in **memory only**, never written to disk: transcripts of private
meetings should not silently accumulate in files. Restarting the server
clears it, which is the honest trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StoredSegment:
    """One stored transcript segment (mirrors the SSE ``segment`` event)."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class StoredTranscript:
    """The last finished transcription, ready for the analysis phase.

    Attributes:
        segments: Decoded segments in order.
        timestamped: The transcript rendered as ``[MM:SS] text`` lines.
        language: Detected (or forced) language code.
        duration: Audio duration in seconds.
        filename: Name of the source file (for the UI restore message).
    """

    segments: tuple[StoredSegment, ...] = field(default_factory=tuple)
    timestamped: str = ""
    language: str = "unknown"
    duration: float = 0.0
    filename: str = ""


_LAST: StoredTranscript | None = None


def save(transcript: StoredTranscript) -> None:
    """Replace the stored transcript with a newly finished one."""
    global _LAST
    _LAST = transcript


def get() -> StoredTranscript | None:
    """Return the last finished transcript, or ``None`` if there is none."""
    return _LAST


def clear() -> None:
    """Forget the stored transcript (used by tests)."""
    global _LAST
    _LAST = None
