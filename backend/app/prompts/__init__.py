"""Prompt templates for the AI analysis/chat over a transcript.

The actual prompt text lives in editable Markdown files next to this
module (``system.md``, ``context.md`` and ``actions/*.md``), so prompts
can be tuned without touching Python. This package loads them at import
time and exposes the same public API the rest of the app expects:
:data:`SYSTEM_PROMPT`, :data:`QUICK_ACTIONS` and :func:`build_context`.

All user-facing prompt text is Spanish; the assistant answers in the
user's language. The system prompt frames the model as an audio-analysis
assistant working from a timestamped transcript so it can cite the exact
moment (``[MM:SS]``) where something was said.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _read(*parts: str) -> str:
    """Read a prompt file relative to this package and strip trailing space."""
    return _PROMPTS_DIR.joinpath(*parts).read_text(encoding="utf-8").strip()


@dataclass(frozen=True, slots=True)
class QuickAction:
    """A one-click analysis preset.

    Attributes:
        label: Button text shown to the user (Spanish, with an emoji anchor).
        prompt: The instruction sent to the model (loaded from a ``.md`` file).
    """

    label: str
    prompt: str


SYSTEM_PROMPT: str = _read("system.md")
"""System prompt establishing the assistant's behavior."""

_CONTEXT_TEMPLATE: str = _read("context.md")

# Label (UI string) → markdown file under ``actions/``. The body is the
# instruction; labels stay in code because they are interface text.
_ACTION_FILES: tuple[tuple[str, str], ...] = (
    ("📝 Resumen", "summary.md"),
    ("🔑 Puntos clave", "key_points.md"),
    ("✅ Tareas y acuerdos", "tasks.md"),
    ("🗒️ Acta de reunión", "minutes.md"),
)

QUICK_ACTIONS: tuple[QuickAction, ...] = tuple(
    QuickAction(label=label, prompt=_read("actions", filename)) for label, filename in _ACTION_FILES
)
"""One-click analysis presets shown as buttons in the AI panel."""


def build_context(transcript_timestamped: str, *, language: str, duration_seconds: float) -> str:
    """Assemble the context block prepended to the model's system prompt.

    Uses literal ``{{placeholder}}`` replacement (not :meth:`str.format`)
    so braces inside the transcript can never break templating.

    Args:
        transcript_timestamped: The transcript rendered as ``[MM:SS] text``.
        language: Detected (or forced) language code of the audio.
        duration_seconds: Audio duration in seconds.

    Returns:
        A Spanish context block embedding the transcript and its metadata.
    """
    minutes = duration_seconds / 60.0
    return (
        _CONTEXT_TEMPLATE.replace("{{language}}", language)
        .replace("{{minutes}}", f"{minutes:.1f}")
        .replace("{{transcript}}", transcript_timestamped)
    )
