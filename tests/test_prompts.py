"""Tests for `app.prompts`."""

from __future__ import annotations

from app.prompts import QUICK_ACTIONS, SYSTEM_PROMPT, QuickAction, build_context


def test_system_prompt_mentions_timestamps() -> None:
    """The system prompt must instruct the model to cite [MM:SS] moments."""
    assert "[MM:SS]" in SYSTEM_PROMPT or "MM:SS" in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.strip()


def test_quick_actions_present_and_nonempty() -> None:
    assert len(QUICK_ACTIONS) >= 3
    for action in QUICK_ACTIONS:
        assert isinstance(action, QuickAction)
        assert action.label.strip()
        assert action.prompt.strip()


def test_build_context_includes_transcript_and_meta() -> None:
    transcript = "[00:00] Hola\n[00:05] Mundo"
    ctx = build_context(transcript, language="es", duration_seconds=125.0)
    assert transcript in ctx
    assert "es" in ctx
    assert "2.1" in ctx  # 125s -> 2.08 min, rounded to 2.1
    assert "TRANSCRIPCIÓN" in ctx


def test_build_context_zero_duration() -> None:
    ctx = build_context("[00:00] x", language="en", duration_seconds=0.0)
    assert "0.0" in ctx
    assert "en" in ctx
