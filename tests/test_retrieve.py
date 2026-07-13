"""Tests for the lexical transcript retriever (`app.services.retrieve`)."""

from __future__ import annotations

from app.services import retrieve

_TRANSCRIPT = "\n".join(
    [
        "[00:10] Bienvenidos a la reunión de estrategia.",
        "[05:28] Las máquinas han aprendido a aprender.",
        "[18:43] En 2014, IBT nació con tecnología de punta para el sector.",
        "[20:08] Hay más de 40 sistemas vivos en uso hoy.",
        "[27:23] Para la modernización se pensó en comprar otro GIS.",
        "[34:38] Un aplicativo para pedir citas está por salir.",
    ]
)


def test_finds_specific_fact_line() -> None:
    """A concrete question surfaces the exact line that holds the answer."""
    out = retrieve.search_transcript(_TRANSCRIPT, "¿cuándo se fundó IBT?", max_chars=2000)
    assert "[18:43]" in out
    assert "2014" in out


def test_accent_and_case_insensitive() -> None:
    out = retrieve.search_transcript(_TRANSCRIPT, "MODERNIZACION", max_chars=2000)
    assert "[27:23]" in out


def test_empty_when_only_stopwords() -> None:
    """A question with no content terms returns nothing (caller falls back)."""
    assert retrieve.search_transcript(_TRANSCRIPT, "y de lo que se", max_chars=2000) == ""


def test_no_match_returns_empty() -> None:
    assert retrieve.search_transcript(_TRANSCRIPT, "criptomonedas blockchain", max_chars=2000) == ""


def test_char_budget_is_respected() -> None:
    """The excerpt never exceeds the budget; a tight budget still returns some."""
    out = retrieve.search_transcript(_TRANSCRIPT, "sistemas citas aplicativo IBT", max_chars=120)
    assert 0 < len(out) <= 120 + 60  # one line (+ its neighbour window)


def test_excerpts_are_chronological() -> None:
    """Selected lines come back in transcript order, not by score."""
    out = retrieve.search_transcript(
        _TRANSCRIPT, "aplicativo citas máquinas aprender", max_chars=2000, window=0
    )
    lines = out.splitlines()
    marks = [ln[1:6] for ln in lines]
    assert marks == sorted(marks)
