"""Lexical retrieval over a timestamped transcript, for free-form questions.

Long meetings are condensed to notes for the summary actions (see
:mod:`app.services.summarize`), but that summary is *lossy*: a specific
question — "when was IBT founded?" — cannot be answered from a global summary
that dropped the detail. Worse, a small model then answers "not mentioned"
with false confidence.

So for a **free question** on a long audio we do the opposite of condensing:
we pull the transcript LINES most relevant to the question and answer over
those real excerpts, keeping every ``[MM:SS]`` mark exact. This is a tiny,
dependency-free lexical retriever (keyword overlap) — no embeddings, no model,
fully local. It is deliberately high-recall: when in doubt it includes more
lines, since the model can ignore an irrelevant excerpt but cannot invent a
dropped one.
"""

from __future__ import annotations

import re
import unicodedata

# Content-free words to ignore when scoring, so a question keys on its nouns
# ("fundó", "IBT") instead of its scaffolding ("cuándo", "se", "the"). Kept
# small on purpose — over-filtering hurts recall more than a few stopwords help.
_STOPWORDS: frozenset[str] = frozenset(
    {
        # Spanish
        "que",
        "de",
        "la",
        "el",
        "en",
        "y",
        "a",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "del",
        "al",
        "se",
        "su",
        "sus",
        "lo",
        "le",
        "les",
        "con",
        "por",
        "para",
        "como",
        "mas",
        "pero",
        "este",
        "esta",
        "esto",
        "estos",
        "estas",
        "ese",
        "esa",
        "eso",
        "esos",
        "esas",
        "cuando",
        "donde",
        "quien",
        "quienes",
        "cual",
        "cuales",
        "cuanto",
        "cuanta",
        "cuantos",
        "cuantas",
        "porque",
        "sobre",
        "hay",
        "fue",
        "son",
        "es",
        "era",
        "ser",
        "hubo",
        "dice",
        "dijo",
        "tal",
        "muy",
        "ya",
        "si",
        "no",
        "sin",
        "hasta",
        "desde",
        "entre",
        "tambien",
        "todo",
        "toda",
        "todos",
        "todas",
        "algun",
        "alguna",
        "me",
        "te",
        "nos",
        "yo",
        "tu",
        "ella",
        "ellos",
        "ellas",
        "usted",
        # English
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "was",
        "are",
        "for",
        "on",
        "with",
        "what",
        "when",
        "where",
        "who",
        "which",
        "how",
        "why",
        "did",
        "does",
        "do",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "about",
    }
)

_TOKEN_RE = re.compile(r"[0-9a-zñ]+")


def _fold(text: str) -> str:
    """Lowercase and strip accents, for accent-insensitive matching."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _terms(text: str) -> list[str]:
    """Tokenize into content terms (accent-folded, stopwords/short words dropped)."""
    return [
        tok
        for tok in _TOKEN_RE.findall(_fold(text))
        if tok not in _STOPWORDS and (len(tok) >= 3 or tok.isdigit())
    ]


def query_terms(query: str) -> set[str]:
    """Return the distinct content terms of a user question."""
    return set(_terms(query))


def search_transcript(
    transcript: str,
    query: str,
    *,
    max_chars: int,
    window: int = 1,
) -> str:
    """Return the transcript excerpts most relevant to ``query``, in time order.

    Each non-empty line is scored by how many *distinct* query terms it
    contains (ties broken by total hits). The best lines — plus a one-line
    neighbour window for context — are collected within a character budget and
    concatenated chronologically, so the excerpt reads naturally and every
    ``[MM:SS]`` mark is a real transcript mark.

    Args:
        transcript: The full ``[MM:SS] text`` transcript.
        query: The user's free-form question.
        max_chars: Character budget for the returned excerpt (keeps the prompt
            within the model's context window).
        window: Neighbouring lines to include on each side of a match.

    Returns:
        The selected excerpt, or ``""`` when the query has no content terms or
        nothing matches (the caller then falls back to the condensed notes).
    """
    qterms = query_terms(query)
    if not qterms:
        return ""

    lines = [ln for ln in transcript.splitlines() if ln.strip()]
    scored: list[tuple[int, int, int]] = []  # (distinct hits, total hits, index)
    for i, line in enumerate(lines):
        line_terms = _terms(line)
        if not line_terms:
            continue
        line_set = set(line_terms)
        distinct = sum(1 for t in qterms if t in line_set)
        if distinct:
            total = sum(1 for t in line_terms if t in qterms)
            scored.append((distinct, total, i))
    if not scored:
        return ""

    scored.sort(reverse=True)  # best matches first
    keep: set[int] = set()
    budget = max_chars
    for _distinct, _total, i in scored:
        neighbours = [
            j for j in range(max(0, i - window), min(len(lines), i + window + 1)) if j not in keep
        ]
        cost = sum(len(lines[j]) + 1 for j in neighbours)
        if cost > budget and keep:
            break  # budget spent; stop before overflowing the context window
        keep.update(neighbours)
        budget -= cost
        if budget <= 0:
            break
    return "\n".join(lines[j] for j in sorted(keep))
