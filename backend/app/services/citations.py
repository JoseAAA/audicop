"""Deterministic verification of the [MM:SS] citations in AI answers.

Small local models get the *facts* right but the *timestamp attribution*
wrong surprisingly often: they anchor on a mark cited earlier in the
conversation, or drop the mark entirely. Prompting reduces but does not
eliminate it — so we verify server-side, where the ground truth (the
timestamped transcript) lives.

For each completed line of the streamed answer:

1. Tokenize the line (accent-insensitive words, plus numbers — ``97`` or
   ``138`` are highly discriminative).
2. Score every transcript line by lexical overlap, weighting rare tokens
   higher (a word that appears in only one transcript line pins the fact).
3. If the line cites a mark whose transcript line clearly does NOT contain
   the fact while another line clearly does → **replace** the mark.
4. If the line states transcript facts but cites no mark → **prepend** the
   best mark.

Conservative on purpose: lines with ranges or multiple marks, and lines
whose overlap is weak (meta answers like "No se menciona en la
transcripción"), are left untouched.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass

# A single [MM:SS] / [HH:MM:SS] mark (no ranges — ranges are left alone).
_MARK_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")
# u2013/u2014 in the class below = en dash / em dash
_RANGE_RE = re.compile(
    "\\[\\d{1,2}:\\d{2}(?::\\d{2})?\\s*[\\u2013\\u2014-]\\s*\\d{1,2}:\\d{2}(?::\\d{2})?\\]"
)
_TOKEN_RE = re.compile(r"[a-zñ]{4,}|\d{2,}")

# Minimum overlap score to act at all, and dominance factor required to
# override a mark the model already cited.
_MIN_SCORE = 1.5
_REPLACE_FACTOR = 2.0


def _tokens(text: str) -> set[str]:
    """Return accent-insensitive word tokens (len >= 4) and numbers (len >= 2)."""
    folded = unicodedata.normalize("NFD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return set(_TOKEN_RE.findall(folded))


@dataclass(frozen=True, slots=True)
class _Entry:
    """One transcript line: its mark and its token set."""

    mark: str
    tokens: frozenset[str]


def parse_transcript(timestamped: str) -> list[_Entry]:
    """Split a ``[MM:SS] text`` transcript into scoreable entries."""
    entries: list[_Entry] = []
    for line in timestamped.splitlines():
        m = _MARK_RE.match(line.strip())
        if m is None:
            continue
        entries.append(_Entry(mark=m.group(1), tokens=frozenset(_tokens(line[m.end() :]))))
    return entries


class CitationFixer:
    """Fixes/adds [MM:SS] citations in a streamed answer, line by line.

    Feed it text chunks as they stream; it emits corrected text. Because a
    line can only be verified once complete, output is re-chunked at line
    boundaries — the UI still renders progressively (bullet by bullet).
    """

    def __init__(self, transcript_timestamped: str) -> None:
        """Build the fixer over the ground-truth timestamped transcript."""
        self._entries = parse_transcript(transcript_timestamped)
        # Rare tokens pin facts to lines: weight by in how many lines each
        # token appears (1 line → 1.0, 2 → 0.5, 3 → 0.25, more → 0).
        counts: dict[str, int] = {}
        for e in self._entries:
            for t in e.tokens:
                counts[t] = counts.get(t, 0) + 1
        self._weights = {t: {1: 1.0, 2: 0.5, 3: 0.25}.get(n, 0.0) for t, n in counts.items()}
        self._buffer = ""

    def _score(self, entry: _Entry, line_tokens: set[str]) -> float:
        return sum(self._weights.get(t, 0.0) for t in entry.tokens & line_tokens)

    def fix_line(self, line: str) -> str:
        """Return ``line`` with its citation verified (replaced/added if needed)."""
        if not self._entries or _RANGE_RE.search(line):
            return line  # ranges: too ambiguous to score halves — leave alone
        marks = _MARK_RE.findall(line)
        if len(marks) > 1:
            return line  # multi-cite lines are usually fine; don't second-guess
        line_tokens = _tokens(_MARK_RE.sub(" ", line))
        scored = [(self._score(e, line_tokens), e) for e in self._entries]
        best_score, best = max(scored, key=lambda pair: pair[0])
        if best_score < _MIN_SCORE:
            return line  # weak evidence (meta answer / paraphrase): hands off
        if not marks:
            stripped = line.lstrip()
            if stripped.startswith("**"):
                # Bold-labelled lines (**Tema:**, **Conclusión:**, section
                # headers) are synthesis, not a quoted moment: no mark.
                return line
            indent = line[: len(line) - len(stripped)]
            bullet = ""
            m = re.match(r"([-*•]\s+|\d+\.\s+)", stripped)
            if m is not None:
                bullet = m.group(0)
                stripped = stripped[m.end() :]
            return f"{indent}{bullet}[{best.mark}] {stripped}"
        cited = next((e for e in self._entries if e.mark == marks[0]), None)
        cited_score = self._score(cited, line_tokens) if cited is not None else 0.0
        if best.mark != marks[0] and best_score >= max(_MIN_SCORE, _REPLACE_FACTOR * cited_score):
            return line.replace(f"[{marks[0]}]", f"[{best.mark}]", 1)
        return line

    def feed(self, chunk: str) -> Iterator[str]:
        """Consume a streamed chunk; yield corrected text for completed lines."""
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            yield self.fix_line(line) + "\n"

    def flush(self) -> Iterator[str]:
        """Emit the (corrected) final line once the stream ends."""
        if self._buffer:
            yield self.fix_line(self._buffer)
            self._buffer = ""
