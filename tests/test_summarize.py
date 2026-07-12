"""Tests for `app.services.summarize` (map-reduce chunking for long audios)."""

from __future__ import annotations

from app.core import config
from app.services.summarize import (
    chunk_by_lines,
    estimate_tokens,
    map_instructions,
    needs_map_reduce,
)


def _transcript(lines: int, line_len: int = 80) -> str:
    body = "palabra " * (line_len // 8)
    return "\n".join(f"[{i:02d}:00] {body.strip()}" for i in range(lines))


def test_estimate_tokens_rough_ratio() -> None:
    assert estimate_tokens("a" * 1000) == 350


def test_needs_map_reduce_thresholds() -> None:
    short = "a" * 100
    long = "a" * int((config.LLM_MAPREDUCE_TOKEN_THRESHOLD / 0.35) + 100)
    assert needs_map_reduce(short) is False
    assert needs_map_reduce(long) is True


def test_short_transcript_is_single_chunk() -> None:
    text = _transcript(5)
    assert chunk_by_lines(text, max_tokens=10_000, overlap_tokens=100) == [text]


def test_chunks_respect_budget_and_line_boundaries() -> None:
    text = _transcript(40)
    chunks = chunk_by_lines(text, max_tokens=300, overlap_tokens=0)
    assert len(chunks) > 1
    max_chars = int(300 / 0.35)
    for chunk in chunks:
        assert len(chunk) <= max_chars + 100  # one line of slack
        for line in chunk.splitlines():
            assert line.startswith("[")  # never split mid-line


def test_all_lines_survive_chunking() -> None:
    text = _transcript(40)
    chunks = chunk_by_lines(text, max_tokens=300, overlap_tokens=0)
    joined = "\n".join(chunks)
    for line in text.splitlines():
        assert line in joined


def test_overlap_repeats_previous_tail() -> None:
    text = _transcript(40)
    chunks = chunk_by_lines(text, max_tokens=300, overlap_tokens=60)
    assert len(chunks) > 1
    first_tail = chunks[0].splitlines()[-1]
    assert first_tail in chunks[1]  # the tail seeds the next chunk


def test_empty_transcript_no_chunks() -> None:
    assert chunk_by_lines("", max_tokens=100, overlap_tokens=10) == []


def test_map_instructions_mention_marks_and_injection_guard() -> None:
    text = map_instructions()
    assert "[MM:SS]" in text
    assert "ignora" in text.lower()
