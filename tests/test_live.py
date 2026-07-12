"""Tests for `app.services.live` (live transcription while recording)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.adapters.transcriber import TranscriptSegment
from app.core import config
from app.services.live import LiveTranscriber

_SR = config.TARGET_SAMPLE_RATE


def _block(seconds: float = 0.5, value: float = 0.1) -> np.ndarray:
    return np.full(int(_SR * seconds), value, dtype=np.float32)


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting for live worker")


def _make(tmp_path: Path, tracks: tuple[str, ...], window_s: float = 1.0):
    events: list[dict] = []
    decoded: list[Path] = []

    def decode(path: Path):
        decoded.append(path)
        return [TranscriptSegment(start=0.0, end=0.5, text="hola equipo")]

    live = LiveTranscriber(
        tracks=tracks,
        work_dir=tmp_path,
        decode=decode,
        on_event=events.append,
        window_s=window_s,
    )
    return live, events, decoded


def test_single_track_window_decodes_with_offsets(tmp_path: Path) -> None:
    live, events, decoded = _make(tmp_path, ("mic",), window_s=1.0)
    live.start()
    for _ in range(4):  # 2 s fed in 0.5 s blocks → two 1 s windows
        live.feed("mic", _block())
    _wait_until(lambda: len(decoded) >= 2)
    live.stop()

    segs = [e for e in events if "text" in e]
    assert len(segs) >= 2
    assert segs[0]["start"] == 0.0  # first window starts at 0
    assert segs[1]["start"] == 1.0  # second window offset by one window
    assert events[-1] == {"done": True}


def test_two_tracks_are_mixed_pairwise(tmp_path: Path) -> None:
    live, events, decoded = _make(tmp_path, ("mic", "others"), window_s=1.0)
    live.start()
    # Only one track feeding → nothing decodes (pairing requires both).
    live.feed("mic", _block())
    live.feed("mic", _block())
    time.sleep(0.5)
    assert decoded == []
    # The second track catches up → the pair mixes and the window decodes.
    live.feed("others", _block())
    live.feed("others", _block())
    _wait_until(lambda: len(decoded) >= 1)
    live.stop()
    assert any("text" in e for e in events)


def test_stop_flushes_tail(tmp_path: Path) -> None:
    live, events, decoded = _make(tmp_path, ("mic",), window_s=10.0)
    live.start()
    live.feed("mic", _block(1.0))  # 1 s — under the window, over the min tail
    time.sleep(0.3)
    assert decoded == []  # not enough for a window yet
    live.stop()  # flush must decode the pending tail
    assert len(decoded) == 1
    assert events[-1] == {"done": True}


def test_tiny_tail_is_dropped(tmp_path: Path) -> None:
    live, events, decoded = _make(tmp_path, ("mic",), window_s=10.0)
    live.start()
    live.feed("mic", _block(0.2))  # under LIVE_MIN_TAIL_S
    time.sleep(0.3)
    live.stop()
    assert decoded == []
    assert events == [{"done": True}]


def test_decode_failure_keeps_streaming(tmp_path: Path) -> None:
    events: list[dict] = []
    calls: list[int] = []

    def flaky_decode(path: Path):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return [TranscriptSegment(start=0.0, end=0.5, text="recuperado")]

    live = LiveTranscriber(
        tracks=("mic",),
        work_dir=tmp_path,
        decode=flaky_decode,
        on_event=events.append,
        window_s=0.5,
    )
    live.start()
    live.feed("mic", _block(0.5))
    _wait_until(lambda: len(calls) >= 1)
    live.feed("mic", _block(0.5))
    _wait_until(lambda: len(calls) >= 2)
    live.stop()
    assert any(e.get("text") == "recuperado" for e in events)
    assert events[-1] == {"done": True}
