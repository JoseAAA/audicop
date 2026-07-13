"""API tests for the FastAPI app (TestClient, no network, mocked logic)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.adapters.hardware import HardwareInfo
from app.api import chat as chat_mod
from app.api import hardware as hw_mod
from app.api import transcribe as tr_mod
from app.main import app
from app.services import transcript_store
from app.services.recommender import ModelChoice
from app.services.transcript_store import StoredTranscript


def _fake_hw() -> HardwareInfo:
    return HardwareInfo(
        os_name="Linux",
        os_release="6.0",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_brand="Test CPU",
        ram_total_gb=32.0,
        ram_available_gb=16.0,
        has_cuda=True,
        gpu_name="NVIDIA Test",
        gpu_vram_total_gb=12.0,
        gpu_vram_free_gb=10.0,
        gpu_driver_version="555.42",
    )


def _fake_choice() -> ModelChoice:
    return ModelChoice(
        model_size="large-v3",
        compute_type="float16",
        device="cuda",
        rationale="Test rationale",
    )


def _sse_payloads(text: str) -> list[str]:
    """Extract the JSON payloads from an SSE response body."""
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_get_hardware() -> None:
    with (
        patch.object(hw_mod, "detect_hardware", return_value=_fake_hw()),
        patch.object(hw_mod, "recommend", return_value=_fake_choice()),
        TestClient(app) as client,
    ):
        r = client.get("/api/hardware")

    assert r.status_code == 200
    body = r.json()
    assert body["hardware"]["gpu_name"] == "NVIDIA Test"
    assert body["recommendation"]["model_size"] == "large-v3"
    assert body["capacity"]["minutes_per_hour"] >= 1
    assert "es" in body["options"]["languages"]
    # 100% local AI: the endpoint exposes the recommended on-device model
    # (roomy fake GPU → the Qwen3 4B quality tier).
    assert body["ai"]["local"]["available"] is True
    assert body["ai"]["local"]["model_key"] == "qwen3-4b"
    assert "quick_actions" in body["ai"]


def test_transcribe_rejects_unsupported_extension() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/api/transcribe",
            data={"model_size": "tiny", "compute_type": "int8", "device": "cpu"},
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    assert r.status_code == 400
    assert "no soportada" in r.json()["detail"]


def test_transcribe_requires_input() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/api/transcribe",
            data={"model_size": "tiny", "compute_type": "int8", "device": "cpu"},
        )
    assert r.status_code == 400


def test_host_guard_blocks_rebound_host_even_on_get() -> None:
    """DNS rebinding defense: a non-localhost Host is rejected on ANY method,
    so /api/meetings (private content) can't be read via a rebound name."""
    with TestClient(app) as client:
        r = client.get("/api/meetings", headers={"Host": "evil.example:8000"})
    assert r.status_code == 403
    assert "Host" in r.json()["detail"]


def test_host_guard_allows_localhost_variants() -> None:
    with TestClient(app) as client:
        for host in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000", "localhost"):
            r = client.get("/api/health", headers={"Host": host})
            assert r.status_code == 200, host


def test_csrf_blocks_remote_origin() -> None:
    """A POST from a remote site is rejected before reaching the endpoint."""
    with TestClient(app) as client:
        r = client.post(
            "/api/transcribe",
            headers={"Origin": "http://evil.example"},
            data={"model_size": "tiny", "compute_type": "int8", "device": "cpu"},
        )
    assert r.status_code == 403


def test_csrf_allows_localhost_origin() -> None:
    """A same-origin (localhost) POST passes the guard (then fails on missing input)."""
    with TestClient(app) as client:
        r = client.post(
            "/api/transcribe",
            headers={"Origin": "http://localhost:8000"},
            data={"model_size": "tiny", "compute_type": "int8", "device": "cpu"},
        )
    assert r.status_code == 400


def test_transcribe_flow_streams_events() -> None:
    """POST starts a job; SSE streams meta → segment → done."""
    fake_events = [
        {"type": "status", "label": "…"},
        {"type": "meta", "duration": 4.0, "estimated_seconds": 1.0},
        {
            "type": "segment",
            "start": 0.0,
            "end": 2.0,
            "text": "Hola",
            "pct": 0.5,
            "elapsed": 0.1,
            "eta": None,
        },
        {"type": "done", "language": "es", "language_probability": 0.9, "duration": 4.0},
    ]

    with (
        patch.object(tr_mod, "iter_transcription", return_value=iter(fake_events)),
        patch.object(tr_mod, "is_model_cached", return_value=True),
        TestClient(app) as client,
    ):
        start = client.post(
            "/api/transcribe",
            data={
                "model_size": "tiny",
                "compute_type": "int8",
                "device": "cpu",
                "language": "auto",
                "task": "transcribe",
                "vad_filter": "true",
            },
            files={"file": ("clip.mp3", b"fakebytes", "audio/mpeg")},
        )
        assert start.status_code == 200
        job_id = start.json()["job_id"]
        assert start.json()["model_cached"] is True

        with client.stream("GET", f"/api/transcribe/{job_id}/events") as resp:
            body = "".join(resp.iter_text())

    payloads = _sse_payloads(body)
    assert any('"type": "meta"' in p for p in payloads)
    assert any('"type": "segment"' in p for p in payloads)
    assert any('"type": "done"' in p for p in payloads)


def test_transcribe_events_unknown_job() -> None:
    with TestClient(app) as client:
        r = client.get("/api/transcribe/does-not-exist/events")
    assert r.status_code == 404


_CHAT_BODY = {
    "transcript_timestamped": "[00:00] Hola",
    "language": "es",
    "duration": 4.0,
    "history": [{"role": "user", "content": "resume"}],
}


def test_chat_streams_deltas() -> None:
    """The chat endpoint streams the on-device model's reply as SSE deltas."""
    fake_model = SimpleNamespace(stream_chat=lambda **_k: iter(["Resu", "men"]))
    choice = SimpleNamespace(
        available=True,
        repo_id="r",
        filename="m.gguf",
        device="cpu",
        n_gpu_layers=0,
        n_ctx=512,
    )
    with (
        patch.object(chat_mod, "detect_hardware", return_value=None),
        patch.object(chat_mod, "recommend_llm", return_value=choice),
        patch.object(chat_mod.local_llm, "get_local_llm", return_value=fake_model),
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=_CHAT_BODY) as resp,
    ):
        body = "".join(resp.iter_text())

    payloads = _sse_payloads(body)
    # The citation fixer re-chunks the stream at line boundaries, so the two
    # model chunks arrive merged as one completed line.
    assert any('"delta": "Resumen"' in p for p in payloads)
    assert any('"done": true' in p for p in payloads)


def test_chat_falls_back_to_stored_transcript() -> None:
    """An empty transcript in the request uses the server-side stored one."""
    captured: dict[str, object] = {}

    def fake_stream(**kwargs: object):
        captured.update(kwargs)
        return iter(["ok"])

    fake_model = SimpleNamespace(stream_chat=fake_stream)
    transcript_store.save(
        StoredTranscript(timestamped="[00:00] Texto guardado", language="es", duration=3.0)
    )
    body_req = {**_CHAT_BODY, "transcript_timestamped": ""}
    try:
        with (
            patch.object(chat_mod.local_llm, "get_active", return_value=fake_model),
            TestClient(app) as client,
            client.stream("POST", "/api/chat", json=body_req) as resp,
        ):
            body = "".join(resp.iter_text())
    finally:
        transcript_store.clear()

    assert any('"delta": "ok"' in p for p in _sse_payloads(body))
    assert "Texto guardado" in str(captured["system"])


def test_chat_errors_when_no_transcript_anywhere() -> None:
    """Empty request + empty store yields a clear Spanish error event."""
    transcript_store.clear()
    body_req = {**_CHAT_BODY, "transcript_timestamped": ""}
    with (
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=body_req) as resp,
    ):
        body = "".join(resp.iter_text())
    assert any('"error"' in p and "Transcribe un audio" in p for p in _sse_payloads(body))


def test_chat_long_transcript_triggers_map_reduce(monkeypatch) -> None:
    """Above the token threshold the chat condenses chunk-by-chunk (with phase
    events) and answers over the condensed notes."""
    from app.core import config as cfg

    monkeypatch.setattr(cfg, "LLM_MAPREDUCE_TOKEN_THRESHOLD", 50)
    monkeypatch.setattr(cfg, "LLM_SYNTHESIS_TARGET_TOKENS", 50)
    monkeypatch.setattr(cfg, "LLM_CHUNK_PROMPT_OVERHEAD_TOKENS", 10)
    monkeypatch.setattr(cfg, "LLM_CHUNK_OVERLAP_TOKENS", 0)

    calls: list[str] = []

    def fake_stream(**kwargs: object):
        msgs = kwargs["messages"]
        content = msgs[-1].content  # type: ignore[index]
        calls.append(content)
        if "TAREA: Condensa" in content:
            return iter(["[00:00] nota condensada"])
        return iter(["[00:00] Respuesta final"])

    fake_model = SimpleNamespace(stream_chat=fake_stream)
    long_transcript = "\n".join(f"[{i:02d}:00] bla bla bla bla bla" for i in range(20))
    # A quick action (TAREA:) takes the condense-the-whole-meeting path.
    body_req = {
        **_CHAT_BODY,
        "transcript_timestamped": long_transcript,
        "history": [{"role": "user", "content": "TAREA: Resume la reunión"}],
    }

    with (
        patch.object(chat_mod.local_llm, "get_active", return_value=fake_model),
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=body_req) as resp,
    ):
        body = "".join(resp.iter_text())

    payloads = _sse_payloads(body)
    assert any('"phase": "map"' in p for p in payloads)  # progress events emitted
    assert any('"phase": "combine"' in p for p in payloads)  # notes → prose ran
    assert any("Respuesta final" in p for p in payloads)
    map_calls = [c for c in calls if "TAREA: Condensa" in c]
    assert len(map_calls) >= 2  # long transcript → several chunks
    # The combine pass received the condensed notes (not the raw transcript).
    combine_calls = [c for c in calls if "TAREA: Une" in c]
    assert len(combine_calls) == 1 and "nota condensada" in combine_calls[0]


def test_chat_map_reduce_recurses_when_notes_still_long(monkeypatch) -> None:
    """If round-1 notes still exceed the budget, they are condensed again
    (round 2) before answering — otherwise the final pass echoes dozens of
    notes and gets truncated (real bug on a 2.5 h meeting)."""
    from app.core import config as cfg

    monkeypatch.setattr(cfg, "LLM_MAPREDUCE_TOKEN_THRESHOLD", 50)
    monkeypatch.setattr(cfg, "LLM_SYNTHESIS_TARGET_TOKENS", 50)
    monkeypatch.setattr(cfg, "LLM_CHUNK_PROMPT_OVERHEAD_TOKENS", 10)
    monkeypatch.setattr(cfg, "LLM_CHUNK_OVERLAP_TOKENS", 0)

    rounds_seen: list[int] = []

    def fake_stream(**kwargs: object):
        content = kwargs["messages"][-1].content  # type: ignore[index]
        if "TAREA: Condensa" in content:
            rounds_seen.append(1)
            # Long note → after round 1 the joined notes STILL exceed 50 tokens.
            return iter(["[00:00] nota " + "larga " * 20])
        return iter(["[00:00] Respuesta final"])

    fake_model = SimpleNamespace(stream_chat=fake_stream)
    # Long lines so round-1 notes genuinely SHRINK the text (else the
    # no-progress guard stops the recursion).
    long_transcript = "\n".join(f"[{i:02d}:00] " + "palabra " * 40 for i in range(20))
    body_req = {
        **_CHAT_BODY,
        "transcript_timestamped": long_transcript,
        "history": [{"role": "user", "content": "TAREA: Resume la reunión"}],
    }

    with (
        patch.object(chat_mod.local_llm, "get_active", return_value=fake_model),
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=body_req) as resp,
    ):
        body = "".join(resp.iter_text())

    payloads = _sse_payloads(body)
    assert any('"round": 2' in p for p in payloads)  # a second condensation ran
    assert any("Respuesta final" in p for p in payloads)  # and it still answered


def test_chat_condensed_notes_are_cached_across_questions(monkeypatch) -> None:
    """The expensive map phase runs ONCE per transcript: the second question
    reuses the cached notes (no new 'TAREA: Condensa' calls, no phase events)."""
    from app.core import config as cfg

    monkeypatch.setattr(cfg, "LLM_MAPREDUCE_TOKEN_THRESHOLD", 50)
    monkeypatch.setattr(cfg, "LLM_SYNTHESIS_TARGET_TOKENS", 50)
    monkeypatch.setattr(cfg, "LLM_CHUNK_PROMPT_OVERHEAD_TOKENS", 10)
    monkeypatch.setattr(cfg, "LLM_CHUNK_OVERLAP_TOKENS", 0)
    chat_mod._NOTES_CACHE.clear()

    condensa_calls: list[int] = []

    def fake_stream(**kwargs: object):
        content = kwargs["messages"][-1].content  # type: ignore[index]
        if "TAREA: Condensa" in content:
            condensa_calls.append(1)
            return iter(["[00:00] nota"])
        return iter(["[00:00] Respuesta"])

    fake_model = SimpleNamespace(stream_chat=fake_stream)
    long_transcript = "\n".join(f"[{i:02d}:00] " + "palabra " * 40 for i in range(10))
    body_req = {
        **_CHAT_BODY,
        "transcript_timestamped": long_transcript,
        "history": [{"role": "user", "content": "TAREA: Resume la reunión"}],
    }

    with (
        patch.object(chat_mod.local_llm, "get_active", return_value=fake_model),
        TestClient(app) as client,
    ):
        with client.stream("POST", "/api/chat", json=body_req) as r1:
            "".join(r1.iter_text())
        first_round_calls = len(condensa_calls)
        assert first_round_calls >= 2  # the map phase ran

        with client.stream("POST", "/api/chat", json=body_req) as r2:
            body2 = "".join(r2.iter_text())

    assert len(condensa_calls) == first_round_calls  # no re-condensation
    assert not any('"phase"' in p for p in _sse_payloads(body2))
    assert any("Respuesta" in p for p in _sse_payloads(body2))
    chat_mod._NOTES_CACHE.clear()


def test_chat_free_question_retrieves_from_full_transcript(monkeypatch) -> None:
    """A specific question on a long audio is answered from the transcript
    EXCERPTS that match it — never from lossy condensed notes (which would drop
    the fact and wrongly answer 'no se menciona'). No condensation runs."""
    from app.core import config as cfg

    monkeypatch.setattr(cfg, "LLM_MAPREDUCE_TOKEN_THRESHOLD", 50)

    seen_system: list[str] = []

    def fake_stream(**kwargs: object):
        content = kwargs["messages"][-1].content  # type: ignore[index]
        if "TAREA: Condensa" in content:
            raise AssertionError("a free question must NOT trigger condensation")
        seen_system.append(str(kwargs["system"]))
        return iter(["IBT nació en 2014 [18:43]."])

    fake_model = SimpleNamespace(stream_chat=fake_stream)
    lines = [f"[{i:02d}:00] relleno de la reunion numero {i}" for i in range(30)]
    lines[18] = "[18:43] En 2014, IBT nació con tecnología de punta."
    long_transcript = "\n".join(lines)
    body_req = {
        **_CHAT_BODY,
        "transcript_timestamped": long_transcript,
        "history": [{"role": "user", "content": "¿cuándo se fundó IBT?"}],
    }

    with (
        patch.object(chat_mod.local_llm, "get_active", return_value=fake_model),
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=body_req) as resp,
    ):
        body = "".join(resp.iter_text())

    payloads = _sse_payloads(body)
    assert not any('"phase"' in p for p in payloads)  # no map/combine phases
    # The retrieved excerpt (with the founding line) reached the model's context.
    assert any("[18:43]" in s and "2014" in s for s in seen_system)
    assert any("2014" in p for p in payloads)  # and the answer used it


def test_chat_short_transcript_stays_single_pass() -> None:
    """Below the threshold there is exactly one model call and no phase events."""
    calls: list[object] = []

    def fake_stream(**kwargs: object):
        calls.append(kwargs)
        return iter(["ok"])

    fake_model = SimpleNamespace(stream_chat=fake_stream)
    with (
        patch.object(chat_mod.local_llm, "get_active", return_value=fake_model),
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=_CHAT_BODY) as resp,
    ):
        body = "".join(resp.iter_text())

    assert len(calls) == 1
    assert not any('"phase"' in p for p in _sse_payloads(body))


def test_chat_reuses_loaded_model_without_re_recommending() -> None:
    """A loaded model is reused as-is: re-measuring memory would count the
    loaded model's own footprint and wrongly reject it mid-conversation."""
    fake_model = SimpleNamespace(stream_chat=lambda **_k: iter(["ok"]))

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("must not re-recommend while a model is loaded")

    with (
        patch.object(chat_mod.local_llm, "get_active", return_value=fake_model),
        patch.object(chat_mod, "recommend_llm", side_effect=_boom),
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=_CHAT_BODY) as resp,
    ):
        body = "".join(resp.iter_text())

    assert any('"delta": "ok"' in p for p in _sse_payloads(body))


def test_chat_surfaces_error_when_no_local_model() -> None:
    """When no local model fits the machine, the rationale streams as an error."""
    choice = SimpleNamespace(
        available=False,
        rationale="Poca memoria libre para un modelo local.",
        repo_id="",
        filename="",
        device="cpu",
        n_gpu_layers=0,
        n_ctx=0,
    )
    with (
        patch.object(chat_mod, "detect_hardware", return_value=None),
        patch.object(chat_mod, "recommend_llm", return_value=choice),
        TestClient(app) as client,
        client.stream("POST", "/api/chat", json=_CHAT_BODY) as resp,
    ):
        body = "".join(resp.iter_text())

    assert any('"error"' in p and "memoria" in p for p in _sse_payloads(body))
