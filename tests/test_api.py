"""API tests for the FastAPI app (TestClient, no network, mocked logic)."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.adapters.hardware import HardwareInfo
from app.adapters.llm import LLMError
from app.api import chat as chat_mod
from app.api import hardware as hw_mod
from app.api import transcribe as tr_mod
from app.main import app
from app.services.recommender import ModelChoice


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
    "provider": "gemini",
    "model": "gemini-2.0-flash",
    "api_key": "k",
    "transcript_timestamped": "[00:00] Hola",
    "language": "es",
    "duration": 4.0,
    "history": [{"role": "user", "content": "resume"}],
}


def test_chat_streams_deltas() -> None:
    def fake_stream(**_kwargs: object):
        yield "Hola"
        yield " mundo"

    with patch.object(chat_mod.llm, "stream_chat", side_effect=fake_stream):  # noqa: SIM117
        with TestClient(app) as client:
            with client.stream("POST", "/api/chat", json=_CHAT_BODY) as resp:
                body = "".join(resp.iter_text())

    payloads = _sse_payloads(body)
    assert any('"delta": "Hola"' in p for p in payloads)
    assert any('"done": true' in p for p in payloads)


def test_chat_surfaces_llm_error() -> None:
    def boom(**_kwargs: object):
        raise LLMError("clave inválida")
        yield  # pragma: no cover - makes this a generator

    body_req = {**_CHAT_BODY, "provider": "openai", "model": "gpt-4o-mini", "history": []}
    with patch.object(chat_mod.llm, "stream_chat", side_effect=boom):  # noqa: SIM117
        with TestClient(app) as client:
            with client.stream("POST", "/api/chat", json=body_req) as resp:
                body = "".join(resp.iter_text())

    assert any('"error"' in p and "inválida" in p for p in _sse_payloads(body))
