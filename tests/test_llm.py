"""Tests for `audicop.llm`.

Both provider SDKs are faked via ``sys.modules`` so the suite never makes
a network call or needs the real ``openai`` / ``google-genai`` packages.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from audicop import llm
from audicop.llm import ChatMessage, LLMError, available_models, stream_chat


def test_available_models() -> None:
    assert "gpt-4o-mini" in available_models("openai")
    assert available_models("gemini")[0].startswith("gemini")


def test_stream_chat_requires_api_key() -> None:
    with pytest.raises(LLMError, match="Falta la API key"):
        list(stream_chat(provider="openai", api_key="  ", model="x", system="s", messages=[]))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
def _openai_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def test_stream_openai_yields_chunks() -> None:
    fake_stream = [_openai_chunk("Hola"), _openai_chunk(" mundo"), _openai_chunk("")]
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(fake_stream)
    fake_openai_module = MagicMock()
    fake_openai_module.OpenAI.return_value = fake_client

    with patch.dict("sys.modules", {"openai": fake_openai_module}):
        out = list(
            stream_chat(
                provider="openai",
                api_key="sk-test",
                model="gpt-4o-mini",
                system="sys",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )

    assert "".join(out) == "Hola mundo"
    # system + user message forwarded
    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert kwargs["stream"] is True


def test_stream_openai_missing_sdk() -> None:
    with (
        patch.dict("sys.modules", {"openai": None}),
        pytest.raises(LLMError, match="no está instalado"),
    ):
        list(
            stream_chat(
                provider="openai",
                api_key="sk-test",
                model="gpt-4o-mini",
                system="s",
                messages=[],
            )
        )


def test_stream_openai_auth_error_is_friendly() -> None:
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("Invalid API key (401)")
    fake_openai_module = MagicMock()
    fake_openai_module.OpenAI.return_value = fake_client

    with (
        patch.dict("sys.modules", {"openai": fake_openai_module}),
        pytest.raises(LLMError, match="rechazó la API key"),
    ):
        list(
            stream_chat(
                provider="openai",
                api_key="sk-bad",
                model="gpt-4o-mini",
                system="s",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
def _fake_genai_modules(stream_chunks: list[SimpleNamespace] | Exception):
    """Build fake `google`, `google.genai` and `google.genai.types` modules.

    Returns ``(fake_client, sys_modules_patch_dict)`` ready for patch.dict.
    """
    fake_client = MagicMock()
    if isinstance(stream_chunks, Exception):
        fake_client.models.generate_content_stream.side_effect = stream_chunks
    else:
        fake_client.models.generate_content_stream.return_value = iter(stream_chunks)

    genai_mod = MagicMock()
    genai_mod.Client.return_value = fake_client

    types_mod = MagicMock()
    types_mod.Content.side_effect = lambda **kw: SimpleNamespace(**kw)
    types_mod.Part.from_text.side_effect = lambda *, text: SimpleNamespace(text=text)
    types_mod.GenerateContentConfig.side_effect = lambda **kw: SimpleNamespace(**kw)
    genai_mod.types = types_mod

    google_mod = ModuleType("google")
    google_mod.genai = genai_mod  # type: ignore[attr-defined]

    patch_dict = {
        "google": google_mod,
        "google.genai": genai_mod,
        "google.genai.types": types_mod,
    }
    return fake_client, patch_dict


def test_stream_gemini_yields_chunks() -> None:
    chunks = [
        SimpleNamespace(text="Hola"),
        SimpleNamespace(text=" Gemini"),
        SimpleNamespace(text=None),
    ]
    fake_client, patch_dict = _fake_genai_modules(chunks)

    with patch.dict("sys.modules", patch_dict):
        out = list(
            stream_chat(
                provider="gemini",
                api_key="g-test",
                model="gemini-2.0-flash",
                system="sys",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )

    assert "".join(out) == "Hola Gemini"
    _, kwargs = fake_client.models.generate_content_stream.call_args
    assert kwargs["model"] == "gemini-2.0-flash"
    assert kwargs["config"].system_instruction == "sys"


def test_stream_gemini_quota_error_is_friendly() -> None:
    _, patch_dict = _fake_genai_modules(RuntimeError("429 quota exceeded"))

    with (
        patch.dict("sys.modules", patch_dict),
        pytest.raises(LLMError, match="límite de uso o cuota"),
    ):
        list(
            stream_chat(
                provider="gemini",
                api_key="g-test",
                model="gemini-2.0-flash",
                system="s",
                messages=[ChatMessage(role="user", content="hi")],
            )
        )


def test_stream_gemini_missing_sdk() -> None:
    # A `google` module with no `genai` attribute + a None submodule forces
    # `from google import genai` to raise ImportError.
    bare_google = ModuleType("google")
    with (
        patch.dict("sys.modules", {"google": bare_google, "google.genai": None}),
        pytest.raises(LLMError, match="no está instalado"),
    ):
        list(
            stream_chat(
                provider="gemini",
                api_key="g-test",
                model="gemini-2.0-flash",
                system="s",
                messages=[],
            )
        )


def test_friendly_error_network() -> None:
    msg = llm._friendly_error("OpenAI", RuntimeError("Connection timeout"))
    assert "conexión" in msg.lower() or "conectar" in msg.lower()


def test_friendly_error_generic() -> None:
    msg = llm._friendly_error("Gemini", RuntimeError("something weird"))
    assert "Gemini" in msg
    assert "something weird" in msg
