"""Provider-agnostic LLM client for the transcript chat/analysis.

Bring-your-own-key: the user supplies an API key (kept only in Streamlit
session state) and picks a provider + model. Supported providers are
OpenAI and Google Gemini. Each provider SDK is imported lazily inside its
``_stream_*`` function as a safety net, so a broken/partial install
degrades to a clear message instead of crashing the whole app.

Responses are streamed (an iterator of text chunks) so the UI can render
them token-by-token with ``st.write_stream``.

PRIVACY: calling :func:`stream_chat` sends the provided messages — which
include the transcript text — to the chosen cloud provider. The audio
itself never leaves the machine; only the already-transcribed text does,
and only when the user actively uses the chat. See ``AGENTS.md`` §3.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final, Literal

logger = logging.getLogger(__name__)

Provider = Literal["openai", "gemini"]

PROVIDER_LABELS: Final[dict[Provider, str]] = {
    "openai": "OpenAI (ChatGPT)",
    "gemini": "Google Gemini",
}

# Curated model shortlists. Kept small on purpose: the user wants "elegir un
# modelo", not a wall of 40 options. First entry is the sensible default.
MODELS_BY_PROVIDER: Final[dict[Provider, tuple[str, ...]]] = {
    "openai": ("gpt-4o-mini", "gpt-4o"),
    "gemini": ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"),
}

API_KEY_HELP: Final[dict[Provider, str]] = {
    "openai": "Consíguela en https://platform.openai.com/api-keys",
    "gemini": "Consíguela gratis en https://aistudio.google.com/apikey",
}


class LLMError(RuntimeError):
    """Raised when an LLM provider call fails (auth, network, quota, etc.)."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single chat turn.

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        content: The message text.
    """

    role: Literal["user", "assistant"]
    content: str


def available_models(provider: Provider) -> tuple[str, ...]:
    """Return the curated model list for a provider."""
    return MODELS_BY_PROVIDER.get(provider, ())


def stream_chat(
    *,
    provider: Provider,
    api_key: str,
    model: str,
    system: str,
    messages: Sequence[ChatMessage],
) -> Iterator[str]:
    """Stream an assistant reply from the chosen provider.

    Args:
        provider: ``"openai"`` or ``"gemini"``.
        api_key: The user's API key (never persisted by Audicop).
        model: Model identifier from :func:`available_models`.
        system: System prompt establishing the assistant's behavior.
        messages: Conversation so far, oldest first.

    Yields:
        Text chunks of the assistant's reply, in order.

    Raises:
        LLMError: If the key is missing, the SDK is not installed, or the
            provider returns an error.
    """
    if not api_key.strip():
        raise LLMError("Falta la API key. Pégala en el panel para usar el chat.")

    if provider == "openai":
        yield from _stream_openai(api_key=api_key, model=model, system=system, messages=messages)
    elif provider == "gemini":
        yield from _stream_gemini(api_key=api_key, model=model, system=system, messages=messages)
    else:  # pragma: no cover - guarded by the Provider type
        raise LLMError(f"Proveedor no soportado: {provider!r}")


def _stream_openai(
    *,
    api_key: str,
    model: str,
    system: str,
    messages: Sequence[ChatMessage],
) -> Iterator[str]:
    """Stream a reply from the OpenAI Chat Completions API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMError("El soporte de OpenAI no está instalado. Reinstala con `uv sync`.") from exc

    payload = [{"role": "system", "content": system}]
    payload.extend({"role": m.role, "content": m.content} for m in messages)

    try:
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=payload,  # type: ignore[arg-type]  # SDK accepts these dicts
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(_friendly_error("OpenAI", exc)) from exc


def _stream_gemini(
    *,
    api_key: str,
    model: str,
    system: str,
    messages: Sequence[ChatMessage],
) -> Iterator[str]:
    """Stream a reply from the Google Gemini API (google-genai SDK)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMError("El soporte de Gemini no está instalado. Reinstala con `uv sync`.") from exc

    # Gemini has no dedicated "system" role; pass it via system_instruction.
    contents = [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part.from_text(text=m.content)],
        )
        for m in messages
    ]

    try:
        client = genai.Client(api_key=api_key)
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        for chunk in stream:
            piece = getattr(chunk, "text", None)
            if piece:
                yield piece
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(_friendly_error("Gemini", exc)) from exc


def _friendly_error(provider_name: str, exc: Exception) -> str:
    """Turn a provider exception into an actionable Spanish message."""
    text = str(exc).lower()
    if any(k in text for k in ("api key", "api_key", "unauthorized", "401", "invalid")):
        return (
            f"{provider_name} rechazó la API key. Revisa que sea correcta y que "
            f"tenga saldo/permisos."
        )
    if any(k in text for k in ("quota", "rate", "429", "limit")):
        return f"{provider_name} reporta límite de uso o cuota. Inténtalo más tarde."
    if any(k in text for k in ("connection", "network", "timeout", "getaddrinfo")):
        return f"No se pudo conectar con {provider_name}. Revisa tu conexión a internet."
    return f"Error de {provider_name}: {exc}"
