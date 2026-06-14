"""AI chat endpoint: stream an answer over the transcript (bring-your-own-key).

Stateless: the browser sends the API key, the conversation history and the
timestamped transcript on every request. Nothing is persisted server-side.
The blocking provider generator is run by Starlette in a threadpool, so it
does not block the event loop.

PRIVACY: this is the only endpoint that sends data to a third party — the
transcript text goes to the chosen provider with the user's key. The audio
never leaves the machine. See AGENTS.md §3.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import prompts
from app.adapters import llm
from app.adapters.llm import ChatMessage, LLMError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class _Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Body for POST /api/chat."""

    provider: Literal["openai", "gemini"]
    model: str
    api_key: str
    transcript_timestamped: str
    language: str = "es"
    duration: float = 0.0
    history: list[_Turn] = Field(default_factory=list)


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Stream the assistant reply as SSE (`data: {"delta": "..."}`)."""
    system = (
        prompts.SYSTEM_PROMPT
        + "\n\n"
        + prompts.build_context(
            req.transcript_timestamped,
            language=req.language,
            duration_seconds=req.duration,
        )
    )
    messages = [ChatMessage(role=t.role, content=t.content) for t in req.history]

    def event_stream() -> Iterator[str]:
        try:
            for piece in llm.stream_chat(
                provider=req.provider,
                api_key=req.api_key,
                model=req.model,
                system=system,
                messages=messages,
            ):
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
        except LLMError as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
