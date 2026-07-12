"""AI chat endpoint: stream an answer over the transcript — 100% on-device.

Stateless: the browser sends the conversation history and the timestamped
transcript on every request. Nothing is persisted server-side. The blocking
llama.cpp generator is run by Starlette in a threadpool, so it does not block
the event loop.

PRIVACY: the analysis runs entirely on the user's machine via a local model
(llama.cpp). Nothing — not the audio, not the transcript text — leaves the
device. There are no cloud providers and no API keys. See AGENTS.md §3.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import prompts
from app.adapters import local_llm
from app.adapters.hardware import detect_hardware
from app.adapters.local_llm import ChatMessage, LLMError, LocalLLM
from app.core import config
from app.services import meeting_store, summarize, transcript_store
from app.services.citations import CitationFixer
from app.services.recommender import recommend_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# Condensed-notes cache for long audios, keyed by transcript hash. The map
# phase costs minutes of LLM work and depends only on the transcript, so it
# runs ONCE per audio: every later question reuses the notes instantly (and
# they are also persisted with the meeting, surviving restarts).
_NOTES_CACHE: dict[str, str] = {}


class _Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Body for POST /api/chat. The analysis runs entirely on-device.

    ``transcript_timestamped`` may be empty: the endpoint then falls back to
    the transcript stored server-side by the last finished transcription (see
    :mod:`app.services.transcript_store`).
    """

    transcript_timestamped: str = ""
    language: str = "es"
    duration: float = 0.0
    meeting_id: str = ""
    history: list[_Turn] = Field(default_factory=list)


def get_ready_model() -> LocalLLM:
    """Return the on-device model, reusing the loaded one when present.

    If a model is already loaded, reuse it directly: re-measuring free memory
    at that point would count the memory the loaded model (and Whisper) already
    consume and wrongly reject a model that is literally running. Hardware is
    only (re-)detected when nothing is loaded yet. Raises :class:`LLMError`
    when no local model fits the machine, surfaced by the endpoint as an SSE
    error event.
    """
    model = local_llm.get_active()
    if model is not None:
        return model
    choice = recommend_llm(detect_hardware(), gpu_offload=local_llm.supports_gpu_offload())
    if not choice.available:
        raise LLMError(choice.rationale)
    return local_llm.get_local_llm(
        repo_id=choice.repo_id,
        filename=choice.filename,
        device=choice.device,
        n_gpu_layers=choice.n_gpu_layers,
        n_ctx=choice.n_ctx,
    )


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Stream the assistant reply as SSE (`data: {"delta": "..."}`)."""
    transcript = req.transcript_timestamped
    language = req.language
    duration = req.duration
    if not transcript.strip():
        stored = transcript_store.get()
        if stored is not None:
            transcript = stored.timestamped
            language = stored.language
            duration = stored.duration

    messages = [ChatMessage(role=t.role, content=t.content) for t in req.history]

    def event_stream() -> Iterator[str]:
        try:
            if not transcript.strip():
                raise LLMError("No hay ninguna transcripción todavía. Transcribe un audio primero.")
            model = get_ready_model()

            # Long audio → RECURSIVE map-reduce: condense chunk by chunk and,
            # if the joined notes still exceed the budget (a 2-3 h meeting
            # easily does), condense the notes again. Without this second
            # round the final pass receives dozens of notes and echoes them
            # instead of synthesizing. Marks stay valid: chunks are cut at
            # line boundaries so every note carries an original [MM:SS].
            working = transcript
            cache_key = ""
            if summarize.needs_map_reduce(transcript):
                # Reuse notes condensed earlier for this same transcript —
                # in-memory first, then the copy persisted with the meeting.
                cache_key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                cached = _NOTES_CACHE.get(cache_key) or (
                    meeting_store.get_condensed(req.meeting_id) if req.meeting_id else ""
                )
                if cached:
                    working = cached
            round_no = 0
            while (
                cache_key  # only long audios (above the map-reduce threshold)
                and summarize.estimate_tokens(working) > config.LLM_SYNTHESIS_TARGET_TOKENS
                and round_no < 4
            ):
                round_no += 1
                chunks = summarize.chunk_by_lines(
                    working,
                    max_tokens=config.LLM_MAPREDUCE_TOKEN_THRESHOLD
                    - config.LLM_CHUNK_PROMPT_OVERHEAD_TOKENS,
                    overlap_tokens=config.LLM_CHUNK_OVERLAP_TOKENS,
                )
                notes: list[str] = []
                for i, chunk in enumerate(chunks, start=1):
                    progress = {
                        "phase": "map",
                        "current": i,
                        "total": len(chunks),
                        "round": round_no,
                    }
                    yield f"data: {json.dumps(progress)}\n\n"
                    part = "".join(
                        model.stream_chat(
                            system=prompts.SYSTEM_PROMPT,
                            messages=[
                                ChatMessage(
                                    role="user",
                                    content=(
                                        f"{summarize.map_instructions()}\n\n"
                                        f"=== PARTE {i}/{len(chunks)} ===\n{chunk}"
                                    ),
                                )
                            ],
                            strict=True,  # condensation is format-critical
                        )
                    ).strip()
                    if part:
                        notes.append(part)
                if not notes:
                    raise LLMError("No se pudo condensar el audio largo. Inténtalo de nuevo.")
                joined = "\n".join(notes)
                shrunk_enough = len(joined) < 0.85 * len(working)
                working = joined
                if not shrunk_enough:
                    break  # marginal shrink — more rounds just burn minutes
            if round_no > 0:
                # COMBINE (meetily's trick): rewrite the notes as narrative
                # prose. Note-shaped context makes small models echo the notes
                # verbatim ("60 key points"); prose forces real synthesis.
                yield f"data: {json.dumps({'phase': 'combine'})}\n\n"
                narrative = "".join(
                    model.stream_chat(
                        system=prompts.SYSTEM_PROMPT,
                        messages=[
                            ChatMessage(
                                role="user",
                                content=(
                                    f"{summarize.combine_instructions()}\n\n"
                                    f"<notas>\n{working}\n</notas>"
                                ),
                            )
                        ],
                        strict=True,
                    )
                ).strip()
                if narrative:
                    working = narrative
            if round_no > 0 and cache_key:
                # Condensation ran: remember it so the next question is instant.
                _NOTES_CACHE[cache_key] = working
                if req.meeting_id:
                    meeting_store.save_condensed(req.meeting_id, working)

            system = (
                prompts.SYSTEM_PROMPT
                + "\n\n"
                + prompts.build_context(
                    working,
                    language=language,
                    duration_seconds=duration,
                )
            )
            if working is not transcript:
                system += (
                    "\n\nIMPORTANTE: lo anterior es un RESUMEN CONDENSADO de "
                    "una reunión larga, no la transcripción completa. NO lo "
                    "copies: SINTETIZA y selecciona solo lo más importante de "
                    "toda la reunión, con sus marcas [MM:SS]."
                )
            # Quick actions come from prompt templates that all start with
            # "TAREA:" and define an exact format → strict sampling. Free
            # questions keep the regular preset.
            strict = bool(messages) and messages[-1].content.startswith("TAREA:")
            # Verify citations against the real (FULL) transcript as lines
            # complete: small models misattribute [MM:SS] marks (or drop
            # them), and the server holds the ground truth to fix that.
            fixer = CitationFixer(transcript)
            for piece in model.stream_chat(system=system, messages=messages, strict=strict):
                for fixed in fixer.feed(piece):
                    yield f"data: {json.dumps({'delta': fixed}, ensure_ascii=False)}\n\n"
            for fixed in fixer.flush():
                yield f"data: {json.dumps({'delta': fixed}, ensure_ascii=False)}\n\n"
        except LLMError as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
