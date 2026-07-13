"""Local, in-process LLM via llama.cpp (``llama-cpp-python``).

Audicop's AI analysis runs **100% on-device**: a small quantized model (GGUF)
loaded with llama.cpp, chosen for the machine by
:func:`app.services.recommender.recommend_llm`. There are no cloud providers
and no API keys — neither the audio nor the transcript text leaves the device.

``llama_cpp`` is imported lazily inside :meth:`LocalLLM.load` (and
:func:`supports_gpu_offload`) so a missing or partial install degrades to a
clear message instead of crashing the app at import time. The launcher installs
the right wheel per hardware (CPU or CUDA) — see ``scripts/start.*`` and
AGENTS.md §7.

Responses are streamed (an iterator of text chunks) so the UI can render them
token-by-token.

PRIVACY: nothing here makes a network call except the **one-time** model
download from HuggingFace on first use. Inference is fully local.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from app.adapters.cuda_dll import ensure_cuda_dll_search_path
from app.core import config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the local model fails to download, load, or generate."""


# STATUS_ILLEGAL_INSTRUCTION: ctypes surfaces it as this OSError.winerror on
# Windows; POSIX raises SIGILL / "Illegal instruction". It means the installed
# llama.cpp wheel uses a CPU instruction (e.g. AVX-512) this machine lacks.
_ILLEGAL_INSTRUCTION_WINERROR = -1073741795  # 0xC000001D


def _describe_load_failure(filename: str, exc: Exception) -> str:
    """Turn a model-load exception into an actionable, user-facing message.

    The one failure worth special-casing is an *illegal instruction* crash:
    the prebuilt engine was compiled for CPU features this processor does not
    have. Rather than a raw hex code, tell the user what to do and reassure
    them the rest of Audicop (transcription) still works.
    """
    text = str(exc)
    if getattr(exc, "winerror", None) == _ILLEGAL_INSTRUCTION_WINERROR or any(
        token in text.lower() for token in ("0xc000001d", "-1073741795", "illegal instruction")
    ):
        return (
            "Tu procesador no es compatible con el motor de IA que se instaló "
            "(usa instrucciones de CPU que este equipo no tiene). Cierra Audicop "
            "y vuelve a lanzarlo con `start.cmd` (Windows) o `./scripts/start.sh`: "
            "el instalador pondrá una versión compatible. La transcripción sigue "
            "funcionando; solo el análisis con IA local queda deshabilitado."
        )
    return f"No se pudo cargar el modelo local {filename!r}: {exc}"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single chat turn.

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        content: The message text.
    """

    role: Literal["user", "assistant"]
    content: str


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def strip_think_blocks(pieces: Iterator[str]) -> Iterator[str]:
    """Drop ``<think>…</think>`` reasoning blocks from a token stream.

    Reasoning-tuned models (Qwen3 and friends) prepend their chain of thought
    inside think tags; users should only ever see the answer. Works across
    chunk boundaries: a tag split between two streamed pieces is still caught
    by holding back the longest possible partial-tag suffix.
    """
    buf = ""
    in_think = False
    for piece in pieces:
        buf += piece
        out = ""
        while buf:
            if in_think:
                end = buf.find(_THINK_CLOSE)
                if end == -1:
                    buf = buf[-(len(_THINK_CLOSE) - 1) :]  # may hold a partial close tag
                    break
                buf = buf[end + len(_THINK_CLOSE) :]
                in_think = False
            else:
                start = buf.find(_THINK_OPEN)
                if start == -1:
                    hold = 0  # keep a suffix that could be the start of an open tag
                    for k in range(min(len(_THINK_OPEN) - 1, len(buf)), 0, -1):
                        if buf.endswith(_THINK_OPEN[:k]):
                            hold = k
                            break
                    out += buf[: len(buf) - hold]
                    buf = buf[len(buf) - hold :]
                    break
                out += buf[:start]
                buf = buf[start + len(_THINK_OPEN) :]
                in_think = True
        if out:
            yield out
    if buf and not in_think:
        yield buf


_GPU_OFFLOAD: bool | None = None


def supports_gpu_offload() -> bool:
    """Return ``True`` if the installed llama.cpp build can offload to a GPU.

    The default wheel is CPU-only, so even on a CUDA machine we must not size
    the model against VRAM nor pass ``n_gpu_layers``: it would silently run on
    CPU while the recommender claimed GPU. Returns ``False`` when ``llama_cpp``
    is missing or built without GPU support. Probed once and cached.
    """
    global _GPU_OFFLOAD
    if _GPU_OFFLOAD is None:
        try:
            ensure_cuda_dll_search_path()
            import llama_cpp

            fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
            _GPU_OFFLOAD = bool(fn()) if callable(fn) else False
        except Exception:  # pragma: no cover - defensive (missing/broken build)
            _GPU_OFFLOAD = False
    return _GPU_OFFLOAD


def _gguf_cache_dir() -> Path:
    """Return Audicop's directory for downloaded GGUF models (plain files)."""
    return Path.home() / config.AUDICOP_CACHE_SUBPATH / "gguf"


def is_model_cached(filename: str) -> bool:
    """Return ``True`` if the given GGUF file is already downloaded locally."""
    path = _gguf_cache_dir() / filename
    return path.is_file() and path.stat().st_size > 0


def ensure_downloaded(repo_id: str, filename: str) -> Path:
    """Download the GGUF to Audicop's cache if missing; return its path.

    Uses ``hf_hub_download`` with ``local_dir`` so the file lands as a plain
    copy (no symlinks) — the same Windows-safe approach the transcriber uses
    for Whisper weights. Idempotent: a present, non-empty file is reused, not
    re-fetched.

    Args:
        repo_id: HuggingFace repo hosting the GGUF.
        filename: GGUF filename within the repo.

    Returns:
        Absolute path to the local GGUF file.

    Raises:
        LLMError: If the download fails (network, missing file, etc.).
    """
    target_dir = _gguf_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / filename
    if final_path.is_file() and final_path.stat().st_size > 0:
        return final_path
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
        )
    except Exception as exc:
        raise LLMError(
            f"No se pudo descargar el modelo local {filename!r} desde {repo_id!r}: {exc}"
        ) from exc
    return Path(downloaded)


class LocalLLM:
    """A lazily-loaded local model backed by ``llama_cpp.Llama``.

    Construct it from an :class:`app.services.recommender.LlmChoice` (or the
    equivalent fields). The model is downloaded on first use and the loaded
    instance is reused thereafter; call :meth:`unload` to free its memory
    before loading a different model — for example to release the VRAM the
    transcriber used before summarizing.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        filename: str,
        device: str = "cpu",
        n_gpu_layers: int = 0,
        n_ctx: int = config.LLM_DEFAULT_N_CTX,
    ) -> None:
        """Initialize the wrapper (no download or load happens here).

        Args:
            repo_id: HuggingFace repo hosting the GGUF.
            filename: GGUF filename within the repo.
            device: ``"cuda"`` or ``"cpu"`` (informational; offload is driven
                by ``n_gpu_layers``).
            n_gpu_layers: Layers to offload to GPU (``-1`` = all, ``0`` = none).
            n_ctx: Context window in tokens.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.device = device
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._model: Llama | None = None

    def load(self) -> Llama:
        """Download (if needed) and load the model, returning the ``Llama``.

        Returns:
            The cached :class:`llama_cpp.Llama` instance.

        Raises:
            LLMError: If ``llama_cpp`` is not installed, or the model fails to
                download or load.
        """
        if self._model is not None:
            return self._model
        ensure_cuda_dll_search_path()  # llama.cpp's CUDA build needs the NVIDIA DLLs on PATH
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise LLMError(
                "El motor de IA local no está instalado. Cierra la app y vuelve a "
                "lanzarla con `start.cmd` (Windows) o `./scripts/start.sh`: el "
                "instalador lo añade automáticamente."
            ) from exc

        path = ensure_downloaded(self.repo_id, self.filename)
        # Physical cores only: hyperthreads hurt llama.cpp's compute-bound
        # matmuls, and saturating every logical core starves the OS/UI.
        import psutil

        threads = max(1, psutil.cpu_count(logical=False) or 1)
        logger.info(
            "Cargando LLM local %s (n_gpu_layers=%s, n_ctx=%s, n_threads=%s)",
            self.filename,
            self.n_gpu_layers,
            self.n_ctx,
            threads,
        )
        try:
            self._model = Llama(
                model_path=str(path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=threads,
                verbose=False,
            )
        except Exception as exc:
            raise LLMError(_describe_load_failure(self.filename, exc)) from exc
        return self._model

    def unload(self) -> None:
        """Release the loaded model so its memory can be reclaimed.

        Closes the llama.cpp context explicitly (freeing VRAM/RAM right away
        instead of waiting for garbage collection) — used when a transcription
        starts and Whisper needs the memory.
        """
        import gc

        if self._model is not None:
            close = getattr(self._model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - defensive
                    logger.debug("close() del modelo local falló", exc_info=True)
        self._model = None
        gc.collect()

    def stream_chat(
        self,
        *,
        system: str,
        messages: Sequence[ChatMessage],
        strict: bool = False,
    ) -> Iterator[str]:
        """Stream an assistant reply token-by-token, fully on-device.

        Args:
            system: System prompt establishing the assistant's behavior.
            messages: Conversation so far, oldest first.
            strict: Use the tighter sampling preset for format-critical
                outputs (quick actions / map phase). At the default
                temperature small models sometimes drift into copying the
                transcript verbatim instead of following the format.

        Yields:
            Text chunks of the assistant's reply, in order.

        Raises:
            LLMError: If the model cannot be loaded or generation fails.
        """
        model = self.load()
        payload: list[dict[str, str]] = [{"role": "system", "content": system}]
        payload.extend({"role": m.role, "content": m.content} for m in messages)
        try:
            # llama_cpp types `messages` as a union of TypedDicts; plain
            # role/content dicts are valid at runtime (the cast keeps mypy happy).
            # Sampling presets tuned for Qwen-style summarizers (see config).
            stream = model.create_chat_completion(
                messages=cast("list[Any]", payload),
                stream=True,
                temperature=(config.LLM_TEMPERATURE_STRICT if strict else config.LLM_TEMPERATURE),
                max_tokens=config.LLM_MAX_TOKENS,
                top_k=config.LLM_TOP_K,
                top_p=config.LLM_TOP_P_STRICT if strict else config.LLM_TOP_P,
                presence_penalty=config.LLM_PRESENCE_PENALTY,
                repeat_penalty=config.LLM_REPEAT_PENALTY,
            )

            def pieces() -> Iterator[str]:
                for chunk in stream:
                    if isinstance(chunk, str):  # streamed type is a union; skip non-dict
                        continue
                    choices = chunk.get("choices")
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    piece = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(piece, str) and piece:
                        yield piece

            # Reasoning models leak <think> blocks into the stream; hide them.
            yield from strip_think_blocks(pieces())
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Error generando con el modelo local: {exc}") from exc


_INSTANCE: LocalLLM | None = None


def unload_active() -> None:
    """Free the loaded model's memory, keeping the configured instance.

    Called when a transcription starts: Whisper is about to need the memory,
    and the phases are sequential (they never share it). The next chat request
    re-detects hardware and reloads the best-fitting model.
    """
    if _INSTANCE is not None:
        _INSTANCE.unload()


def get_active() -> LocalLLM | None:
    """Return the process-wide model that is already loaded in memory, if any.

    The chat endpoint uses this to keep serving with the model that is already
    up: re-running the hardware recommendation after the LLM (and Whisper) are
    loaded would measure the memory THEY consume and wrongly conclude the
    machine can no longer fit a model — rejecting a model that is literally
    already running.
    """
    if _INSTANCE is not None and _INSTANCE._model is not None:
        return _INSTANCE
    return None


def get_local_llm(
    *,
    repo_id: str,
    filename: str,
    device: str,
    n_gpu_layers: int,
    n_ctx: int,
) -> LocalLLM:
    """Return a process-wide :class:`LocalLLM`, reloading only when it changes.

    Caches a single instance so the (large) weights are not reloaded on every
    request. If a different model / offload / context is requested, the
    previous instance is unloaded first to free its memory.

    Args:
        repo_id: HuggingFace repo hosting the GGUF.
        filename: GGUF filename within the repo.
        device: ``"cuda"`` or ``"cpu"``.
        n_gpu_layers: Layers to offload to GPU (``-1`` = all, ``0`` = none).
        n_ctx: Context window in tokens.

    Returns:
        A ready-to-use :class:`LocalLLM` (not yet loaded into memory).
    """
    global _INSTANCE
    if (
        _INSTANCE is None
        or _INSTANCE.filename != filename
        or _INSTANCE.n_gpu_layers != n_gpu_layers
        or _INSTANCE.n_ctx != n_ctx
    ):
        if _INSTANCE is not None:
            _INSTANCE.unload()
        _INSTANCE = LocalLLM(
            repo_id=repo_id,
            filename=filename,
            device=device,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
        )
    return _INSTANCE
