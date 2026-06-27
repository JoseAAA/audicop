"""Thin wrapper over `faster_whisper.WhisperModel`.

The wrapper has two responsibilities:

1. Lazily load the underlying ``WhisperModel`` (the pipeline caches the
   instance so weights are not reloaded on every request).
2. Yield decoded segments incrementally so the pipeline can stream them
   and report progress based on ``segment.end / total_duration``.

It deliberately knows nothing about the web layer; the pipeline adapts the
generator returned by :meth:`Transcriber.transcribe`.

On Windows we patch the DLL search path before importing
``faster_whisper`` so that ctranslate2 finds the cuBLAS / cuDNN runtime
DLLs that ship with the ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12``
PyPI packages. Without this, you get
``cublas64_12.dll is not found or cannot be loaded`` errors at decode
time even though the libraries are installed in the venv.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.core import config

logger = logging.getLogger(__name__)


def _ensure_cuda_dll_search_path() -> None:
    """Make the bundled NVIDIA DLLs discoverable on Windows.

    The faster-whisper docs explicitly state that the
    ``pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`` route is
    "Linux only" — on Linux, ``LD_LIBRARY_PATH`` is auto-populated, but
    on Windows the DLL loader is stricter:

    - Python 3.8+ no longer searches ``PATH`` for ``.pyd`` dependencies.
    - :func:`os.add_dll_directory` only helps when the loader uses the
      ``LOAD_LIBRARY_SEARCH_USER_DIRS`` flag.
    - ctranslate2's compiled extension calls plain ``LoadLibraryW`` from
      C++ at decode time (when it actually needs cuBLAS), and that call
      uses the legacy "default search order" — which IS controlled by
      ``PATH``.

    So we do BOTH:

    1. ``os.add_dll_directory(bin)`` — covers Python-side imports.
    2. Prepend ``bin`` to ``os.environ["PATH"]`` — covers the runtime
       ``LoadLibrary`` calls from ctranslate2.

    No-op on non-Windows systems. Idempotent: appending the same dir
    twice is harmless because Windows deduplicates the search path.
    """
    if sys.platform != "win32":
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        logger.debug("nvidia/ not found under site-packages; skipping DLL setup.")
        return

    bin_dirs: list[str] = []
    for sub in nvidia_root.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            bin_dirs.append(str(bin_dir))

    if not bin_dirs:
        return

    for path_str in bin_dirs:
        try:
            os.add_dll_directory(path_str)
        except (OSError, FileNotFoundError):
            logger.debug("os.add_dll_directory(%s) failed", path_str, exc_info=True)

    existing_path = os.environ.get("PATH", "")
    new_path = os.pathsep.join(bin_dirs)
    if existing_path:
        new_path = new_path + os.pathsep + existing_path
    os.environ["PATH"] = new_path
    logger.debug("Prepended NVIDIA bin dirs to PATH: %s", bin_dirs)


_ensure_cuda_dll_search_path()

# Imported AFTER the DLL path setup above; re-ordering this would break
# ctranslate2 on Windows when the CUDA libs come from PyPI wheels.
from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402


def _repo_id_for(model_size: str) -> str:
    """Return the HuggingFace repo id for a faster-whisper model size.

    Prefers faster-whisper's own size→repo mapping so that models hosted
    outside the ``Systran`` org — notably ``large-v3-turbo``, which lives
    under ``mobiuslabsgmbh`` — resolve to the correct repository. Falls
    back to :data:`config.FASTER_WHISPER_REPO_TEMPLATE` if that (private)
    mapping is ever unavailable.
    """
    try:
        from faster_whisper.utils import _MODELS

        repo = _MODELS.get(model_size)
        if repo:
            return str(repo)
    except Exception:  # pragma: no cover - defensive against upstream API changes
        logger.debug("faster_whisper _MODELS unavailable; using repo template", exc_info=True)
    return config.FASTER_WHISPER_REPO_TEMPLATE.format(size=model_size)


def _huggingface_cache_root() -> Path:
    """Return the HuggingFace Hub cache directory for the current user."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _audicop_cache_root() -> Path:
    """Return Audicop's private models directory (no symlinks, plain copies)."""
    return Path.home() / config.AUDICOP_CACHE_SUBPATH


def _audicop_model_dir(model_size: str) -> Path:
    """Return the directory where a model lives in Audicop's private cache.

    The directory name mirrors the model's HuggingFace repo id (``/``
    replaced by ``--``), so each model — whatever org hosts it — gets its
    own folder and there are no collisions.
    """
    return _audicop_cache_root() / _repo_id_for(model_size).replace("/", "--")


_SYMLINK_PROBE_RESULT: bool | None = None


def _hf_symlinks_supported() -> bool:
    """Probe whether we can create symlinks in HF Hub's cache directory.

    Runs the test exactly once per process and caches the result. On
    Linux/macOS this returns ``True`` immediately. On Windows it tries
    to create a real symlink in the HF cache root; success means the
    user has either admin, Developer Mode enabled, or the
    ``SeCreateSymbolicLinkPrivilege`` granted by group policy.

    When this returns ``False`` we route model downloads through
    :func:`_download_model_no_symlinks` instead, which produces a plain
    directory of copied files — no symlinks anywhere.
    """
    global _SYMLINK_PROBE_RESULT
    if _SYMLINK_PROBE_RESULT is not None:
        return _SYMLINK_PROBE_RESULT
    if sys.platform != "win32":
        _SYMLINK_PROBE_RESULT = True
        return True

    cache_root = _huggingface_cache_root()
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("Could not create HF cache root", exc_info=True)
        _SYMLINK_PROBE_RESULT = False
        return False

    src = cache_root / ".audicop_symlink_probe_src"
    dst = cache_root / ".audicop_symlink_probe_dst"
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        src.touch(exist_ok=True)
        os.symlink(str(src), str(dst))
        _SYMLINK_PROBE_RESULT = True
    except OSError as exc:
        logger.info(
            "Symlinks not available in %s (%s); using copy-only model cache.",
            cache_root,
            exc,
        )
        _SYMLINK_PROBE_RESULT = False
    finally:
        for p in (dst, src):
            try:
                if p.is_symlink() or p.exists():
                    p.unlink()
            except OSError:
                pass
    return _SYMLINK_PROBE_RESULT


def is_model_cached(model_size: str) -> bool:
    """Return ``True`` if the given Whisper model is already cached locally.

    Checks both the standard HuggingFace Hub cache and Audicop's private
    no-symlink cache (used as fallback on restricted Windows accounts).
    False negatives are harmless: the user just sees the "downloading"
    notice for a model that turns out to load instantly.
    """
    repo_dirname = "models--" + _repo_id_for(model_size).replace("/", "--")
    hf_dir = _huggingface_cache_root() / repo_dirname
    if hf_dir.is_dir() and any(hf_dir.iterdir()):
        return True
    audicop_dir = _audicop_model_dir(model_size)
    return audicop_dir.is_dir() and (audicop_dir / "model.bin").is_file()


def _download_model_no_symlinks(model_size: str) -> Path:
    """Download a faster-whisper model as plain copies (no symlinks).

    Used when the standard HF cache path can't work — typically on
    Windows accounts without symlink privilege. Idempotent: if the
    target dir already has the model files, ``snapshot_download``
    skips the work.

    Raises:
        TranscriptionError: If the download fails for any reason.
    """
    from huggingface_hub import snapshot_download

    target = _audicop_model_dir(model_size)
    target.mkdir(parents=True, exist_ok=True)
    repo_id = _repo_id_for(model_size)
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(target))
    except Exception as exc:
        raise TranscriptionError(
            f"No se pudo descargar el modelo {model_size!r} a la caché local ({target}): {exc}"
        ) from exc
    return target


class TranscriptionError(RuntimeError):
    """Raised when faster-whisper fails to load or decode a file."""


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A single decoded segment of speech.

    Attributes:
        start: Start timestamp in seconds.
        end: End timestamp in seconds.
        text: Decoded text (already stripped of leading/trailing spaces).
    """

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionInfo:
    """Metadata returned alongside the segment stream.

    Attributes:
        language: Language code detected (or forced) by Whisper.
        language_probability: Confidence score from the language detector.
        duration: Duration of the input audio in seconds.
    """

    language: str
    language_probability: float
    duration: float


class Transcriber:
    """Stateful wrapper around a single :class:`WhisperModel` instance.

    The model is loaded on first use and reused thereafter. Recreate the
    instance if you want to switch model size, compute type, or device.
    """

    def __init__(
        self,
        *,
        model_size: str,
        compute_type: str,
        device: str,
        download_root: Path | None = None,
    ) -> None:
        """Initialize a transcriber for a given model/device combination.

        Args:
            model_size: Whisper model identifier (e.g. ``"large-v3"``).
            compute_type: CTranslate2 compute type (e.g. ``"int8_float16"``).
            device: ``"cuda"`` or ``"cpu"``.
            download_root: Optional local directory used for the model
                cache. Defaults to faster-whisper's standard location.
        """
        if model_size not in config.VALID_MODEL_SIZES:
            raise ValueError(
                f"Tamaño de modelo no soportado: {model_size!r}. "
                f"Válidos: {config.VALID_MODEL_SIZES}"
            )
        if compute_type not in config.VALID_COMPUTE_TYPES:
            raise ValueError(
                f"compute_type no soportado: {compute_type!r}. "
                f"Válidos: {config.VALID_COMPUTE_TYPES}"
            )
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"device debe ser 'cuda' o 'cpu', no {device!r}")

        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self._download_root = str(download_root) if download_root is not None else None
        self._model: WhisperModel | None = None
        self._batched_pipeline: BatchedInferencePipeline | None = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load(self) -> WhisperModel:
        """Load the underlying ``WhisperModel`` if not already loaded.

        On platforms where HuggingFace's symlink-based cache can't work
        (typically restricted Windows accounts), we transparently fall
        back to a plain-files cache under
        ``~/.cache/audicop/models/`` — same model, no symlinks. Users
        never see WinError 1314 anymore.

        Returns:
            The cached :class:`WhisperModel` instance.

        Raises:
            TranscriptionError: If the model cannot be downloaded or loaded.
        """
        if self._model is not None:
            return self._model

        use_local_dir = not _hf_symlinks_supported()
        local_path: Path | None = None
        if use_local_dir:
            local_path = _download_model_no_symlinks(self.model_size)

        model_ref = str(local_path) if local_path is not None else self.model_size
        logger.info(
            "Cargando WhisperModel ref=%s compute=%s device=%s (symlinks=%s)",
            model_ref,
            self.compute_type,
            self.device,
            not use_local_dir,
        )
        try:
            self._model = WhisperModel(
                model_ref,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self._download_root if not use_local_dir else None,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"No se pudo cargar el modelo {self.model_size!r} "
                f"({self.compute_type}) en {self.device}: {exc}"
            ) from exc
        return self._model

    def _get_batched_pipeline(self, model: WhisperModel) -> BatchedInferencePipeline:
        """Return a cached :class:`BatchedInferencePipeline` over ``model``."""
        if self._batched_pipeline is None:
            self._batched_pipeline = BatchedInferencePipeline(model=model)
        return self._batched_pipeline

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def transcribe(
        self,
        wav_path: Path,
        *,
        language: str | None = None,
        task: str = "transcribe",
        beam_size: int = config.DEFAULT_BEAM_SIZE,
        vad_filter: bool = config.DEFAULT_VAD_FILTER,
        initial_prompt: str | None = None,
    ) -> tuple[Iterator[TranscriptSegment], TranscriptionInfo]:
        """Transcribe a 16 kHz mono WAV file.

        The function returns immediately with a generator over decoded
        segments and a :class:`TranscriptionInfo` describing the audio.
        Iterating the generator drives the actual decoding.

        On CUDA we route the decode through faster-whisper's
        :class:`BatchedInferencePipeline`, which processes several VAD
        chunks at once and is markedly faster than the sequential path.
        On CPU the batched pipeline brings little benefit, so we use the
        plain sequential decode.

        Args:
            wav_path: Path to the prepared WAV file.
            language: Two-letter language code, or ``None`` to autodetect.
            task: ``"transcribe"`` or ``"translate"`` (translate to English).
            beam_size: Beam search width passed to faster-whisper.
            vad_filter: Whether to apply Silero VAD before decoding.
            initial_prompt: Optional text (names, jargon, acronyms) that
                primes the decoder's vocabulary to improve accuracy on
                domain-specific audio. Empty/``None`` means no priming.

        Returns:
            Tuple ``(segments_iterator, info)``.

        Raises:
            FileNotFoundError: If ``wav_path`` does not exist.
            TranscriptionError: If decoding fails.
        """
        if task not in config.DEFAULT_TASKS:
            raise ValueError(f"task debe estar en {config.DEFAULT_TASKS}, no {task!r}")
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV de entrada no encontrado: {wav_path}")

        model = self.load()
        prompt = initial_prompt or None
        try:
            if self.device == "cuda":
                pipeline = self._get_batched_pipeline(model)
                segments, info = pipeline.transcribe(
                    str(wav_path),
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    initial_prompt=prompt,
                    batch_size=config.DEFAULT_BATCH_SIZE,
                )
            else:
                segments, info = model.transcribe(
                    str(wav_path),
                    language=language,
                    task=task,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    initial_prompt=prompt,
                )
        except Exception as exc:
            raise TranscriptionError(f"Fallo decodificando {wav_path.name}: {exc}") from exc

        meta = TranscriptionInfo(
            language=getattr(info, "language", "unknown"),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
        )

        def _stream() -> Iterator[TranscriptSegment]:
            for seg in segments:
                yield TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=str(seg.text).strip(),
                )

        return _stream(), meta
