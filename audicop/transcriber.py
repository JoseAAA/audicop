"""Thin wrapper over `faster_whisper.WhisperModel`.

The wrapper has two responsibilities:

1. Lazily load (and cache) the underlying ``WhisperModel`` so we do not
   reload weights on every Streamlit rerun.
2. Yield decoded segments incrementally so the UI can stream them and
   update the progress bar based on ``segment.end / total_duration``.

It deliberately knows nothing about Streamlit; the UI layer adapts the
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

from audicop import config

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
from faster_whisper import WhisperModel  # noqa: E402


def _huggingface_cache_root() -> Path:
    """Return the HuggingFace Hub cache directory for the current user."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_model_cached(model_size: str) -> bool:
    """Return ``True`` if the given Whisper model is already cached locally.

    A best-effort check: we look for the conventional HuggingFace cache
    path produced by ``Systran/faster-whisper-<size>``. False negatives
    are harmless (the user just sees the "downloading" notice for a
    model that turns out to load instantly).
    """
    cache_dir = _huggingface_cache_root() / f"{config.HF_REPO_PREFIX}{model_size}"
    return cache_dir.is_dir() and any(cache_dir.iterdir())


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

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load(self) -> WhisperModel:
        """Load the underlying ``WhisperModel`` if not already loaded.

        Returns:
            The cached :class:`WhisperModel` instance.

        Raises:
            TranscriptionError: If the model cannot be downloaded or loaded.
        """
        if self._model is not None:
            return self._model

        logger.info(
            "Cargando WhisperModel size=%s compute=%s device=%s",
            self.model_size,
            self.compute_type,
            self.device,
        )
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self._download_root,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"No se pudo cargar el modelo {self.model_size!r} "
                f"({self.compute_type}) en {self.device}: {exc}"
            ) from exc
        return self._model

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
    ) -> tuple[Iterator[TranscriptSegment], TranscriptionInfo]:
        """Transcribe a 16 kHz mono WAV file.

        The function returns immediately with a generator over decoded
        segments and a :class:`TranscriptionInfo` describing the audio.
        Iterating the generator drives the actual decoding.

        Args:
            wav_path: Path to the prepared WAV file.
            language: Two-letter language code, or ``None`` to autodetect.
            task: ``"transcribe"`` or ``"translate"`` (translate to English).
            beam_size: Beam search width passed to faster-whisper.
            vad_filter: Whether to apply Silero VAD before decoding.

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
        try:
            segments, info = model.transcribe(
                str(wav_path),
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
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
