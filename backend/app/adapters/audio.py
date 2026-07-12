"""Audio normalization helpers.

Whisper expects 16 kHz mono PCM. To support arbitrary user input — any of
the audio or video extensions in :mod:`app.core.config` — we shell out to
the ffmpeg binary that ships with ``imageio-ffmpeg``. That binary is a
pip dependency, so the user does not have to install ffmpeg system-wide.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg

from app.core import config

logger = logging.getLogger(__name__)


class AudioConversionError(RuntimeError):
    """Raised when ffmpeg fails to convert the source media to WAV."""


def _ffmpeg_exe() -> str:
    """Return the path to the bundled ffmpeg binary.

    Raises:
        AudioConversionError: if ``imageio-ffmpeg`` cannot resolve a binary
            (e.g. the wheel did not download the platform binary).
    """
    try:
        path = str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # pragma: no cover - very rare
        raise AudioConversionError(
            "No se encontró el binario de ffmpeg empaquetado. "
            "Reinstala `imageio-ffmpeg` y vuelve a intentarlo."
        ) from exc
    if not path or not Path(path).exists():
        raise AudioConversionError(
            "El binario de ffmpeg no existe en disco. Reinstala `imageio-ffmpeg`."
        )
    return path


def get_duration_seconds(wav_path: Path) -> float:
    """Return the duration in seconds of a 16-bit PCM WAV file.

    Args:
        wav_path: Path to a WAV file produced by :func:`to_wav_16k`.

    Returns:
        Duration in seconds, or ``0.0`` if the file cannot be read.
    """
    try:
        with wave.open(str(wav_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate() or config.TARGET_SAMPLE_RATE
            return frames / float(rate)
    except (wave.Error, OSError):
        logger.exception("No se pudo leer la duración de %s", wav_path)
        return 0.0


def to_wav_16k(source: Path, *, dest_dir: Path | None = None) -> Path:
    """Convert any supported media file to 16 kHz mono WAV.

    The output is written to a temporary directory (``dest_dir`` if given,
    otherwise a fresh ``tempfile`` directory). The caller is responsible
    for deleting the resulting file (or the parent directory) when done.

    Args:
        source: Path to the input audio or video file. Must exist.
        dest_dir: Optional directory to place the output WAV in. Created
            if it does not exist.

    Returns:
        Path to the produced WAV file.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        AudioConversionError: If ffmpeg exits with a non-zero status.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {source}")

    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix=config.TEMP_PREFIX))
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / (source.stem + config.WAV_SUFFIX)
    cmd = [
        _ffmpeg_exe(),
        "-y",  # overwrite
        "-i",
        str(source),
        "-vn",  # drop video stream if any
        "-ac",
        str(config.TARGET_CHANNELS),
        "-ar",
        str(config.TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        "-loglevel",
        "error",
        str(dest),
    ]
    logger.debug("Running ffmpeg: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise AudioConversionError(
            f"No se pudo ejecutar ffmpeg: {exc}. Comprueba la instalación de imageio-ffmpeg."
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        logger.error("ffmpeg fallo (rc=%s): %s", result.returncode, stderr)
        raise AudioConversionError(
            f"ffmpeg no pudo convertir el archivo (código {result.returncode}). "
            f"Detalle: {stderr[:500] if stderr else 'sin salida'}"
        )

    if not dest.exists():
        raise AudioConversionError(
            "ffmpeg terminó sin error pero no se generó el archivo WAV de salida."
        )
    logger.info("Convertido %s → %s", source.name, dest)
    return dest


def save_compressed_copy(wav: Path, dest: Path) -> bool:
    """Save a listening copy of ``wav`` at ``dest`` (AAC ~48 kbps in .m4a).

    Used to keep each meeting's audio in the local library without WAV-sized
    files (a 1 h WAV is ~115 MB; the AAC copy is ~20 MB). Falls back to a
    plain WAV copy if encoding fails. Returns ``True`` when a file was saved.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(wav),
        "-c:a",
        "aac",
        "-b:a",
        "48k",
        "-loglevel",
        "error",
        str(dest),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0 and dest.exists():
            return True
        logger.warning("ffmpeg no pudo comprimir el audio (rc=%s)", result.returncode)
    except OSError:
        logger.warning("No se pudo ejecutar ffmpeg para comprimir", exc_info=True)
    try:  # fallback: keep the raw WAV rather than losing the audio
        shutil.copyfile(wav, dest.with_suffix(config.WAV_SUFFIX))
        return True
    except OSError:
        logger.exception("No se pudo guardar la copia de audio")
        return False


def cleanup(path: Path) -> None:
    """Best-effort removal of a temporary file or directory.

    Never raises; failures are logged at debug level. Use this from a
    ``finally`` block to clean up after a transcription run.
    """
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
    except OSError:
        logger.debug("No se pudo limpiar %s", path, exc_info=True)
