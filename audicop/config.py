"""Project-wide constants for Audicop.

This module is the single source of truth for things like supported file
extensions, audio normalization parameters, and the table that drives
hardware-based model recommendation. Keeping these in one place avoids
sprinkling magic numbers across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS: Final[tuple[str, ...]] = ("mp3", "wav", "m4a", "ogg", "flac", "aac")
"""Audio file extensions Audicop accepts directly (lowercase, no dot)."""

VIDEO_EXTENSIONS: Final[tuple[str, ...]] = ("mp4", "mkv", "mov", "avi", "webm")
"""Video file extensions Audicop accepts (audio track is extracted)."""

SUPPORTED_EXTENSIONS: Final[tuple[str, ...]] = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS
"""All extensions accepted by the file uploader."""

MAX_UPLOAD_MB: Final[int] = 2000
"""Maximum upload size in megabytes (mirrors `.streamlit/config.toml`).

2 GB covers comfortably any audio of up to 3 hours plus moderate-bitrate
video. For larger files (HD video > 2 GB) the user should switch to the
"Archivo local" tab and paste the absolute path — Audicop reads the
file directly from disk in that case, no upload involved.
"""

MAX_DURATION_HOURS: Final[float] = 3.0
"""Soft limit on audio duration shown to the user.

Whisper itself processes audio with a 30-second sliding window, so VRAM
stays bounded regardless of total length — files longer than this still
work, we just warn that they exceed our tested envelope.
"""

LONG_FILE_THRESHOLD_S: Final[float] = 60 * 60.0
"""Duration (seconds) above which we warn the user about expected wall time."""

# Rough realtime factor (audio seconds processed per wall second). Used to
# show the user an *order-of-magnitude* time estimate. Real numbers depend
# on the specific CPU/GPU; these are calibrated from public benchmarks +
# our own measurements on consumer hardware. When in doubt, lean low.
REALTIME_FACTORS: Final[dict[tuple[str, str], float]] = {
    # CPU side (int8) — slow; small/medium/large barely usable on consumer CPUs.
    ("tiny", "cpu"): 5.0,
    ("base", "cpu"): 2.5,
    ("small", "cpu"): 1.0,
    ("medium", "cpu"): 0.4,
    ("large-v3", "cpu"): 0.15,
    # GPU side — measured on RTX 3060 Laptop, conservative for older cards.
    ("tiny", "cuda"): 30.0,
    ("base", "cuda"): 25.0,
    ("small", "cuda"): 18.0,
    ("medium", "cuda"): 10.0,
    ("large-v3", "cuda"): 6.0,
}


def estimate_processing_seconds(duration_s: float, model_size: str, device: str) -> float:
    """Return an order-of-magnitude estimate for how long transcription will take.

    Args:
        duration_s: Audio duration in seconds.
        model_size: Whisper model size (e.g. ``"large-v3"``).
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        Estimated wall-clock seconds. Falls back to a 1x factor if the
        combination isn't in the table (which only happens for unusual
        custom configurations).
    """
    factor = REALTIME_FACTORS.get((model_size, device), 1.0)
    if factor <= 0:
        return duration_s
    return duration_s / factor


# ---------------------------------------------------------------------------
# Audio normalization
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE: Final[int] = 16_000
"""Sample rate Whisper expects (Hz)."""

TARGET_CHANNELS: Final[int] = 1
"""Channel count Whisper expects (mono)."""

WAV_SUFFIX: Final[str] = ".wav"

# ---------------------------------------------------------------------------
# Whisper / faster-whisper
# ---------------------------------------------------------------------------

VALID_MODEL_SIZES: Final[tuple[str, ...]] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
)

VALID_COMPUTE_TYPES: Final[tuple[str, ...]] = (
    "int8",
    "int8_float16",
    "float16",
    "float32",
)

# Approximate download sizes for the faster-whisper models hosted at
# https://huggingface.co/Systran (CTranslate2 conversions of OpenAI Whisper).
# Used to set user expectations on first run.
MODEL_DOWNLOAD_SIZES_MB: Final[dict[str, int]] = {
    "tiny": 75,
    "base": 145,
    "small": 480,
    "medium": 1500,
    "large-v3": 3000,
}

# HuggingFace Hub names the cache directory after the repo. faster-whisper
# pulls from `Systran/faster-whisper-<size>`, which becomes:
#   ~/.cache/huggingface/hub/models--Systran--faster-whisper-<size>
HF_REPO_PREFIX: Final[str] = "models--Systran--faster-whisper-"

# Audicop's own model cache, used when the standard HF cache cannot be
# populated because Windows refuses symlink creation (WinError 1314 — see
# `transcriber._hf_symlinks_supported`). Files here are plain copies, no
# symlinks, so any user account on any Windows policy can read/write them.
AUDICOP_CACHE_SUBPATH: Final[str] = ".cache/audicop/models"

# faster-whisper repo template on HuggingFace.
FASTER_WHISPER_REPO_TEMPLATE: Final[str] = "Systran/faster-whisper-{size}"

DEFAULT_LANGUAGES: Final[tuple[str, ...]] = ("auto", "es", "en", "pt", "fr", "it", "de")
DEFAULT_TASKS: Final[tuple[str, ...]] = ("transcribe", "translate")
DEFAULT_BEAM_SIZE: Final[int] = 5
DEFAULT_VAD_FILTER: Final[bool] = True

# ---------------------------------------------------------------------------
# Hardware thresholds (used by the recommender)
#
# These thresholds intentionally describe FREE / AVAILABLE memory at the
# moment of detection — not total memory. A 16 GB laptop is rarely able to
# dedicate 16 GB to inference: the OS, browser, and other apps all hold
# RAM. Same story for VRAM when the display is attached to the same GPU.
# By reading the live "free" / "available" figures we adapt to whatever
# the user actually has free right now and stay polite to the rest of the
# system.
# ---------------------------------------------------------------------------

VRAM_FREE_HIGH_GB: Final[float] = 8.0
"""Free VRAM (GB) above which we can comfortably run large-v3 in float16.

float16 large-v3 needs ~5 GB at peak; 8 GB free leaves comfortable headroom.
"""

VRAM_FREE_MID_GB: Final[float] = 4.0
"""Free VRAM (GB) above which we run large-v3 quantized (int8_float16).

int8_float16 large-v3 needs ~3 GB at peak.
"""

VRAM_FREE_LOW_GB: Final[float] = 2.5
"""Free VRAM (GB) above which we run medium quantized."""

RAM_AVAILABLE_HIGH_GB: Final[float] = 6.0
"""Available RAM (GB) above which we run small (int8) on CPU.

`small` int8 needs ~3 GB working set including audio buffers + Streamlit.
"""

RAM_AVAILABLE_MID_GB: Final[float] = 3.0
"""Available RAM (GB) above which we run base (int8) on CPU."""


@dataclass(frozen=True, slots=True)
class ModelTier:
    """One row of the hardware → model lookup table.

    Attributes:
        model_size: Whisper model size identifier (e.g. ``"large-v3"``).
        compute_type: CTranslate2 compute type (e.g. ``"float16"``).
        device: Either ``"cuda"`` or ``"cpu"``.
        rationale: Short Spanish explanation shown to the user.
    """

    model_size: str
    compute_type: str
    device: str
    rationale: str


# Lookup table — referenced by `recommender.recommend()`.
# IMPORTANT: matches the table in the README and the task brief verbatim.
GPU_HIGH_TIER: Final[ModelTier] = ModelTier(
    model_size="large-v3",
    compute_type="float16",
    device="cuda",
    rationale=(
        "VRAM libre ≥ 8 GB: usamos `large-v3` en `float16` para máxima precisión. "
        "Quedará VRAM de sobra para el escritorio y otras apps."
    ),
)

GPU_MID_TIER: Final[ModelTier] = ModelTier(
    model_size="large-v3",
    compute_type="int8_float16",
    device="cuda",
    rationale=(
        "VRAM libre entre 4 y 8 GB: `large-v3` cuantizado (`int8_float16`) entra "
        "con holgura sin asfixiar al sistema operativo."
    ),
)

GPU_LOW_TIER: Final[ModelTier] = ModelTier(
    model_size="medium",
    compute_type="int8_float16",
    device="cuda",
    rationale=(
        "VRAM libre entre 2.5 y 4 GB: bajamos a `medium` con `int8_float16` "
        "para dejar margen al display y al navegador."
    ),
)

GPU_TINY_TIER: Final[ModelTier] = ModelTier(
    model_size="small",
    compute_type="int8_float16",
    device="cuda",
    rationale=(
        "VRAM libre < 2.5 GB: usamos `small` con `int8_float16`. La GPU sigue "
        "acelerando, pero protegemos al resto del sistema."
    ),
)

CPU_HIGH_TIER: Final[ModelTier] = ModelTier(
    model_size="small",
    compute_type="int8",
    device="cpu",
    rationale=(
        "Sin GPU y RAM libre ≥ 6 GB: usamos `small` en CPU con `int8`. Buen "
        "equilibrio calidad/velocidad sin saturar la RAM disponible."
    ),
)

CPU_MID_TIER: Final[ModelTier] = ModelTier(
    model_size="base",
    compute_type="int8",
    device="cpu",
    rationale=(
        "Sin GPU y RAM libre entre 3 y 6 GB: usamos `base` en CPU con `int8` "
        "para no competir con el sistema operativo y otras apps."
    ),
)

CPU_LOW_TIER: Final[ModelTier] = ModelTier(
    model_size="tiny",
    compute_type="int8",
    device="cpu",
    rationale=(
        "Sin GPU y RAM libre < 3 GB: usamos `tiny`. Calidad modesta, pero el "
        "equipo no se ahogará. Cierra alguna app y recarga si quieres más calidad."
    ),
)

# ---------------------------------------------------------------------------
# UI strings
# ---------------------------------------------------------------------------

APP_TITLE: Final[str] = "🎙️ Audicop"
APP_TAGLINE: Final[str] = "Suelta cualquier audio o vídeo. Recibe el texto. Sin configurar nada."
TEMP_PREFIX: Final[str] = "audicop_"
