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
"""Upload size shown to the user in the UI (megabytes).

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
    # large-v3-turbo has a 4-layer decoder (vs 32 in large-v3), so it is
    # ~2-3x faster than large-v3 on CPU while keeping near-large-v3 quality.
    ("large-v3-turbo", "cpu"): 0.4,
    ("large-v3", "cpu"): 0.15,
    # GPU side — measured on RTX 3060 Laptop, conservative for older cards.
    ("tiny", "cuda"): 30.0,
    ("base", "cuda"): 25.0,
    ("small", "cuda"): 18.0,
    ("medium", "cuda"): 10.0,
    # Turbo's tiny decoder + the batched pipeline make it dramatically faster
    # than large-v3 on GPU; conservative figure for older cards.
    ("large-v3-turbo", "cuda"): 15.0,
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
    "large-v3-turbo",
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
    "large-v3-turbo": 1600,
    "large-v3": 3000,
}

# Batch size for the GPU BatchedInferencePipeline. 8 keeps VRAM use modest
# on consumer cards while still giving most of the throughput win; larger
# batches help on big GPUs but risk OOM on the 4-6 GB cards we target.
DEFAULT_BATCH_SIZE: Final[int] = 8

# Audicop's own model cache, used when the standard HF cache cannot be
# populated because Windows refuses symlink creation (WinError 1314 — see
# `transcriber._hf_symlinks_supported`). Files here are plain copies, no
# symlinks, so any user account on any Windows policy can read/write them.
AUDICOP_CACHE_SUBPATH: Final[str] = ".cache/audicop/models"

# Fallback faster-whisper repo template on HuggingFace. The transcriber
# prefers faster-whisper's own size→repo mapping (so models hosted outside
# the `Systran` org, like `large-v3-turbo`, resolve correctly) and only
# falls back to this template if that mapping is unavailable.
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

VRAM_FREE_HIGH_GB: Final[float] = 6.0
"""Free VRAM (GB) above which we run large-v3-turbo in float16.

float16 large-v3-turbo needs ~2-2.5 GB at peak (the turbo decoder is far
smaller than large-v3's), so 6 GB free leaves comfortable headroom.
"""

VRAM_FREE_MID_GB: Final[float] = 2.5
"""Free VRAM (GB) above which we run large-v3-turbo quantized (int8_float16).

int8_float16 large-v3-turbo needs ~1.5 GB at peak — it fits where plain
large-v3 never could, which is why turbo is the workhorse across most cards.
"""

VRAM_FREE_LOW_GB: Final[float] = 1.5
"""Free VRAM (GB) above which we run small quantized (int8_float16)."""

RAM_AVAILABLE_HIGH_GB: Final[float] = 6.0
"""Available RAM (GB) above which we run small (int8) on CPU.

`small` int8 needs ~3 GB working set including audio buffers + server.
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
# IMPORTANT: matches the table in the README.
GPU_HIGH_TIER: Final[ModelTier] = ModelTier(
    model_size="large-v3-turbo",
    compute_type="float16",
    device="cuda",
    rationale=(
        "VRAM libre ≥ 6 GB: usamos `large-v3-turbo` en `float16` — calidad casi "
        "idéntica a `large-v3` pero mucho más rápido. Sobra VRAM para el escritorio."
    ),
)

GPU_MID_TIER: Final[ModelTier] = ModelTier(
    model_size="large-v3-turbo",
    compute_type="int8_float16",
    device="cuda",
    rationale=(
        "VRAM libre entre 2.5 y 6 GB: `large-v3-turbo` cuantizado (`int8_float16`) "
        "entra con holgura y mantiene gran calidad sin asfixiar al sistema."
    ),
)

GPU_LOW_TIER: Final[ModelTier] = ModelTier(
    model_size="small",
    compute_type="int8_float16",
    device="cuda",
    rationale=(
        "VRAM libre entre 1.5 y 2.5 GB: usamos `small` con `int8_float16`. La GPU "
        "sigue acelerando, pero protegemos al display y al navegador."
    ),
)

GPU_TINY_TIER: Final[ModelTier] = ModelTier(
    model_size="base",
    compute_type="int8_float16",
    device="cuda",
    rationale=(
        "VRAM libre < 1.5 GB: usamos `base` con `int8_float16` para no quedarnos "
        "sin memoria. Cierra alguna app y recarga si quieres más calidad."
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

# ---------------------------------------------------------------------------
# Recording (own voice + meeting capture modes)
# ---------------------------------------------------------------------------

RECORDINGS_DIR_NAME: Final[str] = "audicop_recordings"
"""Subdirectory under the OS temp dir where in-progress recordings are written."""

RECORD_BLOCK_FRAMES: Final[int] = 8_000
"""Frames read per capture loop iteration (0.5 s at 16 kHz).

Reading in modest blocks keeps the WAV writer fed steadily, which avoids
soundcard's "data discontinuity" warnings without buffering large chunks
in memory.
"""

# App identifier (a substring of the exe / package name reported by Windows'
# microphone consent store) → friendly meeting name. `adapters.meeting`
# matches these ONLY against apps that are *actively using the microphone*,
# so background processes never trigger a false positive. This is what
# catches a browser meeting (Google Meet runs in chrome.exe) which a
# process-name scan never could. ".exe" suffixes keep matches precise — e.g.
# "zoom.exe" won't match PowerToys "ZoomIt.exe", and "teams.exe" won't match
# Steam's "steamservice.exe".
MEETING_APP_HINTS: Final[dict[str, str]] = {
    "chrome.exe": "una reunión en el navegador",
    "msedge.exe": "una reunión en el navegador",
    "firefox.exe": "una reunión en el navegador",
    "brave.exe": "una reunión en el navegador",
    "opera.exe": "una reunión en el navegador",
    "ms-teams.exe": "Microsoft Teams",
    "teams.exe": "Microsoft Teams",
    "msteams": "Microsoft Teams",  # new Teams is a packaged app (family name)
    "zoom.exe": "Zoom",
    "webex.exe": "Webex",
    "slack.exe": "Slack",
    "discord.exe": "Discord",
    "skype.exe": "Skype",
}

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

SRT_SUFFIX: Final[str] = ".srt"
VTT_SUFFIX: Final[str] = ".vtt"
TXT_SUFFIX: Final[str] = ".txt"

# Above this transcript length we warn the user that the AI chat may exceed
# the model's context window or cost more (rough char proxy for tokens).
LONG_TRANSCRIPT_CHARS: Final[int] = 120_000
