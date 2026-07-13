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

# Batch size for the GPU BatchedInferencePipeline. 8 keeps VRAM use modest
# on consumer cards while still giving most of the throughput win; larger
# batches help on big GPUs but risk OOM on the 4-6 GB cards we target.
DEFAULT_BATCH_SIZE: Final[int] = 8

# Audicop's own model cache, used when the standard HF cache cannot be
# populated because Windows refuses symlink creation (WinError 1314 — see
# `transcriber._hf_symlinks_supported`). Files here are plain copies, no
# symlinks, so any user account on any Windows policy can read/write them.
AUDICOP_CACHE_SUBPATH: Final[str] = ".cache/audicop/models"

# The local meetings library (SQLite). USER DATA, not cache: it holds the
# transcripts and AI notes the user chose to keep. Lives on the user's disk
# only, never leaves the machine, and each meeting can be deleted from the UI.
MEETINGS_DB_SUBPATH: Final[str] = ".audicop/meetings.db"

# Per-meeting listening copy of the audio (AAC .m4a, local disk only). Lets
# the user replay a meeting after reopening it; deleted with the meeting.
MEETINGS_AUDIO_SUBPATH: Final[str] = ".audicop/audio"

# Fallback faster-whisper repo template on HuggingFace. The transcriber
# prefers faster-whisper's own size→repo mapping (so models hosted outside
# the `Systran` org, like `large-v3-turbo`, resolve correctly) and only
# falls back to this template if that mapping is unavailable.
FASTER_WHISPER_REPO_TEMPLATE: Final[str] = "Systran/faster-whisper-{size}"

DEFAULT_LANGUAGES: Final[tuple[str, ...]] = ("auto", "es", "en", "pt", "fr", "it", "de")
DEFAULT_TASKS: Final[tuple[str, ...]] = ("transcribe", "translate")
DEFAULT_VAD_FILTER: Final[bool] = True

# Beam search width per device. On GPU the wider beam is nearly free; on CPU
# it multiplies wall time, and field results from meetily showed beam 2 keeps
# quality while being markedly faster/stabler on modest machines.
DEFAULT_BEAM_SIZE: Final[int] = 5
BEAM_SIZE_CPU: Final[int] = 2

# ---------------------------------------------------------------------------
# Voice-activity detection + decode thresholds
#
# Tuned values ported from meetily's production audio pipeline (silero VAD
# analyzed at 30 ms; thresholds battle-tested across their releases). Only
# speech reaches Whisper, which cuts decode work sharply and removes the
# classic "hallucinate during silence" failure.
# ---------------------------------------------------------------------------

VAD_THRESHOLD: Final[float] = 0.5
"""Silero probability at/above which audio counts as speech."""

VAD_NEG_THRESHOLD: Final[float] = 0.35
"""Probability below which speech is considered ended (hysteresis)."""

VAD_MIN_SPEECH_MS: Final[int] = 250
"""Discard blips shorter than this — Whisper misreads sub-250 ms fragments."""

VAD_MIN_SILENCE_MS: Final[int] = 400
"""Silence needed to close a speech segment; bridges natural pauses."""

VAD_SPEECH_PAD_MS: Final[int] = 400
"""Padding kept around each speech segment so word edges aren't clipped."""

NO_SPEECH_THRESHOLD: Final[float] = 0.55
"""Whisper's own no-speech gate. Defaults (0.6-0.75) drop quiet speakers;
0.55 keeps low-volume voices while VAD already filters true silence."""

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
# Local LLM (llama.cpp) — private, in-process summarization / chat
#
# Mirrors the Whisper recommender philosophy: pick a model by FREE memory,
# never total. On top of reading the live "available" / "free" figures, we
# subtract an EXPLICIT reserve so the model never claims the last slice of
# memory the OS and other apps need — a 16 GB laptop must stay usable.
# ---------------------------------------------------------------------------

MEMORY_RESERVE_GB: Final[float] = 2.0
"""RAM (GB) always left for the OS and other apps when sizing the local LLM.

Subtracted from ``ram_available_gb`` before choosing a tier, so the figure we
size against ("usable") is what we can actually take without thrashing.
"""

VRAM_RESERVE_GB: Final[float] = 1.0
"""VRAM (GB) left free for the display, CUDA context and allocator slack.

Subtracted from ``gpu_vram_free_gb`` before choosing a GPU tier. Overflowing
VRAM does not slow you down gently — it spills to system RAM over PCIe and
throughput collapses, so we stay well under.
"""

LLM_DEFAULT_N_CTX: Final[int] = 8192
"""Context window (tokens) for the local model.

Big enough for a transcript chunk plus the answer; long transcripts are
summarized with map-reduce so they never need to fit whole in the window.
"""

LLM_TEMPERATURE: Final[float] = 0.3
"""Sampling temperature for the local model.

Low on purpose: summaries, tasks and minutes must stay faithful to the
transcript. Small models drift/hallucinate more at higher temperatures.
"""

# Map-reduce for transcripts too long for a single pass (values ported from
# meetily's tuned pipeline). Above the threshold the transcript is condensed
# chunk by chunk before the final answer pass; chunks reserve prompt overhead
# and share a small overlap so boundary sentences keep their context.
LLM_MAPREDUCE_TOKEN_THRESHOLD: Final[int] = 4000
LLM_CHUNK_PROMPT_OVERHEAD_TOKENS: Final[int] = 300
LLM_CHUNK_OVERLAP_TOKENS: Final[int] = 100

LLM_RETRIEVAL_MAX_CHARS: Final[int] = 8000
"""Character budget for retrieved transcript excerpts on a free question.

For a specific question on a long audio we answer from the transcript LINES
that match the question (see :mod:`app.services.retrieve`), NOT from the lossy
global summary — otherwise a dropped detail becomes a false "no se menciona".
~8000 chars ≈ 2800 tokens, leaving room for the system prompt and the reply.
"""

LLM_SYNTHESIS_TARGET_TOKENS: Final[int] = 2500
"""Size the condensed notes are reduced to before the final answer pass.

Deliberately much smaller than the raw threshold: when a 2-3 h meeting is
condensed to ~60 notes, a small model *echoes* them (60 "key points"…)
instead of synthesizing. Reducing to roughly a page of notes forces real
synthesis and keeps answers short. Computed once per audio and cached.
"""

# Qwen-family sampling preset (ported from meetily's tuned local-LLM presets):
# a tight nucleus keeps structured output clean, presence penalty discourages
# re-listing the same item, and a mild repeat penalty avoids degrading fluency.
LLM_TOP_K: Final[int] = 20
LLM_TOP_P: Final[float] = 0.8
LLM_PRESENCE_PENALTY: Final[float] = 0.3
LLM_REPEAT_PENALTY: Final[float] = 1.05

# Strict variant for STRUCTURED outputs (the quick actions / map phase, whose
# prompts define an exact format). Small models drift into copying the
# transcript verbatim at higher temperatures; meetily's "tight_structured"
# preset (temp 0.1) keeps them on-format.
LLM_TEMPERATURE_STRICT: Final[float] = 0.1
LLM_TOP_P_STRICT: Final[float] = 0.88

LLM_MAX_TOKENS: Final[int] = 1536
"""Cap on generated tokens per reply.

Analysis outputs (summary, tasks, minutes) fit comfortably; the cap keeps a
small model from rambling for minutes on a slow CPU. Sized so long-meeting
notes (acta of a 2-3 h session) don't get cut mid-sentence.
"""


@dataclass(frozen=True, slots=True)
class LlmModelSpec:
    """A local GGUF model Audicop can download and run with llama.cpp.

    Attributes:
        key: Stable internal identifier used by the recommender.
        repo_id: HuggingFace repo hosting the GGUF file.
        filename: GGUF filename within the repo (a single quantized file).
        label: Short user-facing name.
        download_size_mb: Approximate download size, to set expectations.
    """

    key: str
    repo_id: str
    filename: str
    label: str
    download_size_mb: int


# Curated GGUF shortlist (Q4_K_M — the quality/size sweet spot). All repos
# are ungated. Qwen3-4B-Instruct-2507 is the quality pick for GPUs (clearly
# better instruction-following/synthesis, same generation meetily ships);
# Qwen2.5 stays the CPU workhorse where its smaller size keeps answers fast.
LLM_QWEN3_4B: Final[LlmModelSpec] = LlmModelSpec(
    key="qwen3-4b",
    repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
    filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    label="Qwen3 4B (privado)",
    download_size_mb=2500,
)
LLM_QWEN_3B: Final[LlmModelSpec] = LlmModelSpec(
    key="qwen2.5-3b",
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    label="Qwen2.5 3B (privado)",
    download_size_mb=2000,
)
LLM_QWEN_1_5B: Final[LlmModelSpec] = LlmModelSpec(
    key="qwen2.5-1.5b",
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    label="Qwen2.5 1.5B (privado)",
    download_size_mb=1100,
)
LLM_LLAMA_1B: Final[LlmModelSpec] = LlmModelSpec(
    key="llama-3.2-1b",
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    label="Llama 3.2 1B (privado)",
    download_size_mb=800,
)

LLM_MODELS: Final[dict[str, LlmModelSpec]] = {
    m.key: m for m in (LLM_QWEN3_4B, LLM_QWEN_3B, LLM_QWEN_1_5B, LLM_LLAMA_1B)
}
"""Lookup of every selectable local model, keyed by :attr:`LlmModelSpec.key`."""

# Tier thresholds, expressed in USABLE memory = (free / available) minus the
# reserve above. Derived from each model's Q4_K_M weight plus ~1.4x headroom
# for KV-cache / context on GPU, and a comfortable working set on CPU.
LLM_VRAM_USABLE_HIGH_GB: Final[float] = 3.5
"""Usable VRAM (GB) at/above which Qwen3 4B runs fully on the GPU."""

LLM_VRAM_USABLE_MID_GB: Final[float] = 2.0
"""Usable VRAM (GB) at/above which Qwen2.5 3B runs on the GPU."""

LLM_VRAM_USABLE_LOW_GB: Final[float] = 1.5
"""Usable VRAM (GB) at/above which Qwen2.5 1.5B runs on the GPU."""

LLM_RAM_USABLE_HIGH_GB: Final[float] = 5.0
"""Usable RAM (GB) at/above which Qwen2.5 3B runs on CPU."""

LLM_RAM_USABLE_MID_GB: Final[float] = 2.5
"""Usable RAM (GB) at/above which Qwen2.5 1.5B runs on CPU."""

LLM_RAM_USABLE_LOW_GB: Final[float] = 1.5
"""Usable RAM (GB) at/above which Llama 3.2 1B runs on CPU (floor for local)."""


@dataclass(frozen=True, slots=True)
class LlmTier:
    """One row of the hardware -> local-LLM lookup table.

    Attributes:
        model_key: Key into :data:`LLM_MODELS`, or ``""`` when no local model
            fits and the user should use a cloud provider instead.
        device: ``"cuda"`` or ``"cpu"``.
        rationale: Short Spanish explanation shown to the user.
    """

    model_key: str
    device: str
    rationale: str


LLM_GPU_HIGH_TIER: Final[LlmTier] = LlmTier(
    model_key="qwen3-4b",
    device="cuda",
    rationale=(
        "VRAM libre de sobra: corremos Qwen3 4B entero en la GPU — el mejor "
        "sintetizador que cabe en una laptop. Rápido y 100% privado."
    ),
)
LLM_GPU_MID_TIER: Final[LlmTier] = LlmTier(
    model_key="qwen2.5-3b",
    device="cuda",
    rationale=(
        "VRAM libre media: Qwen2.5 3B entero en la GPU. Resúmenes rápidos sin "
        "que el texto salga de tu equipo."
    ),
)
LLM_GPU_LOW_TIER: Final[LlmTier] = LlmTier(
    model_key="qwen2.5-1.5b",
    device="cuda",
    rationale=(
        "VRAM libre ajustada: usamos Qwen2.5 1.5B en la GPU. Rápido y privado, "
        "dejando memoria para el display."
    ),
)
LLM_CPU_HIGH_TIER: Final[LlmTier] = LlmTier(
    model_key="qwen2.5-3b",
    device="cpu",
    rationale=(
        "Sin GPU pero con RAM holgada (ya descontado lo que necesita el "
        "sistema): Qwen2.5 3B en CPU. Buen resumen y estructura, 100% local."
    ),
)
LLM_CPU_MID_TIER: Final[LlmTier] = LlmTier(
    model_key="qwen2.5-1.5b",
    device="cpu",
    rationale=(
        "Sin GPU y RAM media: Qwen2.5 1.5B en CPU. Más ligero y rápido, "
        "manteniendo todo en tu equipo."
    ),
)
LLM_CPU_LOW_TIER: Final[LlmTier] = LlmTier(
    model_key="llama-3.2-1b",
    device="cpu",
    rationale=(
        "RAM justa: Llama 3.2 1B en CPU. Calidad modesta pero privada; cierra "
        "apps y recarga si quieres subir de modelo."
    ),
)
LLM_NONE_TIER: Final[LlmTier] = LlmTier(
    model_key="",
    device="cpu",
    rationale=(
        "Poca memoria libre para un modelo local. Cierra algunas apps y recarga, "
        "o usa un equipo con más memoria (o una GPU). También puedes copiar el "
        "texto y analizarlo donde quieras."
    ),
)

TEMP_PREFIX: Final[str] = "audicop_"
"""Prefix for the temp directories that hold uploads/conversions in flight."""

# ---------------------------------------------------------------------------
# Recording (own voice + meeting capture modes)
# ---------------------------------------------------------------------------

RECORDINGS_DIR_NAME: Final[str] = "audicop_recordings"
"""Subdirectory under the OS temp dir where in-progress recordings are written."""

LIVE_WINDOW_S: Final[float] = 6.0
"""Seconds of audio accumulated before each live-transcription decode.

Short enough to feel "live", long enough for Whisper to have context and to
keep decode overhead low (a GPU decodes 6 s in well under a second).
"""

LIVE_MIN_TAIL_S: Final[float] = 0.75
"""Minimum leftover audio worth decoding when the recording stops."""

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
