"""Pick the right Whisper model and compute type for the local hardware.

The recommendation logic is intentionally pure and deterministic: given a
:class:`~app.adapters.hardware.HardwareInfo`, :func:`recommend` returns a
:class:`ModelChoice` that tells the rest of the app which model to load,
on which device, and with what compute type.

The full table lives in :mod:`app.core.config` so the README and the code
stay in sync.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.adapters.hardware import HardwareInfo
from app.core import config
from app.core.config import ModelTier

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """A concrete recommendation produced by :func:`recommend`.

    Attributes:
        model_size: Whisper model size identifier (e.g. ``"large-v3"``).
        compute_type: CTranslate2 compute type (e.g. ``"int8_float16"``).
        device: Either ``"cuda"`` or ``"cpu"``.
        rationale: Spanish, user-facing explanation of why this tier was chosen.
    """

    model_size: str
    compute_type: str
    device: str
    rationale: str

    @classmethod
    def from_tier(cls, tier: ModelTier) -> ModelChoice:
        """Build a :class:`ModelChoice` from a :class:`~app.core.config.ModelTier`."""
        return cls(
            model_size=tier.model_size,
            compute_type=tier.compute_type,
            device=tier.device,
            rationale=tier.rationale,
        )


def _gpu_tier(vram_free_gb: float) -> ModelTier:
    """Return the GPU model tier for a given amount of FREE VRAM (GB)."""
    if vram_free_gb >= config.VRAM_FREE_HIGH_GB:
        return config.GPU_HIGH_TIER
    if vram_free_gb >= config.VRAM_FREE_MID_GB:
        return config.GPU_MID_TIER
    if vram_free_gb >= config.VRAM_FREE_LOW_GB:
        return config.GPU_LOW_TIER
    return config.GPU_TINY_TIER


def _cpu_tier(ram_available_gb: float) -> ModelTier:
    """Return the CPU model tier for a given amount of AVAILABLE RAM (GB)."""
    if ram_available_gb >= config.RAM_AVAILABLE_HIGH_GB:
        return config.CPU_HIGH_TIER
    if ram_available_gb >= config.RAM_AVAILABLE_MID_GB:
        return config.CPU_MID_TIER
    return config.CPU_LOW_TIER


def recommend(hw: HardwareInfo) -> ModelChoice:
    """Recommend a Whisper model and compute type for the given hardware.

    The recommendation is based on **currently free / available** memory,
    not on total memory. This is important because the OS, browser and
    other apps reserve a non-trivial chunk of RAM and VRAM that we must
    leave alone — recommending a model that fits in "total" but not in
    "available" is the fastest way to thrash a laptop.

    Logic:

        - GPU, free VRAM >= 6 GB        -> large-v3-turbo / float16
        - GPU, free VRAM 2.5-6 GB       -> large-v3-turbo / int8_float16
        - GPU, free VRAM 1.5-2.5 GB     -> small / int8_float16
        - GPU, free VRAM < 1.5 GB       -> base / int8_float16
        - CPU, available RAM >= 6 GB    -> small / int8
        - CPU, available RAM 3-6 GB     -> base / int8
        - CPU, available RAM < 3 GB     -> tiny / int8

    ``large-v3-turbo`` is also selectable manually for CPU users who want
    near-large-v3 quality and accept a slower run; it is not a default
    there because it roughly doubles wall time versus ``small``.

    If a GPU is present but its free VRAM is unknown, we fall back to
    total VRAM (or to the CPU path) so the recommendation is still safe.

    Args:
        hw: Snapshot of the local hardware, typically produced by
            :func:`app.adapters.hardware.detect_hardware`.

    Returns:
        A :class:`ModelChoice` ready to feed into the transcriber.
    """
    if hw.has_cuda:
        free_vram = hw.gpu_vram_free_gb
        if free_vram is None:
            free_vram = hw.gpu_vram_total_gb
        if free_vram is not None:
            tier = _gpu_tier(free_vram)
            logger.info(
                "GPU path: vram_free=%.1f GB → %s/%s",
                free_vram,
                tier.model_size,
                tier.compute_type,
            )
            return ModelChoice.from_tier(tier)

    tier = _cpu_tier(hw.ram_available_gb)
    logger.info(
        "CPU path: ram_available=%.1f GB → %s/%s",
        hw.ram_available_gb,
        tier.model_size,
        tier.compute_type,
    )
    return ModelChoice.from_tier(tier)


@dataclass(frozen=True, slots=True)
class LlmChoice:
    """A concrete local-LLM recommendation produced by :func:`recommend_llm`.

    Attributes:
        available: ``True`` if a local model fits; ``False`` means the caller
            should fall back to a cloud provider (or "copy the text").
        model_key: Key into :data:`app.core.config.LLM_MODELS` (``""`` when
            not available).
        repo_id: HuggingFace repo of the GGUF (``""`` when not available).
        filename: GGUF filename (``""`` when not available).
        label: User-facing model name (``""`` when not available).
        device: ``"cuda"`` or ``"cpu"``.
        n_gpu_layers: Layers to offload to GPU (``-1`` = all, ``0`` = CPU only).
        n_ctx: Context window in tokens (``0`` when not available).
        download_size_mb: Approximate first-run download size in MB.
        rationale: Spanish, user-facing explanation of the choice.
    """

    available: bool
    model_key: str
    repo_id: str
    filename: str
    label: str
    device: str
    n_gpu_layers: int
    n_ctx: int
    download_size_mb: int
    rationale: str

    @classmethod
    def from_tier(cls, tier: config.LlmTier, *, n_gpu_layers: int) -> LlmChoice:
        """Build an :class:`LlmChoice` from a tier, resolving its model spec.

        A tier with an empty ``model_key`` yields an unavailable choice (the
        UI should then offer a cloud provider).
        """
        if not tier.model_key:
            return cls(
                available=False,
                model_key="",
                repo_id="",
                filename="",
                label="",
                device=tier.device,
                n_gpu_layers=0,
                n_ctx=0,
                download_size_mb=0,
                rationale=tier.rationale,
            )
        spec = config.LLM_MODELS[tier.model_key]
        return cls(
            available=True,
            model_key=spec.key,
            repo_id=spec.repo_id,
            filename=spec.filename,
            label=spec.label,
            device=tier.device,
            n_gpu_layers=n_gpu_layers,
            n_ctx=config.LLM_DEFAULT_N_CTX,
            download_size_mb=spec.download_size_mb,
            rationale=tier.rationale,
        )


def _llm_gpu_tier(usable_vram_gb: float) -> config.LlmTier | None:
    """Return the GPU local-LLM tier for usable VRAM, or ``None`` if too little.

    ``None`` signals the caller to fall back to the CPU path rather than
    forcing a model that would spill out of VRAM.
    """
    if usable_vram_gb >= config.LLM_VRAM_USABLE_HIGH_GB:
        return config.LLM_GPU_HIGH_TIER
    if usable_vram_gb >= config.LLM_VRAM_USABLE_MID_GB:
        return config.LLM_GPU_MID_TIER
    if usable_vram_gb >= config.LLM_VRAM_USABLE_LOW_GB:
        return config.LLM_GPU_LOW_TIER
    return None


def _llm_cpu_tier(usable_ram_gb: float) -> config.LlmTier:
    """Return the CPU local-LLM tier for usable RAM.

    Returns :data:`config.LLM_NONE_TIER` when even the smallest model would
    leave too little headroom.
    """
    if usable_ram_gb >= config.LLM_RAM_USABLE_HIGH_GB:
        return config.LLM_CPU_HIGH_TIER
    if usable_ram_gb >= config.LLM_RAM_USABLE_MID_GB:
        return config.LLM_CPU_MID_TIER
    if usable_ram_gb >= config.LLM_RAM_USABLE_LOW_GB:
        return config.LLM_CPU_LOW_TIER
    return config.LLM_NONE_TIER


def recommend_llm(hw: HardwareInfo, *, gpu_offload: bool = False) -> LlmChoice:
    """Recommend a local LLM (or a cloud fallback) for the given hardware.

    Like :func:`recommend`, sizing is based on **currently free** memory; on
    top of that we subtract an explicit reserve
    (:data:`config.MEMORY_RESERVE_GB` / :data:`config.VRAM_RESERVE_GB`) so the
    model never claims the last slice of memory the OS and other apps need.

    The GPU path is tried first, **but only when ``gpu_offload`` is True** —
    that flag must reflect whether the *installed* llama.cpp build can actually
    use the GPU (see :func:`app.adapters.local_llm.supports_gpu_offload`). The
    default CPU-only wheel can't, so even on a CUDA machine we size against RAM
    rather than promising VRAM we cannot use. If usable VRAM is too small even
    for the smallest GPU tier, we fall back to the CPU path, which may itself
    return an *unavailable* choice (``available is False``), signalling the UI
    to offer a cloud provider instead.

    Note:
        Whisper and the LLM both consume memory. Callers should size the LLM
        *after* the transcription model has been released, ideally by
        re-running :func:`app.adapters.hardware.detect_hardware` so the free
        figures reflect the freed memory.

    Args:
        hw: Snapshot of the local hardware.
        gpu_offload: Whether the installed engine can offload to the GPU.

    Returns:
        An :class:`LlmChoice`; check ``.available`` before attempting to load.
    """
    if gpu_offload and hw.has_cuda and hw.gpu_vram_free_gb is not None:
        usable_vram = max(0.0, hw.gpu_vram_free_gb - config.VRAM_RESERVE_GB)
        gpu_tier = _llm_gpu_tier(usable_vram)
        if gpu_tier is not None:
            logger.info(
                "LLM GPU path: usable_vram=%.1f GB → %s",
                usable_vram,
                gpu_tier.model_key,
            )
            return LlmChoice.from_tier(gpu_tier, n_gpu_layers=-1)

    usable_ram = max(0.0, hw.ram_available_gb - config.MEMORY_RESERVE_GB)
    cpu_tier = _llm_cpu_tier(usable_ram)
    logger.info(
        "LLM CPU path: usable_ram=%.1f GB → %s",
        usable_ram,
        cpu_tier.model_key or "<none>",
    )
    return LlmChoice.from_tier(cpu_tier, n_gpu_layers=0)
