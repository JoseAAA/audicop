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
