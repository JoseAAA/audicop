"""Tests for `app.services.recommender`."""

from __future__ import annotations

import pytest

from app.adapters.hardware import HardwareInfo
from app.services.recommender import ModelChoice, recommend


def _hw(
    *,
    has_cuda: bool = False,
    gpu_vram_total_gb: float | None = None,
    gpu_vram_free_gb: float | None = None,
    ram_total_gb: float = 32.0,
    ram_available_gb: float = 16.0,
) -> HardwareInfo:
    """Build a minimal HardwareInfo for the recommender tests."""
    return HardwareInfo(
        os_name="Linux",
        os_release="6.0",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        cpu_brand="Test CPU",
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        has_cuda=has_cuda,
        gpu_name="NVIDIA Test" if has_cuda else None,
        gpu_vram_total_gb=gpu_vram_total_gb,
        gpu_vram_free_gb=gpu_vram_free_gb,
        gpu_driver_version="555.42" if has_cuda else None,
    )


@pytest.mark.parametrize(
    ("vram_free", "expected_size", "expected_compute"),
    [
        (24.0, "large-v3-turbo", "float16"),
        (6.0, "large-v3-turbo", "float16"),
        (5.9, "large-v3-turbo", "int8_float16"),
        (2.5, "large-v3-turbo", "int8_float16"),
        (2.4, "small", "int8_float16"),
        (1.5, "small", "int8_float16"),
        (1.4, "base", "int8_float16"),
    ],
)
def test_gpu_tiers_use_free_vram(
    vram_free: float, expected_size: str, expected_compute: str
) -> None:
    """Free VRAM (not total) drives the GPU tier choice."""
    # total much higher than free to prove we read free, not total
    choice = recommend(_hw(has_cuda=True, gpu_vram_total_gb=24.0, gpu_vram_free_gb=vram_free))
    assert isinstance(choice, ModelChoice)
    assert choice.device == "cuda"
    assert choice.model_size == expected_size
    assert choice.compute_type == expected_compute
    assert choice.rationale  # non-empty Spanish message


@pytest.mark.parametrize(
    ("ram_available", "expected_size", "expected_compute"),
    [
        (64.0, "small", "int8"),
        (6.0, "small", "int8"),
        (5.9, "base", "int8"),
        (3.0, "base", "int8"),
        (2.9, "tiny", "int8"),
        (0.5, "tiny", "int8"),
    ],
)
def test_cpu_tiers_use_available_ram(
    ram_available: float, expected_size: str, expected_compute: str
) -> None:
    """Available RAM (not total) drives the CPU tier choice."""
    # total much higher than available to prove we read available
    choice = recommend(_hw(has_cuda=False, ram_total_gb=64.0, ram_available_gb=ram_available))
    assert choice.device == "cpu"
    assert choice.model_size == expected_size
    assert choice.compute_type == expected_compute
    assert choice.rationale


def test_total_ram_irrelevant_when_available_is_low() -> None:
    """A 64 GB machine with only 1 GB available recommends `tiny`, not `small`."""
    choice = recommend(_hw(has_cuda=False, ram_total_gb=64.0, ram_available_gb=1.0))
    assert choice.model_size == "tiny"


def test_total_vram_irrelevant_when_free_is_low() -> None:
    """A 24 GB GPU with only 1 GB free recommends `base`, not the turbo model."""
    choice = recommend(_hw(has_cuda=True, gpu_vram_total_gb=24.0, gpu_vram_free_gb=1.0))
    assert choice.model_size == "base"


def test_gpu_falls_back_to_total_when_free_unknown() -> None:
    """If `gpu_vram_free_gb` is None we use total VRAM as best-effort signal."""
    choice = recommend(_hw(has_cuda=True, gpu_vram_total_gb=12.0, gpu_vram_free_gb=None))
    assert choice.device == "cuda"
    assert choice.model_size == "large-v3-turbo"


def test_cuda_flag_set_but_no_vram_falls_back_to_cpu() -> None:
    """If CUDA is reported but neither total nor free VRAM is known, use CPU path."""
    choice = recommend(
        _hw(
            has_cuda=True,
            gpu_vram_total_gb=None,
            gpu_vram_free_gb=None,
            ram_available_gb=16.0,
        )
    )
    assert choice.device == "cpu"
    assert choice.model_size == "small"


def test_rationale_is_in_spanish() -> None:
    """Rationale text is the user-facing Spanish message."""
    choice = recommend(_hw(has_cuda=True, gpu_vram_total_gb=12.0, gpu_vram_free_gb=10.0))
    assert any(word in choice.rationale for word in ("usamos", "VRAM", "libre"))
