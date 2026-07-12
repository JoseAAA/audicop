"""Tests for the local-LLM recommender (`recommend_llm`)."""

from __future__ import annotations

import pytest

from app.adapters.hardware import HardwareInfo
from app.services.recommender import LlmChoice, recommend_llm


def _hw(
    *,
    has_cuda: bool = False,
    gpu_vram_total_gb: float | None = None,
    gpu_vram_free_gb: float | None = None,
    ram_total_gb: float = 32.0,
    ram_available_gb: float = 16.0,
) -> HardwareInfo:
    """Build a minimal HardwareInfo for the local-LLM recommender tests."""
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
    ("ram_available", "expected_key", "expected_available"),
    [
        (16.0, "qwen2.5-3b", True),  # usable 14 -> 3B
        (7.0, "qwen2.5-3b", True),  # usable 5.0 -> 3B (boundary)
        (6.9, "qwen2.5-1.5b", True),  # usable 4.9 -> 1.5B
        (4.5, "qwen2.5-1.5b", True),  # usable 2.5 -> 1.5B (boundary)
        (4.4, "llama-3.2-1b", True),  # usable 2.4 -> 1B
        (3.5, "llama-3.2-1b", True),  # usable 1.5 -> 1B (boundary)
        (3.4, "", False),  # usable 1.4 -> none
        (2.0, "", False),  # usable 0.0 -> none
    ],
)
def test_cpu_llm_tiers_use_usable_ram(
    ram_available: float, expected_key: str, expected_available: bool
) -> None:
    """The CPU tier is chosen on available RAM minus the explicit reserve."""
    choice = recommend_llm(_hw(ram_available_gb=ram_available))
    assert isinstance(choice, LlmChoice)
    assert choice.device == "cpu"
    assert choice.model_key == expected_key
    assert choice.available is expected_available
    assert choice.rationale  # always a user-facing message
    assert choice.n_gpu_layers == 0


@pytest.mark.parametrize(
    ("vram_free", "expected_key"),
    [
        (8.0, "qwen3-4b"),  # usable 7.0 -> 4B (quality pick on roomy GPUs)
        (4.5, "qwen3-4b"),  # usable 3.5 -> 4B (boundary)
        (4.0, "qwen2.5-3b"),  # usable 3.0 -> 3B
        (3.0, "qwen2.5-3b"),  # usable 2.0 -> 3B (boundary)
        (2.9, "qwen2.5-1.5b"),  # usable 1.9 -> 1.5B
        (2.5, "qwen2.5-1.5b"),  # usable 1.5 -> 1.5B (boundary)
    ],
)
def test_gpu_llm_tiers_use_usable_vram(vram_free: float, expected_key: str) -> None:
    """The GPU tier is chosen on free VRAM minus the explicit reserve."""
    choice = recommend_llm(
        _hw(has_cuda=True, gpu_vram_total_gb=24.0, gpu_vram_free_gb=vram_free),
        gpu_offload=True,
    )
    assert choice.device == "cuda"
    assert choice.model_key == expected_key
    assert choice.available is True
    assert choice.n_gpu_layers == -1


def test_gpu_too_small_falls_back_to_cpu() -> None:
    """Tiny free VRAM drops to the CPU path instead of forcing a GPU model."""
    choice = recommend_llm(
        _hw(
            has_cuda=True,
            gpu_vram_total_gb=24.0,
            gpu_vram_free_gb=2.0,  # usable 1.0 < MID -> CPU path
            ram_available_gb=16.0,
        ),
        gpu_offload=True,
    )
    assert choice.device == "cpu"
    assert choice.model_key == "qwen2.5-3b"


def test_gpu_ignored_without_offload_support() -> None:
    """A CUDA machine still runs on CPU when the engine can't offload (CPU wheel)."""
    choice = recommend_llm(
        _hw(has_cuda=True, gpu_vram_total_gb=24.0, gpu_vram_free_gb=10.0, ram_available_gb=16.0),
        gpu_offload=False,  # default CPU-only wheel
    )
    assert choice.device == "cpu"
    assert choice.model_key == "qwen2.5-3b"
    assert choice.n_gpu_layers == 0


def test_unavailable_when_memory_tiny() -> None:
    """Too little memory yields an unavailable choice (use cloud / copy text)."""
    choice = recommend_llm(_hw(ram_available_gb=2.0))
    assert choice.available is False
    assert choice.model_key == ""
    assert "memoria" in choice.rationale.lower()


def test_available_choice_carries_full_spec() -> None:
    """An available choice exposes the repo, file, label and runtime fields."""
    choice = recommend_llm(_hw(ram_available_gb=16.0))
    assert choice.available is True
    assert choice.repo_id and choice.filename and choice.label
    assert choice.n_ctx > 0
    assert choice.download_size_mb > 0


def test_total_ram_irrelevant_when_available_is_low() -> None:
    """A 64 GB machine with little available RAM still picks the smallest tier."""
    choice = recommend_llm(_hw(ram_total_gb=64.0, ram_available_gb=3.5))
    assert choice.model_key == "llama-3.2-1b"  # usable 1.5 -> 1B, not 3B
