"""Tests for `audicop.hardware`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from audicop import hardware
from audicop.hardware import HardwareInfo, detect_hardware


@pytest.fixture
def fake_psutil() -> MagicMock:
    """Mock psutil with predictable CPU/RAM values (32 GB total, 24 GB available)."""
    fake = MagicMock()
    fake.cpu_count.side_effect = lambda logical=True: 16 if logical else 8
    fake.virtual_memory.return_value = SimpleNamespace(
        total=32 * 1024**3,
        available=24 * 1024**3,
    )
    return fake


def _patch_nvidia_smi(*, present: bool, stdout: str = "", returncode: int = 0):
    """Build the patches that simulate `nvidia-smi` (or its absence)."""
    which_target = "C:\\Windows\\system32\\nvidia-smi.exe" if present else None
    fake_run = MagicMock(return_value=MagicMock(returncode=returncode, stdout=stdout, stderr=""))
    return (
        patch.object(hardware.shutil, "which", return_value=which_target),
        patch.object(hardware.subprocess, "run", fake_run),
    )


def test_detect_hardware_cpu_only(fake_psutil: MagicMock) -> None:
    """No nvidia-smi on PATH → CPU-only HardwareInfo with total + available RAM."""
    which_p, run_p = _patch_nvidia_smi(present=False)
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()

    assert isinstance(info, HardwareInfo)
    assert info.cpu_cores_physical == 8
    assert info.cpu_cores_logical == 16
    assert info.ram_total_gb == 32.0
    assert info.ram_available_gb == 24.0
    assert info.has_cuda is False
    assert info.gpu_name is None
    assert info.gpu_vram_total_gb is None
    assert info.gpu_vram_free_gb is None
    assert info.gpu_driver_version is None


def test_detect_hardware_with_nvidia_gpu(fake_psutil: MagicMock) -> None:
    """nvidia-smi reports total + free VRAM and driver version."""
    stdout = "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 5996, 566.07\n"
    which_p, run_p = _patch_nvidia_smi(present=True, stdout=stdout)
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()

    assert info.has_cuda is True
    assert info.gpu_name == "NVIDIA GeForce RTX 3060 Laptop GPU"
    assert info.gpu_vram_total_gb == 6.0
    assert info.gpu_vram_free_gb == pytest.approx(5.9, abs=0.05)
    assert info.gpu_driver_version == "566.07"


def test_detect_hardware_with_high_vram_gpu(fake_psutil: MagicMock) -> None:
    """24 GB GPU with 20 GB free is parsed correctly."""
    stdout = "NVIDIA GeForce RTX 4090, 24576, 20480, 555.42\n"
    which_p, run_p = _patch_nvidia_smi(present=True, stdout=stdout)
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()

    assert info.gpu_vram_total_gb == 24.0
    assert info.gpu_vram_free_gb == 20.0


def test_detect_hardware_first_gpu_when_multiple(fake_psutil: MagicMock) -> None:
    """If nvidia-smi reports two GPUs, we pick the first."""
    stdout = (
        "NVIDIA GeForce RTX 3090, 24576, 18000, 555.42\n"
        "NVIDIA GeForce GTX 1080, 8192, 8000, 555.42\n"
    )
    which_p, run_p = _patch_nvidia_smi(present=True, stdout=stdout)
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()

    assert info.gpu_name == "NVIDIA GeForce RTX 3090"
    assert info.gpu_vram_total_gb == 24.0
    assert info.gpu_vram_free_gb == pytest.approx(17.6, abs=0.05)


def test_detect_hardware_nvidia_smi_nonzero_exit(fake_psutil: MagicMock) -> None:
    """nvidia-smi exists but returns non-zero → treated as no GPU."""
    which_p, run_p = _patch_nvidia_smi(present=True, returncode=1)
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()

    assert info.has_cuda is False
    assert info.gpu_name is None


def test_detect_hardware_nvidia_smi_timeout(fake_psutil: MagicMock) -> None:
    """nvidia-smi hanging → handled gracefully."""
    import subprocess as sp

    which_p = patch.object(
        hardware.shutil, "which", return_value="C:\\Windows\\system32\\nvidia-smi.exe"
    )
    run_p = patch.object(
        hardware.subprocess, "run", side_effect=sp.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)
    )
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()

    assert info.has_cuda is False
    assert info.gpu_vram_free_gb is None


def test_detect_hardware_nvidia_smi_empty_output(fake_psutil: MagicMock) -> None:
    """Empty stdout from nvidia-smi → no GPU detected."""
    which_p, run_p = _patch_nvidia_smi(present=True, stdout="\n\n")
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()
    assert info.has_cuda is False


def test_detect_hardware_nvidia_smi_garbled_vram(fake_psutil: MagicMock) -> None:
    """Unparseable VRAM → has_cuda True, vram fields None."""
    stdout = "NVIDIA Tesla V100, NaN, NaN, 555.42\n"
    which_p, run_p = _patch_nvidia_smi(present=True, stdout=stdout)
    with patch.object(hardware, "psutil", fake_psutil), which_p, run_p:
        info = detect_hardware()
    assert info.has_cuda is True
    assert info.gpu_vram_total_gb is None
    assert info.gpu_vram_free_gb is None


def test_detect_hardware_psutil_returns_none() -> None:
    """psutil sometimes returns None for cpu_count; we coerce to 0."""
    fake = MagicMock()
    fake.cpu_count.return_value = None
    fake.virtual_memory.return_value = SimpleNamespace(
        total=4 * 1024**3,
        available=2 * 1024**3,
    )
    which_p, run_p = _patch_nvidia_smi(present=False)
    with patch.object(hardware, "psutil", fake), which_p, run_p:
        info = detect_hardware()

    assert info.cpu_cores_physical == 0
    assert info.cpu_cores_logical == 0
    assert info.ram_total_gb == 4.0
    assert info.ram_available_gb == 2.0
