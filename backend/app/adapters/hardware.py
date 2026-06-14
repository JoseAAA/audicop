"""Hardware detection: OS, CPU, RAM, and NVIDIA GPU information.

GPU detection uses ``nvidia-smi`` rather than PyTorch. This is more
reliable because ``nvidia-smi`` ships with every NVIDIA driver install,
while ``torch.cuda.is_available()`` returns ``False`` whenever the
installed torch wheel happens to be the CPU-only build — which is a
common surprise on Windows. We do not depend on torch at all.

The functions never raise: every probe is wrapped so that a single
failure does not block the rest of the detection.
"""

from __future__ import annotations

import logging
import math
import platform
import shutil
import subprocess
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

_BYTES_PER_GB: float = 1024.0**3
_NVIDIA_SMI_TIMEOUT_S: float = 5.0


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Snapshot of the local machine's relevant hardware.

    The ``ram_available_gb`` and ``gpu_vram_free_gb`` fields are the
    point-in-time figures the recommender uses: total memory is rarely
    a useful signal because the OS and other apps consume a meaningful
    chunk of it.

    Attributes:
        os_name: Operating system name (e.g. ``"Windows"``, ``"Linux"``).
        os_release: OS release string from :mod:`platform`.
        cpu_cores_physical: Number of physical CPU cores; ``0`` if unknown.
        cpu_cores_logical: Number of logical CPU cores (threads); ``0`` if unknown.
        cpu_brand: CPU model string from :func:`platform.processor`.
        ram_total_gb: Total system RAM in gigabytes.
        ram_available_gb: RAM currently free for new allocations, in GB
            (``psutil.virtual_memory().available``).
        has_cuda: ``True`` if a CUDA-capable NVIDIA GPU is detected.
        gpu_name: Marketing name of the first NVIDIA GPU (or ``None``).
        gpu_vram_total_gb: Total VRAM of the first NVIDIA GPU (or ``None``).
        gpu_vram_free_gb: Free VRAM at the moment of detection (or ``None``).
        gpu_driver_version: NVIDIA driver version string (or ``None``).
    """

    os_name: str
    os_release: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_brand: str
    ram_total_gb: float
    ram_available_gb: float
    has_cuda: bool
    gpu_name: str | None
    gpu_vram_total_gb: float | None
    gpu_vram_free_gb: float | None
    gpu_driver_version: str | None


def _detect_cpu() -> tuple[int, int]:
    """Return ``(physical_cores, logical_cores)``.

    Returns ``0`` for either value if :func:`psutil.cpu_count` returns
    ``None`` (some virtualized environments).
    """
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    return physical, logical


def _detect_cpu_brand() -> str:
    """Return the CPU model string, or an empty string if unknown."""
    try:
        brand = platform.processor() or ""
    except Exception:  # pragma: no cover - defensive
        return ""
    return brand.strip()


def _detect_ram_gb() -> tuple[float, float]:
    """Return ``(total_gb, available_gb)`` system RAM, each rounded to one decimal."""
    vm = psutil.virtual_memory()
    total = float(round(vm.total / _BYTES_PER_GB, 1))
    available = float(round(vm.available / _BYTES_PER_GB, 1))
    return total, available


def _run_nvidia_smi() -> str | None:
    """Invoke ``nvidia-smi`` and return its stdout, or ``None`` on any failure.

    Queries the first GPU's name, total memory, free memory (both in MiB)
    and the driver version. Output is one CSV line per GPU.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    cmd = [
        nvidia_smi,
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("nvidia-smi falló o expiró", exc_info=True)
        return None
    if result.returncode != 0:
        logger.debug("nvidia-smi rc=%s stderr=%s", result.returncode, result.stderr)
        return None
    return result.stdout


def _mib_to_gb(value: str) -> float | None:
    """Parse a MiB value from nvidia-smi CSV and return GB rounded to 1 decimal.

    Returns ``None`` for empty, non-numeric, NaN, or non-finite values.
    """
    try:
        mib = float(value)
    except ValueError:
        return None
    if not math.isfinite(mib):
        return None
    return round(mib / 1024.0, 1)


def _parse_nvidia_smi(
    stdout: str,
) -> tuple[str, float | None, float | None, str | None] | None:
    """Parse the first GPU from ``nvidia-smi`` CSV output.

    Returns ``(name, vram_total_gb, vram_free_gb, driver_version)`` or
    ``None`` if the output is empty or unparseable.
    """
    line = next((row for row in stdout.splitlines() if row.strip()), None)
    if line is None:
        return None
    parts = [p.strip() for p in line.split(",")]
    if not parts or not parts[0]:
        return None
    name = parts[0]
    vram_total = _mib_to_gb(parts[1]) if len(parts) >= 2 else None
    vram_free = _mib_to_gb(parts[2]) if len(parts) >= 3 else None
    driver = parts[3] if len(parts) >= 4 and parts[3] else None
    return name, vram_total, vram_free, driver


def _detect_nvidia_gpu() -> tuple[bool, str | None, float | None, float | None, str | None]:
    """Detect the first NVIDIA GPU via ``nvidia-smi``.

    Returns:
        Tuple ``(has_cuda, gpu_name, gpu_vram_total_gb, gpu_vram_free_gb,
        gpu_driver_version)``. On any failure returns all-None / False.
    """
    stdout = _run_nvidia_smi()
    if stdout is None:
        return False, None, None, None, None
    parsed = _parse_nvidia_smi(stdout)
    if parsed is None:
        return False, None, None, None, None
    name, vram_total, vram_free, driver = parsed
    return True, name, vram_total, vram_free, driver


def detect_hardware() -> HardwareInfo:
    """Probe the host machine and return a :class:`HardwareInfo` snapshot.

    The function never raises: every probe is wrapped so that a single
    failure does not block the rest of the detection.

    Returns:
        A :class:`HardwareInfo` describing the local hardware.
    """
    physical, logical = _detect_cpu()
    ram_total, ram_available = _detect_ram_gb()
    has_cuda, gpu_name, gpu_vram_total, gpu_vram_free, gpu_driver = _detect_nvidia_gpu()
    info = HardwareInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        cpu_cores_physical=physical,
        cpu_cores_logical=logical,
        cpu_brand=_detect_cpu_brand(),
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total,
        gpu_vram_free_gb=gpu_vram_free,
        gpu_driver_version=gpu_driver,
    )
    logger.debug("Detected hardware: %s", info)
    return info
