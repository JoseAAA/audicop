"""Make the bundled NVIDIA CUDA DLLs discoverable on Windows.

Both the transcriber (ctranslate2) and the local LLM (llama.cpp) load native
extensions that link CUDA runtime DLLs (``cudart``, ``cublas``, ``cudnn``)
shipped by the ``nvidia-*-cu12`` PyPI wheels. On Windows the loader will not
find them unless their ``bin`` directories are on the search path, so this must
run **before** importing either native library.

Kept in one place so the transcriber and ``local_llm`` apply the exact same
fix. No-op on non-Windows systems; idempotent.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_cuda_dll_search_path() -> None:
    """Add the PyPI NVIDIA wheels' ``bin`` dirs to the Windows DLL search path.

    The faster-whisper docs note the ``pip install nvidia-cublas-cu12
    nvidia-cudnn-cu12`` route is "Linux only" — on Linux ``LD_LIBRARY_PATH`` is
    auto-populated, but on Windows the loader is stricter:

    - Python 3.8+ no longer searches ``PATH`` for ``.pyd`` dependencies.
    - :func:`os.add_dll_directory` only helps when the loader uses the
      ``LOAD_LIBRARY_SEARCH_USER_DIRS`` flag.
    - The compiled extensions call plain ``LoadLibraryW`` from C/C++ at runtime
      (when they actually need cuBLAS/cudart), and that uses the legacy
      "default search order" — which IS controlled by ``PATH``.

    So we do BOTH: :func:`os.add_dll_directory` (covers Python-side imports) and
    prepend the dirs to ``PATH`` (covers the runtime ``LoadLibrary`` calls).

    No-op on non-Windows. Idempotent: re-adding the same dir is harmless because
    Windows deduplicates the search path.
    """
    if sys.platform != "win32":
        return
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        logger.debug("nvidia/ not found under site-packages; skipping DLL setup.")
        return

    bin_dirs: list[str] = []
    for sub in nvidia_root.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            bin_dirs.append(str(bin_dir))

    if not bin_dirs:
        return

    for path_str in bin_dirs:
        try:
            os.add_dll_directory(path_str)
        except (OSError, FileNotFoundError):
            logger.debug("os.add_dll_directory(%s) failed", path_str, exc_info=True)

    existing_path = os.environ.get("PATH", "")
    new_path = os.pathsep.join(bin_dirs)
    if existing_path:
        new_path = new_path + os.pathsep + existing_path
    os.environ["PATH"] = new_path
    logger.debug("Prepended NVIDIA bin dirs to PATH: %s", bin_dirs)
