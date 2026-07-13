"""Tests for `app.adapters.local_llm` (the on-device llama.cpp adapter).

``llama_cpp`` and the model download are faked, so the suite never compiles
the native package nor touches the network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.adapters import local_llm
from app.adapters.local_llm import ChatMessage, LLMError


def _chunk(text: str) -> dict[str, object]:
    """Build a llama.cpp streaming chat chunk carrying ``text`` as delta."""
    return {"choices": [{"delta": {"content": text}}]}


def _hf_module(*, returns: str | None = None, raises: Exception | None = None) -> MagicMock:
    """Build a fake ``huggingface_hub`` module exposing ``hf_hub_download``."""
    mod = MagicMock()
    if raises is not None:
        mod.hf_hub_download.side_effect = raises
    else:
        mod.hf_hub_download.return_value = returns
    return mod


def test_unload_clears_model() -> None:
    obj = local_llm.LocalLLM(repo_id="r", filename="m.gguf")
    obj._model = object()
    obj.unload()
    assert obj._model is None


def test_ensure_downloaded_returns_existing_without_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_llm, "_gguf_cache_dir", lambda: tmp_path)
    existing = tmp_path / "m.gguf"
    existing.write_bytes(b"data")
    # A download attempt would raise; reaching the return proves we skipped it.
    with patch.dict("sys.modules", {"huggingface_hub": _hf_module(raises=RuntimeError("nope"))}):
        assert local_llm.ensure_downloaded("r", "m.gguf") == existing


def test_ensure_downloaded_fetches_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_llm, "_gguf_cache_dir", lambda: tmp_path)
    fake_hf = _hf_module(returns=str(tmp_path / "m.gguf"))
    with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
        path = local_llm.ensure_downloaded("r", "m.gguf")
    assert Path(path).name == "m.gguf"
    fake_hf.hf_hub_download.assert_called_once()


def test_ensure_downloaded_wraps_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_llm, "_gguf_cache_dir", lambda: tmp_path)
    with (
        patch.dict("sys.modules", {"huggingface_hub": _hf_module(raises=RuntimeError("down"))}),
        pytest.raises(LLMError, match="No se pudo descargar"),
    ):
        local_llm.ensure_downloaded("r", "m.gguf")


def test_stream_chat_yields_chunks() -> None:
    fake_model = MagicMock()
    fake_model.create_chat_completion.return_value = iter(
        [_chunk("Resu"), _chunk("men"), _chunk("")]
    )
    fake_llama = MagicMock()
    fake_llama.Llama.return_value = fake_model

    obj = local_llm.LocalLLM(repo_id="r", filename="m.gguf", device="cpu", n_ctx=512)
    with (
        patch.dict("sys.modules", {"llama_cpp": fake_llama}),
        patch.object(local_llm, "ensure_downloaded", return_value=Path("m.gguf")),
    ):
        out = list(obj.stream_chat(system="sys", messages=[ChatMessage(role="user", content="hi")]))

    assert "".join(out) == "Resumen"
    _, kwargs = fake_model.create_chat_completion.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["messages"][1] == {"role": "user", "content": "hi"}
    assert kwargs["stream"] is True


def test_load_missing_sdk_is_friendly() -> None:
    obj = local_llm.LocalLLM(repo_id="r", filename="m.gguf")
    with (
        patch.dict("sys.modules", {"llama_cpp": None}),
        pytest.raises(LLMError, match="no está instalado"),
    ):
        obj.load()


def test_load_surfaces_download_failure() -> None:
    obj = local_llm.LocalLLM(repo_id="r", filename="m.gguf")
    fake_llama = MagicMock()
    with (
        patch.dict("sys.modules", {"llama_cpp": fake_llama}),
        patch.object(local_llm, "ensure_downloaded", side_effect=LLMError("no se pudo descargar")),
        pytest.raises(LLMError, match="descargar"),
    ):
        obj.load()


def test_load_illegal_instruction_is_friendly() -> None:
    """An illegal-instruction crash (incompatible CPU) becomes a clear message,
    not a raw 0xc000001d hex code — and points the user at re-running the launcher."""
    obj = local_llm.LocalLLM(repo_id="r", filename="m.gguf")
    fake_llama = MagicMock()
    crash = OSError("[WinError -1073741795] Windows Error 0xc000001d")
    crash.winerror = -1073741795  # type: ignore[attr-defined]
    fake_llama.Llama.side_effect = crash
    with (
        patch.dict("sys.modules", {"llama_cpp": fake_llama}),
        patch.object(local_llm, "ensure_downloaded", return_value=Path("m.gguf")),
        pytest.raises(LLMError, match="no es compatible"),
    ):
        obj.load()


def test_load_other_error_keeps_detail() -> None:
    """A non-illegal-instruction load failure still surfaces the raw detail."""
    obj = local_llm.LocalLLM(repo_id="r", filename="m.gguf")
    fake_llama = MagicMock()
    fake_llama.Llama.side_effect = RuntimeError("gguf corrupto")
    with (
        patch.dict("sys.modules", {"llama_cpp": fake_llama}),
        patch.object(local_llm, "ensure_downloaded", return_value=Path("m.gguf")),
        pytest.raises(LLMError, match="gguf corrupto"),
    ):
        obj.load()


def test_get_active_only_when_loaded() -> None:
    """`get_active` returns the singleton only once its model is in memory."""
    inst = local_llm.get_local_llm(
        repo_id="r", filename="act.gguf", device="cpu", n_gpu_layers=0, n_ctx=512
    )
    assert local_llm.get_active() is None  # created but not loaded yet
    inst._model = object()  # simulate a loaded model
    assert local_llm.get_active() is inst
    inst.unload()
    assert local_llm.get_active() is None


def test_get_local_llm_caches_then_swaps_on_change() -> None:
    a = local_llm.get_local_llm(
        repo_id="r", filename="a.gguf", device="cpu", n_gpu_layers=0, n_ctx=512
    )
    again = local_llm.get_local_llm(
        repo_id="r", filename="a.gguf", device="cpu", n_gpu_layers=0, n_ctx=512
    )
    assert again is a  # same params -> cached instance reused
    other = local_llm.get_local_llm(
        repo_id="r", filename="b.gguf", device="cpu", n_gpu_layers=0, n_ctx=512
    )
    assert other is not a  # different model -> new instance


def test_is_model_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_llm, "_gguf_cache_dir", lambda: tmp_path)
    assert local_llm.is_model_cached("nope.gguf") is False
    (tmp_path / "yes.gguf").write_bytes(b"x")
    assert local_llm.is_model_cached("yes.gguf") is True
