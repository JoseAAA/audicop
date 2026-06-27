"""Tests for `app.adapters.transcriber`.

`WhisperModel` is mocked so the suite never has to download real model
weights. We only verify our wrapper's behavior: validation, lazy load,
segment streaming, error mapping, and the Windows no-symlink fallback.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.adapters import transcriber
from app.adapters.transcriber import (
    Transcriber,
    TranscriptionError,
    TranscriptSegment,
)


@pytest.fixture(autouse=True)
def _reset_symlink_probe() -> None:
    """Each test starts with a fresh symlink-probe cache."""
    transcriber._SYMLINK_PROBE_RESULT = None  # test-only: reset module-level cache


def _wav_path(tmp_path: Path) -> Path:
    """Return a path to a non-empty file Pretending to be a WAV."""
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFF....WAVEfmt ")  # arbitrary; we mock decoding
    return p


@pytest.fixture
def fake_segments() -> list[SimpleNamespace]:
    """Three fake segments simulating faster-whisper output."""
    return [
        SimpleNamespace(start=0.0, end=1.5, text=" Hola mundo "),
        SimpleNamespace(start=1.5, end=3.2, text="esto es una prueba"),
        SimpleNamespace(start=3.2, end=4.0, text=""),  # silent segment
    ]


@pytest.fixture
def fake_info() -> SimpleNamespace:
    """Fake `TranscriptionInfo` returned by faster-whisper."""
    return SimpleNamespace(language="es", language_probability=0.97, duration=4.0)


def test_invalid_model_size_raises() -> None:
    with pytest.raises(ValueError, match="modelo no soportado"):
        Transcriber(model_size="huge", compute_type="int8", device="cpu")


def test_invalid_compute_type_raises() -> None:
    with pytest.raises(ValueError, match="compute_type no soportado"):
        Transcriber(model_size="tiny", compute_type="bfloat42", device="cpu")


def test_invalid_device_raises() -> None:
    with pytest.raises(ValueError, match="device debe ser"):
        Transcriber(model_size="tiny", compute_type="int8", device="tpu")


def test_load_caches_model() -> None:
    """`load()` builds a `WhisperModel` once and reuses it on subsequent calls."""
    fake_model = MagicMock()
    with patch.object(transcriber, "WhisperModel", return_value=fake_model) as ctor:
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        first = t.load()
        second = t.load()

    assert first is second is fake_model
    ctor.assert_called_once()


def test_load_failure_wraps_exception() -> None:
    """If `WhisperModel(...)` blows up, we surface a `TranscriptionError`."""
    with patch.object(transcriber, "WhisperModel", side_effect=RuntimeError("boom")):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        with pytest.raises(TranscriptionError, match="No se pudo cargar el modelo"):
            t.load()


def test_transcribe_streams_segments(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
    fake_info: SimpleNamespace,
) -> None:
    """`transcribe()` returns an iterator over `TranscriptSegment` and metadata."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), fake_info)

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        segs_iter, info = t.transcribe(_wav_path(tmp_path), language="es")

    segments = list(segs_iter)
    assert len(segments) == 3
    assert all(isinstance(s, TranscriptSegment) for s in segments)
    assert segments[0].text == "Hola mundo"  # stripped
    assert segments[1].start == 1.5
    assert info.language == "es"
    assert info.language_probability == pytest.approx(0.97)
    assert info.duration == pytest.approx(4.0)


def test_transcribe_passes_options(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
    fake_info: SimpleNamespace,
) -> None:
    """language/task/beam_size/vad_filter are forwarded to the underlying model."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), fake_info)

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        t.transcribe(
            _wav_path(tmp_path),
            language="en",
            task="translate",
            beam_size=3,
            vad_filter=False,
        )

    fake_model.transcribe.assert_called_once()
    kwargs = fake_model.transcribe.call_args.kwargs
    assert kwargs["language"] == "en"
    assert kwargs["task"] == "translate"
    assert kwargs["beam_size"] == 3
    assert kwargs["vad_filter"] is False


def test_transcribe_missing_file(tmp_path: Path) -> None:
    """Non-existent WAV raises FileNotFoundError before any model work."""
    fake_model = MagicMock()
    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        with pytest.raises(FileNotFoundError):
            t.transcribe(tmp_path / "missing.wav")
    fake_model.transcribe.assert_not_called()


def test_transcribe_invalid_task_raises(tmp_path: Path) -> None:
    """task must be 'transcribe' or 'translate'."""
    t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
    with pytest.raises(ValueError, match="task debe estar en"):
        t.transcribe(_wav_path(tmp_path), task="summarize")


def test_transcribe_decoding_failure(tmp_path: Path) -> None:
    """Underlying decode error becomes a TranscriptionError."""
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("CUDA OOM")

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        with pytest.raises(TranscriptionError, match="Fallo decodificando"):
            t.transcribe(_wav_path(tmp_path))


def test_transcribe_handles_missing_info_fields(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
) -> None:
    """If faster-whisper returns sparse info, we still build a TranscriptionInfo."""
    sparse_info = SimpleNamespace()  # no attributes at all
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), sparse_info)

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        _, info = t.transcribe(_wav_path(tmp_path))

    assert info.language == "unknown"
    assert info.language_probability == 0.0
    assert info.duration == 0.0


# ---------------------------------------------------------------------------
# Symlink probe + no-symlink fallback (Windows)
# ---------------------------------------------------------------------------


def test_symlink_probe_returns_true_on_non_windows() -> None:
    """On Linux/macOS we never run the probe — symlinks always work."""
    with patch.object(transcriber.sys, "platform", "linux"):
        assert transcriber._hf_symlinks_supported() is True


def test_symlink_probe_caches_result(tmp_path: Path) -> None:
    """The probe runs once; subsequent calls return the cached answer."""
    fake_symlink = MagicMock()

    with (
        patch.object(transcriber.sys, "platform", "win32"),
        patch.object(transcriber, "_huggingface_cache_root", return_value=tmp_path),
        patch.object(transcriber.os, "symlink", fake_symlink),
    ):
        first = transcriber._hf_symlinks_supported()
        second = transcriber._hf_symlinks_supported()

    assert first is True
    assert second is True
    fake_symlink.assert_called_once()


def test_symlink_probe_returns_false_on_windows_perm_error(tmp_path: Path) -> None:
    """A WinError-1314-like OSError makes the probe report 'not supported'."""
    err = OSError("[WinError 1314] required privilege not held")
    err.winerror = 1314  # type: ignore[attr-defined]
    fake_symlink = MagicMock(side_effect=err)

    with (
        patch.object(transcriber.sys, "platform", "win32"),
        patch.object(transcriber, "_huggingface_cache_root", return_value=tmp_path),
        patch.object(transcriber.os, "symlink", fake_symlink),
    ):
        assert transcriber._hf_symlinks_supported() is False


def test_load_uses_local_dir_when_symlinks_unsupported(tmp_path: Path) -> None:
    """If the probe says 'no symlinks', `load()` pre-downloads to local dir."""
    fake_model = MagicMock()
    download_target = tmp_path / "model_dir"
    download_target.mkdir()
    fake_download = MagicMock(return_value=download_target)

    with (
        patch.object(transcriber, "WhisperModel", return_value=fake_model) as ctor,
        patch.object(transcriber, "_hf_symlinks_supported", return_value=False),
        patch.object(transcriber, "_download_model_no_symlinks", fake_download),
    ):
        t = Transcriber(model_size="small", compute_type="int8", device="cpu")
        t.load()

    fake_download.assert_called_once_with("small")
    # WhisperModel is fed the local path, not the model name string
    args, kwargs = ctor.call_args
    assert args[0] == str(download_target)
    assert kwargs["device"] == "cpu"
    assert kwargs["download_root"] is None


def test_load_uses_model_name_when_symlinks_supported() -> None:
    """If symlinks work, `load()` passes the model size string to WhisperModel."""
    fake_model = MagicMock()
    fake_download = MagicMock()

    with (
        patch.object(transcriber, "WhisperModel", return_value=fake_model) as ctor,
        patch.object(transcriber, "_hf_symlinks_supported", return_value=True),
        patch.object(transcriber, "_download_model_no_symlinks", fake_download),
    ):
        t = Transcriber(model_size="small", compute_type="int8", device="cpu")
        t.load()

    fake_download.assert_not_called()
    args, _ = ctor.call_args
    assert args[0] == "small"  # raw model size, HF cache will handle it


def test_is_model_cached_checks_audicop_dir(tmp_path: Path) -> None:
    """`is_model_cached` returns True when only the no-symlink cache has it."""
    audicop_dir = tmp_path / "audicop_cache" / "Systran--faster-whisper-small"
    audicop_dir.mkdir(parents=True)
    (audicop_dir / "model.bin").write_bytes(b"not really a model")

    with (
        patch.object(transcriber, "_huggingface_cache_root", return_value=tmp_path / "hf"),
        patch.object(transcriber, "_audicop_cache_root", return_value=tmp_path / "audicop_cache"),
    ):
        assert transcriber.is_model_cached("small") is True
        assert transcriber.is_model_cached("tiny") is False


def test_download_model_no_symlinks_calls_snapshot_download(tmp_path: Path) -> None:
    """`_download_model_no_symlinks` calls snapshot_download with local_dir=our path."""
    fake_snapshot = MagicMock()
    fake_module = MagicMock()
    fake_module.snapshot_download = fake_snapshot

    with (
        patch.object(transcriber, "_audicop_cache_root", return_value=tmp_path / "audicop"),
        patch.dict("sys.modules", {"huggingface_hub": fake_module}),
    ):
        result = transcriber._download_model_no_symlinks("tiny")

    fake_snapshot.assert_called_once()
    kwargs = fake_snapshot.call_args.kwargs
    assert kwargs["repo_id"] == "Systran/faster-whisper-tiny"
    assert kwargs["local_dir"] == str(tmp_path / "audicop" / "Systran--faster-whisper-tiny")
    assert result == tmp_path / "audicop" / "Systran--faster-whisper-tiny"


def test_download_model_no_symlinks_wraps_errors(tmp_path: Path) -> None:
    """If snapshot_download blows up we surface a TranscriptionError."""
    fake_module = MagicMock()
    fake_module.snapshot_download = MagicMock(side_effect=RuntimeError("network kaput"))

    with (
        patch.object(transcriber, "_audicop_cache_root", return_value=tmp_path / "audicop"),
        patch.dict("sys.modules", {"huggingface_hub": fake_module}),
        pytest.raises(TranscriptionError, match="No se pudo descargar"),
    ):
        transcriber._download_model_no_symlinks("tiny")


# ---------------------------------------------------------------------------
# Repo-id resolution (turbo lives outside the Systran org)
# ---------------------------------------------------------------------------


def test_repo_id_for_resolves_turbo_and_systran() -> None:
    """`large-v3-turbo` resolves to its mobiuslabsgmbh repo; classic sizes to Systran."""
    assert (
        transcriber._repo_id_for("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    )
    assert transcriber._repo_id_for("small") == "Systran/faster-whisper-small"


def test_repo_id_for_falls_back_to_template_when_mapping_missing() -> None:
    """If faster-whisper's mapping lacks the size, we use the Systran template."""
    assert transcriber._repo_id_for("not-a-real-size") == "Systran/faster-whisper-not-a-real-size"


def test_download_model_no_symlinks_turbo_uses_correct_repo(tmp_path: Path) -> None:
    """Turbo downloads from mobiuslabsgmbh into a matching local dir name."""
    fake_snapshot = MagicMock()
    fake_module = MagicMock()
    fake_module.snapshot_download = fake_snapshot

    with (
        patch.object(transcriber, "_audicop_cache_root", return_value=tmp_path / "audicop"),
        patch.dict("sys.modules", {"huggingface_hub": fake_module}),
    ):
        result = transcriber._download_model_no_symlinks("large-v3-turbo")

    kwargs = fake_snapshot.call_args.kwargs
    assert kwargs["repo_id"] == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert result.name == "mobiuslabsgmbh--faster-whisper-large-v3-turbo"


# ---------------------------------------------------------------------------
# Batched pipeline (GPU) + initial_prompt
# ---------------------------------------------------------------------------


def test_transcribe_passes_initial_prompt_on_cpu(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
    fake_info: SimpleNamespace,
) -> None:
    """The vocabulary hint reaches the underlying model on the CPU path."""
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), fake_info)

    with patch.object(transcriber, "WhisperModel", return_value=fake_model):
        t = Transcriber(model_size="tiny", compute_type="int8", device="cpu")
        t.transcribe(_wav_path(tmp_path), initial_prompt="Kubernetes, Acme")

    assert fake_model.transcribe.call_args.kwargs["initial_prompt"] == "Kubernetes, Acme"


def test_transcribe_cuda_uses_batched_pipeline(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
    fake_info: SimpleNamespace,
) -> None:
    """On CUDA we decode via BatchedInferencePipeline, not model.transcribe."""
    fake_model = MagicMock()
    fake_pipeline = MagicMock()
    fake_pipeline.transcribe.return_value = (iter(fake_segments), fake_info)

    with (
        patch.object(transcriber, "WhisperModel", return_value=fake_model),
        patch.object(transcriber, "BatchedInferencePipeline", return_value=fake_pipeline) as bip,
    ):
        t = Transcriber(model_size="large-v3-turbo", compute_type="float16", device="cuda")
        segs, _ = t.transcribe(_wav_path(tmp_path), language="es", initial_prompt="Acme")
        list(segs)  # drive the generator

    bip.assert_called_once_with(model=fake_model)
    fake_model.transcribe.assert_not_called()
    kwargs = fake_pipeline.transcribe.call_args.kwargs
    assert kwargs["batch_size"] == transcriber.config.DEFAULT_BATCH_SIZE
    assert kwargs["initial_prompt"] == "Acme"
    assert kwargs["language"] == "es"


def test_batched_pipeline_is_cached(
    tmp_path: Path,
    fake_segments: list[SimpleNamespace],
    fake_info: SimpleNamespace,
) -> None:
    """Two decodes on the same transcriber reuse one BatchedInferencePipeline."""
    fake_pipeline = MagicMock()
    fake_pipeline.transcribe.side_effect = lambda *a, **k: (iter(fake_segments), fake_info)

    with (
        patch.object(transcriber, "WhisperModel", return_value=MagicMock()),
        patch.object(transcriber, "BatchedInferencePipeline", return_value=fake_pipeline) as bip,
    ):
        t = Transcriber(model_size="large-v3-turbo", compute_type="float16", device="cuda")
        list(t.transcribe(_wav_path(tmp_path))[0])
        list(t.transcribe(_wav_path(tmp_path))[0])

    bip.assert_called_once()
