"""Unit tests for Stage 0: Audio Ingest and Normalization (backend/pipeline/ingest.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.pipeline.ingest import (
    ingest_audio,
    peak_normalize,
    resample_audio,
    to_mono,
)


def _generate_synthetic_tone(
    duration_sec: float = 1.0, sr: int = 16000, freq: float = 440.0, amplitude: float = 0.5
) -> np.ndarray:
    """Generate pure sine tone for unit testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_to_mono_1d() -> None:
    """1D array should pass through to_mono as float32."""
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    mono = to_mono(audio)
    assert mono.ndim == 1
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, audio)


def test_to_mono_2d_channels_last() -> None:
    """Nx2 stereo should be averaged across channels to 1D."""
    left = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    right = np.array([0.4, 0.6, 0.8], dtype=np.float32)
    stereo = np.column_stack((left, right))  # Shape (3, 2)
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert mono.shape == (3,)
    expected = (left + right) / 2.0
    np.testing.assert_allclose(mono, expected)


def test_to_mono_2d_channels_first() -> None:
    """2xN stereo should be averaged across channels to 1D."""
    left = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    right = np.array([0.4, 0.6, 0.8], dtype=np.float32)
    stereo = np.vstack((left, right))  # Shape (2, 3)
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert mono.shape == (3,)
    expected = (left + right) / 2.0
    np.testing.assert_allclose(mono, expected)


def test_resample_audio_48k_to_16k() -> None:
    """Resampling from 48 kHz to 16 kHz preserves duration and produces correct length."""
    sr_orig = 48000
    duration = 0.5
    tone_48k = _generate_synthetic_tone(duration_sec=duration, sr=sr_orig, freq=440.0)
    resampled = resample_audio(tone_48k, orig_sr=sr_orig, target_sr=16000)
    assert resampled.ndim == 1
    assert resampled.dtype == np.float32
    assert len(resampled) == int(16000 * duration)


def test_peak_normalize() -> None:
    """Peak normalization scales amplitude to target dBFS."""
    tone = _generate_synthetic_tone(duration_sec=1.0, sr=16000, freq=440.0, amplitude=0.25)
    target_db = -1.0
    target_amp = 10.0 ** (target_db / 20.0)
    normalized = peak_normalize(tone, target_db=target_db)

    assert normalized.dtype == np.float32
    assert np.max(np.abs(normalized)) == pytest.approx(target_amp, rel=1e-3)


def test_peak_normalize_silence() -> None:
    """Zero signal (silence) normalization remains zero without errors."""
    silence = np.zeros(1600, dtype=np.float32)
    normalized = peak_normalize(silence, target_db=-1.0)
    assert np.all(normalized == 0.0)


def test_ingest_audio_from_numpy() -> None:
    """Ingest from numpy array standardizes to 16 kHz mono float32 normalized."""
    tone = _generate_synthetic_tone(duration_sec=1.0, sr=44100, freq=440.0, amplitude=0.3)
    out = ingest_audio(audio_array=tone, sr=44100, target_sr=16000, target_db=-1.0)

    assert out.ndim == 1
    assert out.dtype == np.float32
    assert len(out) == 16000
    expected_peak = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(out)) == pytest.approx(expected_peak, rel=1e-3)


def test_ingest_audio_from_file() -> None:
    """Ingest loads WAV from disk and enforces 16 kHz mono float32."""
    sr_orig = 22050
    duration = 1.0
    tone = _generate_synthetic_tone(duration_sec=duration, sr=sr_orig, freq=300.0, amplitude=0.4)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        sf.write(str(tmp_path), tone, sr_orig)
        out = ingest_audio(file_path=tmp_path, target_sr=16000, target_db=-1.0)
        assert out.ndim == 1
        assert out.dtype == np.float32
        assert len(out) == 16000
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_ingest_audio_errors() -> None:
    """Ingest raises appropriate errors for invalid inputs."""
    with pytest.raises(ValueError, match="Either file_path or audio_array"):
        ingest_audio()

    with pytest.raises(ValueError, match="sr .* must be provided"):
        ingest_audio(audio_array=np.zeros(100, dtype=np.float32))

    with pytest.raises(ValueError, match="Audio array is empty"):
        ingest_audio(audio_array=np.array([], dtype=np.float32), sr=16000)

    nan_audio = np.array([0.1, np.nan, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or infinite"):
        ingest_audio(audio_array=nan_audio, sr=16000)
