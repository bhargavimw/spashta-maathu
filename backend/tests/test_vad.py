"""Unit tests for Stage 1: Voice Activity Detection (backend/pipeline/vad.py)."""

import numpy as np
import pytest

from backend.pipeline.vad import (
    AudioGap,
    SpeechSegment,
    VADResult,
    compute_gaps,
    detect_speech_regions,
)


def _generate_synthetic_speech_burst(
    duration_sec: float = 1.0, sr: int = 16000, fundamental_freq: float = 150.0
) -> np.ndarray:
    """Generate harmonic speech-like signal with envelope for VAD testing."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    # Fundamental + harmonics to simulate vocal cords & formant structure
    signal = (
        0.5 * np.sin(2 * np.pi * fundamental_freq * t)
        + 0.3 * np.sin(2 * np.pi * fundamental_freq * 2 * t)
        + 0.2 * np.sin(2 * np.pi * fundamental_freq * 3 * t)
        + 0.1 * np.sin(2 * np.pi * fundamental_freq * 4 * t)
    ).astype(np.float32)

    # Smooth attack and decay envelope
    envelope = np.ones(n_samples, dtype=np.float32)
    ramp_samples = min(int(0.05 * sr), n_samples // 4)
    if ramp_samples > 0:
        envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples)
        envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples)

    return (signal * envelope).astype(np.float32)


def test_compute_gaps_pure_logic() -> None:
    """Verify compute_gaps pure function math for leading, intermediate, and trailing gaps."""
    sr = 16000
    total_samples = 160000  # 10.0 seconds

    # Two segments:
    # 1: 1.0s to 2.5s (16000 to 40000)
    # 2: 4.0s to 7.0s (64000 to 112000)
    segments = [
        SpeechSegment(start_sample=16000, end_sample=40000, start_sec=1.0, end_sec=2.5),
        SpeechSegment(start_sample=64000, end_sample=112000, start_sec=4.0, end_sec=7.0),
    ]

    gaps = compute_gaps(segments, total_samples=total_samples, sample_rate=sr)

    assert len(gaps) == 3

    # Leading gap: 0.0s to 1.0s
    assert gaps[0].start_sample == 0
    assert gaps[0].end_sample == 16000
    assert gaps[0].start_sec == 0.0
    assert gaps[0].end_sec == 1.0
    assert gaps[0].duration_sec == pytest.approx(1.0)
    assert gaps[0].duration_ms == pytest.approx(1000.0)

    # Inter-segment gap: 2.5s to 4.0s (40000 to 64000)
    assert gaps[1].start_sample == 40000
    assert gaps[1].end_sample == 64000
    assert gaps[1].start_sec == pytest.approx(2.5)
    assert gaps[1].end_sec == pytest.approx(4.0)
    assert gaps[1].duration_sec == pytest.approx(1.5)

    # Trailing gap: 7.0s to 10.0s (112000 to 160000)
    assert gaps[2].start_sample == 112000
    assert gaps[2].end_sample == 160000
    assert gaps[2].start_sec == pytest.approx(7.0)
    assert gaps[2].end_sec == pytest.approx(10.0)
    assert gaps[2].duration_sec == pytest.approx(3.0)


def test_compute_gaps_all_silence() -> None:
    """When no speech is detected, entire duration is represented as one gap."""
    sr = 16000
    total_samples = 32000  # 2.0s
    gaps = compute_gaps([], total_samples=total_samples, sample_rate=sr)

    assert len(gaps) == 1
    assert gaps[0].start_sec == 0.0
    assert gaps[0].end_sec == 2.0
    assert gaps[0].duration_sec == 2.0


def test_compute_gaps_no_silence() -> None:
    """When speech spans entire clip [0, total_samples], gaps list is empty."""
    sr = 16000
    total_samples = 16000
    segments = [SpeechSegment(start_sample=0, end_sample=16000, start_sec=0.0, end_sec=1.0)]
    gaps = compute_gaps(segments, total_samples=total_samples, sample_rate=sr)

    assert len(gaps) == 0


def test_vad_on_silence() -> None:
    """VAD on pure silence produces zero speech segments and one gap."""
    sr = 16000
    duration = 2.0
    silence = np.zeros(int(sr * duration), dtype=np.float32)

    result = detect_speech_regions(silence, sample_rate=sr)

    assert isinstance(result, VADResult)
    assert len(result.segments) == 0
    assert len(result.gaps) == 1
    assert result.gaps[0].duration_sec == pytest.approx(duration)


def test_vad_on_synthetic_speech_bursts() -> None:
    """VAD detects speech bursts and measures intermediate pauses."""
    sr = 16000
    # Structure:
    # 0.5s silence + 1.5s speech + 1.0s silence + 1.5s speech + 0.5s silence = 5.0s
    leading_silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    burst1 = _generate_synthetic_speech_burst(duration_sec=1.5, sr=sr, fundamental_freq=180.0)
    mid_silence = np.zeros(int(1.0 * sr), dtype=np.float32)
    burst2 = _generate_synthetic_speech_burst(duration_sec=1.5, sr=sr, fundamental_freq=220.0)
    trailing_silence = np.zeros(int(0.5 * sr), dtype=np.float32)

    audio = np.concatenate([leading_silence, burst1, mid_silence, burst2, trailing_silence])
    assert len(audio) == 5 * sr

    result = detect_speech_regions(audio, sample_rate=sr)

    assert len(result.segments) >= 1
    assert len(result.gaps) >= 1
    assert result.duration_sec == pytest.approx(5.0)


def test_vad_invariants_and_validation() -> None:
    """VAD rejects invalid sample rate, multi-channel or empty input."""
    sr = 16000
    audio_valid = np.zeros(sr, dtype=np.float32)

    with pytest.raises(ValueError, match="sample_rate=16000"):
        detect_speech_regions(audio_valid, sample_rate=22050)

    with pytest.raises(ValueError, match="Audio must be 1D mono"):
        detect_speech_regions(np.zeros((sr, 2), dtype=np.float32), sample_rate=sr)

    with pytest.raises(ValueError, match="dtype must be float32"):
        detect_speech_regions(np.zeros(sr, dtype=np.float64), sample_rate=sr)

    with pytest.raises(ValueError, match="Audio array is empty"):
        detect_speech_regions(np.array([], dtype=np.float32), sample_rate=sr)
