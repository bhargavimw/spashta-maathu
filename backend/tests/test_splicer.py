"""Unit tests for Stage 5A: Audio Splicer, Crossfader, and Click Detector."""

import numpy as np
import pytest

from backend.pipeline.splicer import (
    SpliceResult,
    crossfade_equal_power,
    detect_join_clicks,
    snap_to_zero_crossing,
    splice_audio,
)
from backend.pipeline.vad import SpeechSegment, VADResult


def _create_synthetic_burst(duration_sec: float, sr: int = 16000, f0: float = 200.0) -> np.ndarray:
    """Create harmonic signal with tapered ends for unit testing."""
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * f0 * t).astype(np.float32)
    return sig


def test_snap_to_zero_crossing_finds_exact_zero() -> None:
    """Snapping finds exact zero crossing near target index."""
    sr = 16000
    # Construct a sine wave at 100 Hz (zero-crossings every 80 samples)
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)

    # Near index 85, a zero crossing occurs at index 80
    snapped = snap_to_zero_crossing(audio, index=85, window_samples=20)
    assert abs(audio[snapped]) < 0.05
    assert 78 <= snapped <= 82


def test_snap_to_zero_crossing_boundary_guards() -> None:
    """Snapping safely guards against indices outside array boundaries."""
    audio = np.ones(100, dtype=np.float32)
    assert snap_to_zero_crossing(audio, index=-10, window_samples=50) == 0
    assert snap_to_zero_crossing(audio, index=500, window_samples=50) == 99


def test_crossfade_equal_power_length_and_finite() -> None:
    """Equal-power crossfade joins segments with correct overlap length and no NaNs."""
    sr = 16000
    cf_samples = 240  # 15 ms
    seg_a = np.ones(1000, dtype=np.float32) * 0.5
    seg_b = np.ones(1000, dtype=np.float32) * 0.5

    joined = crossfade_equal_power(seg_a, seg_b, crossfade_samples=cf_samples)
    expected_length = len(seg_a) + len(seg_b) - cf_samples

    assert len(joined) == expected_length
    assert np.all(np.isfinite(joined))
    # In equal-power blend of equal constant values: 0.5*cos + 0.5*sin ~ 0.5 at edges, ~0.7 at center
    assert np.all(joined > 0.0)


def test_crossfade_equal_power_short_inputs() -> None:
    """Segments shorter than crossfade_samples adaptively crossfade without raising errors."""
    seg_a = np.ones(50, dtype=np.float32)
    seg_b = np.ones(60, dtype=np.float32)

    joined = crossfade_equal_power(seg_a, seg_b, crossfade_samples=240)
    assert np.all(np.isfinite(joined))
    assert len(joined) > 0


def test_detect_join_clicks_clean_signal() -> None:
    """Smooth joined signal produces zero click detections."""
    t = np.linspace(0, 1.0, 1600, endpoint=False, dtype=np.float32)
    smooth_audio = np.sin(2 * np.pi * 5.0 * t).astype(np.float32)

    clicks = detect_join_clicks(smooth_audio, join_indices=[400, 800, 1200], max_step_jump=0.15)
    assert len(clicks) == 0


def test_detect_join_clicks_flags_sharp_jump() -> None:
    """Artificial step discontinuity at a join point is detected as a click."""
    audio = np.zeros(1000, dtype=np.float32)
    # Insert discontinuous jump of 0.8 at index 500
    audio[500:] = 0.8

    clicks = detect_join_clicks(audio, join_indices=[500], max_step_jump=0.15)
    assert 500 in clicks


def test_splice_audio_pure_silence() -> None:
    """Pure silence audio is trimmed to edge silence limit (0.10 s)."""
    sr = 16000
    duration = 4.0
    silence = np.zeros(int(sr * duration), dtype=np.float32)
    vad_res = VADResult(
        segments=[],
        gaps=[],
        sample_rate=sr,
        duration_sec=duration,
        total_samples=len(silence),
    )

    result = splice_audio(silence, vad_res)
    assert isinstance(result, SpliceResult)
    assert result.output_duration_sec == pytest.approx(0.10, rel=1e-2)
    assert result.output_duration_sec < result.original_duration_sec


def test_splice_audio_shortens_long_gaps() -> None:
    """Long pauses (>250 ms) are shortened to target pause length (~200 ms)."""
    sr = 16000
    # Structure:
    # 0.5s silence + 1.0s speech + 1.5s silence + 1.0s speech + 0.5s silence = 4.5s
    s1_start = int(0.5 * sr)
    s1_end = int(1.5 * sr)
    s2_start = int(3.0 * sr)
    s2_end = int(4.0 * sr)
    total_len = int(4.5 * sr)

    audio = np.zeros(total_len, dtype=np.float32)
    burst1 = _create_synthetic_burst(1.0, sr=sr, f0=150.0)
    burst2 = _create_synthetic_burst(1.0, sr=sr, f0=200.0)
    audio[s1_start:s1_end] = burst1
    audio[s2_start:s2_end] = burst2

    segments = [
        SpeechSegment(s1_start, s1_end, 0.5, 1.5),
        SpeechSegment(s2_start, s2_end, 3.0, 4.0),
    ]
    vad_res = VADResult(
        segments=segments,
        gaps=[],
        sample_rate=sr,
        duration_sec=4.5,
        total_samples=total_len,
    )

    result = splice_audio(audio, vad_res)

    assert result.output_duration_sec < result.original_duration_sec
    assert result.pauses_capped >= 1
    # Check that join points are clean (no clicks)
    clicks = detect_join_clicks(result.audio, result.join_indices, max_step_jump=0.20)
    assert len(clicks) == 0
