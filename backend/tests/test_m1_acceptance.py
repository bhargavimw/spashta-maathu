"""Milestone M1 Acceptance Test (plan.md §7).

Acceptance Criterion:
"On a 10 s clip: output is 16 kHz mono; VAD returns segments and gaps; unit tests pass"
"""

import numpy as np
import pytest

from backend.pipeline.ingest import ingest_audio
from backend.pipeline.vad import detect_speech_regions


def _create_synthetic_speech_burst(duration_sec: float, sr: int = 16000, f0: float = 160.0) -> np.ndarray:
    """Create a speech-like multi-harmonic signal with smoothed edges."""
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)

    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    ).astype(np.float32)

    envelope = np.ones(n, dtype=np.float32)
    ramp = min(int(0.04 * sr), n // 4)
    if ramp > 0:
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)

    return (signal * envelope).astype(np.float32)


def test_m1_acceptance_10s_clip() -> None:
    """Verify Milestone M1 acceptance criteria on a 10-second audio clip."""
    sr_input = 24000  # Non-16k input to also verify ingest conversion
    target_sr = 16000

    # Build 10.0 s clip at 24 kHz:
    # 0.0s - 1.0s: silence (1.0s)
    # 1.0s - 4.0s: speech (3.0s)
    # 4.0s - 6.0s: silence (2.0s)
    # 6.0s - 8.5s: speech (2.5s)
    # 8.5s - 10.0s: silence (1.5s)
    g1 = np.zeros(int(1.0 * sr_input), dtype=np.float32)
    s1 = _create_synthetic_speech_burst(duration_sec=3.0, sr=sr_input, f0=180.0)
    g2 = np.zeros(int(2.0 * sr_input), dtype=np.float32)
    s2 = _create_synthetic_speech_burst(duration_sec=2.5, sr=sr_input, f0=220.0)
    g3 = np.zeros(int(1.5 * sr_input), dtype=np.float32)

    raw_audio_10s = np.concatenate([g1, s1, g2, s2, g3])
    assert len(raw_audio_10s) == int(10.0 * sr_input)

    # 1. Test Ingest: output must be 16 kHz mono float32, peak normalized
    ingested_audio = ingest_audio(
        audio_array=raw_audio_10s,
        sr=sr_input,
        target_sr=target_sr,
        target_db=-1.0,
    )

    # Ingest Assertions
    assert ingested_audio.ndim == 1, f"Expected 1D mono, got ndim={ingested_audio.ndim}"
    assert ingested_audio.dtype == np.float32, f"Expected float32, got {ingested_audio.dtype}"
    assert len(ingested_audio) == int(10.0 * target_sr), (
        f"Expected {int(10.0 * target_sr)} samples for 10 s at 16 kHz, got {len(ingested_audio)}"
    )
    expected_peak = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(ingested_audio)) == pytest.approx(expected_peak, rel=1e-3)

    # 2. Test VAD: must return segments and gaps
    vad_result = detect_speech_regions(ingested_audio, sample_rate=target_sr)

    # VAD Assertions
    assert vad_result.sample_rate == 16000
    assert vad_result.duration_sec == pytest.approx(10.0)
    assert len(vad_result.segments) > 0, "VAD returned no speech segments on speech bursts"
    assert len(vad_result.gaps) > 0, "VAD returned no gaps on audio with pauses"

    # Validate segment and gap bounds
    for seg in vad_result.segments:
        assert 0.0 <= seg.start_sec < seg.end_sec <= 10.0
        assert 0 <= seg.start_sample < seg.end_sample <= len(ingested_audio)

    for gap in vad_result.gaps:
        assert 0.0 <= gap.start_sec < gap.end_sec <= 10.0
        assert 0 <= gap.start_sample < gap.end_sample <= len(ingested_audio)
        assert gap.duration_sec == pytest.approx(gap.end_sec - gap.start_sec)
