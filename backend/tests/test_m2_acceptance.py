"""Milestone M2 Acceptance Test (plan.md §7).

Acceptance Criterion:
"Gap-capped output is shorter than input, no clicks at joins (automated check), duration math matches"
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.cli import run_pause_remover
from backend.config import config
from backend.pipeline.ingest import ingest_audio
from backend.pipeline.splicer import detect_join_clicks, splice_audio
from backend.pipeline.vad import detect_speech_regions


def _generate_harmonic_speech(duration_sec: float, sr: int = 16000, f0: float = 180.0) -> np.ndarray:
    """Generate multi-harmonic speech-like signal with pitch and syllabic modulation."""
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)

    # Pitch modulation (vibrato at ~5 Hz)
    pitch = f0 + 10.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / sr

    signal = (
        0.5 * np.sin(phase)
        + 0.3 * np.sin(2 * phase)
        + 0.15 * np.sin(3 * phase)
    ).astype(np.float32)

    # Syllabic envelope (~4 Hz speech syllable cadence)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
    ramp = min(int(0.04 * sr), n // 4)
    if ramp > 0:
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)

    return (signal * envelope).astype(np.float32)


def test_m2_acceptance_gap_capping_and_no_clicks() -> None:
    """Verify Milestone M2 acceptance criteria on audio with unnatural pauses."""
    sr = 16000

    # Build audio with deliberate long pauses:
    # 0.0s - 0.8s: leading silence (0.8s -> should trim to 0.10s)
    # 0.8s - 2.8s: speech burst 1 (2.0s)
    # 2.8s - 4.8s: long unnatural pause (2.0s -> should shorten to 0.20s)
    # 4.8s - 7.3s: speech burst 2 (2.5s)
    # 7.3s - 8.8s: second long pause (1.5s -> should shorten to 0.20s)
    # 8.8s - 10.8s: speech burst 3 (2.0s)
    # 10.8s - 12.0s: trailing silence (1.2s -> should trim to 0.10s)
    # Total input duration = 12.0 seconds

    p_lead = np.zeros(int(0.8 * sr), dtype=np.float32)
    s1 = _generate_harmonic_speech(2.0, sr=sr, f0=160.0)
    p1 = np.zeros(int(2.0 * sr), dtype=np.float32)
    s2 = _generate_harmonic_speech(2.5, sr=sr, f0=210.0)
    p2 = np.zeros(int(1.5 * sr), dtype=np.float32)
    s3 = _generate_harmonic_speech(2.0, sr=sr, f0=180.0)
    p_trail = np.zeros(int(1.2 * sr), dtype=np.float32)

    raw_audio = np.concatenate([p_lead, s1, p1, s2, p2, s3, p_trail])
    input_duration_sec = len(raw_audio) / float(sr)
    assert input_duration_sec == pytest.approx(12.0)

    # 1. Pipeline Ingest & VAD
    ingested = ingest_audio(audio_array=raw_audio, sr=sr, target_sr=sr)
    vad_result = detect_speech_regions(ingested, sample_rate=sr)

    assert len(vad_result.segments) >= 3

    # 2. Splicer Execution
    splice_res = splice_audio(ingested, vad_result)

    # --- Criterion 1: Gap-capped output is shorter than input ---
    assert splice_res.output_duration_sec < splice_res.original_duration_sec, (
        f"Output duration ({splice_res.output_duration_sec:.2f}s) must be shorter than input ({splice_res.original_duration_sec:.2f}s)"
    )
    assert splice_res.time_saved_sec > 2.0, "Expected significant duration reduction on long pauses"

    # --- Criterion 2: No clicks at joins (automated check) ---
    assert len(splice_res.join_indices) > 0, "Expected join points in spliced audio"
    detected_clicks = detect_join_clicks(
        splice_res.audio, splice_res.join_indices, max_step_jump=0.20
    )
    assert len(detected_clicks) == 0, f"Detected clicks at sample indices: {detected_clicks}"

    # --- Criterion 3: Duration math matches ---
    from backend.pipeline.splicer import apply_pause_policy_to_gaps

    timeline = apply_pause_policy_to_gaps(
        vad_result,
        min_gap_to_shorten_sec=config.pause.min_gap_to_shorten_ms / 1000.0,
        shortened_gap_sec=config.pause.shortened_gap_target_ms / 1000.0,
        edge_silence_trim_sec=config.pause.edge_silence_trim_ms / 1000.0,
    )
    expected_duration = sum(item[3] for item in timeline)

    # Allow tiny tolerance for zero-crossing snapping (+/- 15 ms per cut point)
    tolerance_sec = 0.10
    assert splice_res.output_duration_sec == pytest.approx(expected_duration, abs=tolerance_sec), (
        f"Output duration ({splice_res.output_duration_sec:.2f}s) differs from expected ({expected_duration:.2f}s)"
    )


def test_m2_cli_execution_end_to_end() -> None:
    """Verify v1 pause remover CLI runs end-to-end and produces a valid audio file."""
    sr = 16000
    duration = 3.0
    # 0.5s silence + 1.0s speech + 1.5s silence
    audio = np.concatenate([
        np.zeros(int(0.5 * sr), dtype=np.float32),
        _generate_harmonic_speech(1.0, sr=sr, f0=200.0),
        np.zeros(int(1.5 * sr), dtype=np.float32),
    ])

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / "test_input.wav"
        output_file = Path(tmp_dir) / "test_output_clean.wav"

        sf.write(str(input_file), audio, sr)

        exit_code = run_pause_remover(input_file, output_file)
        assert exit_code == 0
        assert output_file.is_file()

        cleaned_data, cleaned_sr = sf.read(str(output_file))
        assert cleaned_sr == 16000
        assert cleaned_data.ndim == 1
        assert len(cleaned_data) / cleaned_sr < duration
