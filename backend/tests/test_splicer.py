"""Unit tests for Stage 5A: Audio Splicer, Crossfader, and Click Detector."""

import numpy as np
import pytest

from backend.pipeline.classifier import (
    ClassifierInput,
    ClassifierOutput,
    LLMClassification,
    RemovedSegment,
    RemovalReason,
)
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


# ---------------------------------------------------------------------------
# Shared helpers for classifier-guided tests
# ---------------------------------------------------------------------------

SR = 16000  # 16 kHz throughout


def _make_sine_burst(start_sample: int, duration_samples: int, f0: float = 200.0) -> np.ndarray:
    """Return a silent-padded 1-D sine burst placed at start_sample."""
    t = np.arange(duration_samples, dtype=np.float32) / SR
    return (0.4 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _build_scenario(
    seg_specs: list[tuple[float, float]],  # [(start_sec, end_sec), ...]
    total_sec: float,
    f0_list: list[float] | None = None,
) -> tuple[np.ndarray, VADResult, ClassifierInput]:
    """
    Build a consistent (audio, vad_result, classifier_input) triple for a
    sequence of non-overlapping speech segments embedded in silence.

    seg_specs – list of (start_sec, end_sec) for each segment, in order.
    total_sec – total clip duration in seconds.
    f0_list   – per-segment sine frequency; defaults to 200 Hz for all.
    """
    n_total = int(total_sec * SR)
    audio = np.zeros(n_total, dtype=np.float32)

    vad_segments = []
    cls_segments_raw = []

    for idx, (start_sec, end_sec) in enumerate(seg_specs):
        s = int(start_sec * SR)
        e = int(end_sec * SR)
        f0 = (f0_list[idx] if f0_list else 200.0)
        burst = _make_sine_burst(s, e - s, f0=f0)
        audio[s:e] = burst

        vad_segments.append(SpeechSegment(
            start_sample=s, end_sample=e,
            start_sec=start_sec, end_sec=end_sec,
        ))

        # gap_after is the time to the *next* segment's start (0.0 for the last)
        if idx < len(seg_specs) - 1:
            gap_after = seg_specs[idx + 1][0] - end_sec
        else:
            gap_after = 0.0

        cls_segments_raw.append({
            "index": idx,
            "text": f"word{idx}",
            "start": start_sec,
            "end": end_sec,
            "gap_after": gap_after,
        })

    vad_result = VADResult(
        segments=vad_segments,
        gaps=[],
        sample_rate=SR,
        duration_sec=total_sec,
        total_samples=n_total,
    )
    classifier_input = ClassifierInput.from_list(cls_segments_raw)
    return audio, vad_result, classifier_input


def _make_output(
    classifier_input: ClassifierInput,
    kept: list[int],
    pauses: list[float],
    removed_indices: list[int] | None = None,
) -> ClassifierOutput:
    """
    Build a ClassifierOutput that keeps *kept* indices with the given inter-
    segment *pauses* (len == len(kept) - 1) and removes the rest.
    """
    all_indices = {s.index for s in classifier_input.segments}
    removed_set = set(removed_indices) if removed_indices is not None else all_indices - set(kept)
    removed = [RemovedSegment(index=i, reason=RemovalReason.FILLER) for i in sorted(removed_set)]

    llm = LLMClassification(
        kept_segments=kept,
        pause_after_kept=pauses,
        removed=removed,
    )
    return ClassifierOutput.from_llm(llm, classifier_input)


# ---------------------------------------------------------------------------
# Test 1 – Only retained segments appear in the output
# ---------------------------------------------------------------------------

def test_classifier_guided_retains_only_kept_segments() -> None:
    """Output audio duration reflects only the kept segments, not the removed one."""
    # Three segments: keep 0 and 2, drop 1 (the middle).
    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)],
        total_sec=5.0,
    )
    # Pause of 0.0 s between kept[0] and kept[1] → crossfade path.
    co = _make_output(ci, kept=[0, 2], pauses=[0.0])

    result = splice_audio(
        audio, vad,
        crossfade_ms=0.0,   # disable crossfade to get exact lengths
        snap_window_ms=0.0,
        classifier_input=ci,
        classifier_output=co,
    )

    assert isinstance(result, SpliceResult)
    # Each kept segment is 1.0 s; with crossfade=0 they are simply concatenated.
    expected_dur = pytest.approx(2.0, abs=0.05)
    assert result.output_duration_sec == expected_dur
    # Original is 5 s; we dropped one 1 s segment and all gaps → shorter.
    assert result.output_duration_sec < result.original_duration_sec
    assert np.all(np.isfinite(result.audio))


# ---------------------------------------------------------------------------
# Test 2 – Exact pause durations are respected
# ---------------------------------------------------------------------------

def test_classifier_guided_inserts_exact_pause() -> None:
    """The silence inserted between two kept segments matches pause_after_kept."""
    requested_pause_sec = 0.30

    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 1.0), (2.0, 3.0)],
        total_sec=4.0,
    )
    co = _make_output(ci, kept=[0, 1], pauses=[requested_pause_sec])

    result = splice_audio(
        audio, vad,
        crossfade_ms=0.0,
        snap_window_ms=0.0,
        classifier_input=ci,
        classifier_output=co,
    )

    # With crossfade=0 and snap=0 the total duration is seg0 + pause + seg1.
    # seg0 = 1.0 s, seg1 = 1.0 s, pause = 0.30 s → 2.30 s.
    assert result.output_duration_sec == pytest.approx(2.30, abs=0.02)
    assert result.pauses_capped >= 1   # original gap was 1.0 s, capped to 0.30 s

    # Verify silent region: find the silence block between the two join indices.
    join = result.join_indices
    assert len(join) >= 2
    silence_start = join[0]
    silence_end = join[1]
    silence_region = result.audio[silence_start:silence_end]
    assert np.all(silence_region == 0.0), "Inserted pause must be pure silence"
    actual_pause_sec = len(silence_region) / SR
    assert actual_pause_sec == pytest.approx(requested_pause_sec, abs=0.002)


# ---------------------------------------------------------------------------
# Test 3 – Retained segments appear in the original order
# ---------------------------------------------------------------------------

def test_classifier_guided_preserves_segment_order() -> None:
    """When three segments are kept, their audio arrives in index order."""
    # Use distinct frequencies so we can fingerprint each segment.
    freqs = [100.0, 200.0, 300.0]
    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 0.5), (1.0, 1.5), (2.0, 2.5)],
        total_sec=3.5,
        f0_list=freqs,
    )
    co = _make_output(ci, kept=[0, 1, 2], pauses=[0.0, 0.0])

    result = splice_audio(
        audio, vad,
        crossfade_ms=0.0,
        snap_window_ms=0.0,
        classifier_input=ci,
        classifier_output=co,
    )

    # With no crossfade the output is a direct concatenation of the three bursts.
    # Check dominant frequency of each 0.5 s window.
    out = result.audio
    seg_len = int(0.5 * SR)
    for i, f0 in enumerate(freqs):
        chunk = out[i * seg_len : (i + 1) * seg_len]
        fft = np.abs(np.fft.rfft(chunk))
        freqs_axis = np.fft.rfftfreq(len(chunk), d=1.0 / SR)
        dominant = float(freqs_axis[np.argmax(fft)])
        assert abs(dominant - f0) < 20.0, (
            f"Segment {i}: expected dominant freq ~{f0} Hz, got {dominant:.1f} Hz"
        )


# ---------------------------------------------------------------------------
# Test 4 – Mismatched classifier / VAD timestamps raise ValueError
# ---------------------------------------------------------------------------

def test_classifier_guided_rejects_mismatched_timestamps() -> None:
    """Timestamps that don't match any VAD segment raise ValueError."""
    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 1.0), (2.0, 3.0)],
        total_sec=4.0,
    )
    # Build a classifier_input whose timestamps are far from the VAD segments.
    bad_ci = ClassifierInput.from_list([
        {"index": 0, "text": "word0", "start": 0.5, "end": 1.5, "gap_after": 1.0},
        {"index": 1, "text": "word1", "start": 2.5, "end": 3.5, "gap_after": 0.0},
    ])
    bad_co = _make_output(bad_ci, kept=[0, 1], pauses=[0.3])

    with pytest.raises(ValueError, match="matched"):
        splice_audio(
            audio, vad,
            crossfade_ms=0.0,
            snap_window_ms=0.0,
            classifier_input=bad_ci,
            classifier_output=bad_co,
        )


# ---------------------------------------------------------------------------
# Test 5 – Providing only one of the classifier pair raises ValueError
# ---------------------------------------------------------------------------

def test_classifier_guided_rejects_half_pair_input_only() -> None:
    """Passing classifier_input without classifier_output raises ValueError."""
    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 1.0), (2.0, 3.0)],
        total_sec=4.0,
    )
    with pytest.raises(ValueError, match="both"):
        splice_audio(audio, vad, classifier_input=ci, classifier_output=None)


def test_classifier_guided_rejects_half_pair_output_only() -> None:
    """Passing classifier_output without classifier_input raises ValueError."""
    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 1.0), (2.0, 3.0)],
        total_sec=4.0,
    )
    co = _make_output(ci, kept=[0, 1], pauses=[0.3])
    with pytest.raises(ValueError, match="both"):
        splice_audio(audio, vad, classifier_input=None, classifier_output=co)


# ---------------------------------------------------------------------------
# Test 6 – Zero-length pause triggers crossfade path without silence gap
# ---------------------------------------------------------------------------

def test_classifier_guided_zero_pause_uses_crossfade() -> None:
    """A pause of 0.0 s crossfades segments without inserting silent samples."""
    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 1.0), (2.0, 3.0)],
        total_sec=4.0,
    )
    co = _make_output(ci, kept=[0, 1], pauses=[0.0])

    # Use a small but nonzero crossfade so the crossfade branch is exercised.
    result = splice_audio(
        audio, vad,
        crossfade_ms=10.0,
        snap_window_ms=0.0,
        classifier_input=ci,
        classifier_output=co,
    )

    # With a 10 ms crossfade and two 1-second segments the output must be
    # shorter than 2 seconds (segments overlap in crossfade zone) and must
    # contain no all-zero interior block longer than the crossfade window.
    cf_samples = int(0.010 * SR)
    assert result.output_duration_sec < 2.0 + 0.001
    assert result.output_duration_sec > 1.9  # sanity: not pathologically short

    # No contiguous zero block longer than cf_samples (no inserted silence).
    out = result.audio
    zero_mask = (out == 0.0).astype(np.int32)
    # Largest run of zeros
    max_run = 0
    run = 0
    for v in zero_mask:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    assert max_run <= cf_samples + 10, (
        f"Found a zero run of {max_run} samples; suggests silence was inserted"
    )
    assert np.all(np.isfinite(result.audio))


# ---------------------------------------------------------------------------
# Test 7 – Output duration and join indices are consistent
# ---------------------------------------------------------------------------

def test_classifier_guided_output_duration_and_join_indices() -> None:
    """output_duration_sec matches audio length; join_indices are ordered and in-bounds."""
    pause_sec = 0.20
    audio, vad, ci = _build_scenario(
        seg_specs=[(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)],
        total_sec=5.0,
    )
    # Keep all three segments with explicit pauses between them.
    co = _make_output(ci, kept=[0, 1, 2], pauses=[pause_sec, pause_sec])

    result = splice_audio(
        audio, vad,
        crossfade_ms=0.0,
        snap_window_ms=0.0,
        classifier_input=ci,
        classifier_output=co,
    )

    # 1. output_duration_sec must equal len(audio) / SR.
    expected_dur = len(result.audio) / SR
    assert result.output_duration_sec == pytest.approx(expected_dur, rel=1e-5)

    # 2. time_saved_sec must be non-negative and equal to the difference.
    assert result.time_saved_sec >= 0.0
    assert result.time_saved_sec == pytest.approx(
        result.original_duration_sec - result.output_duration_sec, abs=1e-5
    )

    # 3. All join indices must be strictly within [0, len(audio)).
    n = len(result.audio)
    for ji in result.join_indices:
        assert 0 <= ji < n, f"join index {ji} out of bounds [0, {n})"

    # 4. Join indices must be strictly ascending.
    for a, b in zip(result.join_indices, result.join_indices[1:]):
        assert a < b, f"join indices not ascending: {a} then {b}"

    # 5. With pause_sec > 0 there must be at least one join index per inter-
    #    segment gap (two gaps ⟹ at least 2 join indices in classifier path,
    #    actually 4 since each pause boundary records two indices).
    assert len(result.join_indices) >= 2
