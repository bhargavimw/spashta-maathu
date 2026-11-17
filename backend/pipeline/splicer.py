"""Stage 5A: Audio Splicer, Crossfader, and Click Detector.

Performs:
- Cut-point snapping to zero crossings / low-energy samples within +/-15 ms (plan.md §3).
- Equal-power crossfading (10-20 ms) at join points (plan.md §3).
- Gap-capping pause removal policy (plan.md §3 & §7 M2).
- Automated click / transient discontinuity detection (plan.md §6 & §7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.config import config
from backend.pipeline.vad import AudioGap, SpeechSegment, VADResult

logger = logging.getLogger("kannada_cleaner.splicer")


def snap_to_zero_crossing(
    audio: np.ndarray, index: int, window_samples: int = 240
) -> int:
    """Find the nearest zero-crossing or lowest-energy sample within a search window.

    Snapping cut points to zero-crossings or low-energy regions avoids clicks, pops,
    and mid-phoneme cut artifacts (plan.md §3).

    Args:
        audio: 1D float32 audio signal.
        index: Target sample index to snap.
        window_samples: Search window radius in samples (+/- 15 ms at 16 kHz is 240 samples).

    Returns:
        Snapped sample index in audio.
    """
    if audio.ndim != 1 or len(audio) == 0:
        return max(0, min(index, len(audio)))

    total_len = len(audio)
    target_idx = max(0, min(index, total_len - 1))

    start = max(0, target_idx - window_samples)
    end = min(total_len, target_idx + window_samples + 1)

    window = audio[start:end]
    if len(window) <= 1:
        return target_idx

    # 1. Search for true zero crossings (sign changes between adjacent samples)
    sign_changes = np.where(window[:-1] * window[1:] <= 0)[0]

    if len(sign_changes) > 0:
        # Map back to window indices and select the one closest to target_idx
        abs_distances = np.abs((start + sign_changes) - target_idx)
        best_candidate = int(sign_changes[np.argmin(abs_distances)])
        # Between sign_changes and sign_changes+1, pick the smaller magnitude
        idx1 = start + best_candidate
        idx2 = min(total_len - 1, idx1 + 1)
        return idx1 if abs(audio[idx1]) <= abs(audio[idx2]) else idx2

    # 2. If no exact sign change, pick the minimum absolute amplitude sample closest to target_idx
    min_val = np.min(np.abs(window))
    candidates = np.where(np.abs(window) == min_val)[0]
    best_candidate = candidates[np.argmin(np.abs((start + candidates) - target_idx))]
    return start + int(best_candidate)


def crossfade_equal_power(
    segment_a: np.ndarray, segment_b: np.ndarray, crossfade_samples: int = 240
) -> np.ndarray:
    """Crossfade two audio segments using an equal-power (cosine/sine) curve.

    Equal-power crossfading maintains constant total perceived loudness:
    fade_out(t) = cos(t * pi / 2), fade_in(t) = sin(t * pi / 2)
    where cos^2(t) + sin^2(t) = 1.

    Args:
        segment_a: Leading 1D float32 audio segment.
        segment_b: Trailing 1D float32 audio segment.
        crossfade_samples: Length of crossfade in samples (default 240 = 15 ms at 16 kHz).

    Returns:
        Joined 1D float32 audio array.
    """
    if segment_a.size == 0:
        return segment_b.astype(np.float32)
    if segment_b.size == 0:
        return segment_a.astype(np.float32)

    # Adjust crossfade length if segments are shorter than requested crossfade
    L = min(crossfade_samples, len(segment_a), len(segment_b))
    if L <= 0:
        return np.concatenate([segment_a, segment_b]).astype(np.float32)

    t = np.linspace(0.0, 1.0, L, endpoint=True, dtype=np.float32)
    fade_out = np.cos(t * (np.pi / 2.0)).astype(np.float32)
    fade_in = np.sin(t * (np.pi / 2.0)).astype(np.float32)

    a_head = segment_a[:-L]
    a_overlap = segment_a[-L:]
    b_overlap = segment_b[:L]
    b_tail = segment_b[L:]

    blended = (a_overlap * fade_out) + (b_overlap * fade_in)

    return np.concatenate([a_head, blended, b_tail]).astype(np.float32)


def detect_join_clicks(
    audio: np.ndarray,
    join_indices: List[int],
    max_step_jump: float = 0.20,
    neighborhood: int = 3,
) -> List[int]:
    """Automated click check: detects sharp energy spikes or sample discontinuities at join points.

    Args:
        audio: 1D float32 audio signal.
        join_indices: Sample indices corresponding to segment join points.
        max_step_jump: Maximum permissible sample-to-sample delta (default: 0.20).
        neighborhood: Number of samples around the join point to inspect.

    Returns:
        List of join indices where a click or discontinuity was detected.
    """
    if audio.ndim != 1 or len(audio) < 2:
        return []

    clicks: List[int] = []
    diff = np.abs(np.diff(audio))

    for idx in join_indices:
        start = max(0, idx - neighborhood)
        end = min(len(diff), idx + neighborhood)
        if start < end:
            max_delta = float(np.max(diff[start:end]))
            if max_delta > max_step_jump:
                clicks.append(idx)

    return clicks


@dataclass(frozen=True)
class SpliceResult:
    """Result of audio splicing operation."""

    audio: np.ndarray
    join_indices: List[int]
    original_duration_sec: float
    output_duration_sec: float
    time_saved_sec: float
    pauses_capped: int


def apply_pause_policy_to_gaps(
    vad_result: VADResult,
    min_gap_to_shorten_sec: float = 0.25,
    shortened_gap_sec: float = 0.20,
    edge_silence_trim_sec: float = 0.10,
) -> List[Tuple[float, float, str, float]]:
    """Determine target timeline of speech and adjusted gaps under pause policy.

    Args:
        vad_result: Output from VAD containing segments and gaps.
        min_gap_to_shorten_sec: Gaps longer than this threshold are shortened (default: 0.25s).
        shortened_gap_sec: Target duration for shortened gaps (default: 0.20s).
        edge_silence_trim_sec: Maximum duration for leading and trailing silence (default: 0.10s).

    Returns:
        List of tuples: (original_start_sec, original_end_sec, type, target_duration_sec)
        where type is "speech", "leading_silence", "pause", or "trailing_silence".
    """
    timeline: List[Tuple[float, float, str, float]] = []

    if not vad_result.segments:
        # Audio has no speech; trim entire clip to edge_silence_trim_sec
        target_dur = min(vad_result.duration_sec, edge_silence_trim_sec)
        return [(0.0, vad_result.duration_sec, "leading_silence", target_dur)]

    total_segments = len(vad_result.segments)

    # 1. Leading silence
    first_seg = vad_result.segments[0]
    if first_seg.start_sec > 0:
        lead_dur = first_seg.start_sec
        target_lead = min(lead_dur, edge_silence_trim_sec)
        timeline.append((0.0, first_seg.start_sec, "leading_silence", target_lead))

    # 2. Interleave speech segments and inter-segment gaps
    for i in range(total_segments):
        seg = vad_result.segments[i]
        timeline.append((seg.start_sec, seg.end_sec, "speech", seg.duration_sec))

        if i < total_segments - 1:
            next_seg = vad_result.segments[i + 1]
            gap_dur = next_seg.start_sec - seg.end_sec
            if gap_dur > min_gap_to_shorten_sec:
                target_gap = shortened_gap_sec
            else:
                target_gap = gap_dur
            timeline.append((seg.end_sec, next_seg.start_sec, "pause", target_gap))

    # 3. Trailing silence
    last_seg = vad_result.segments[-1]
    if last_seg.end_sec < vad_result.duration_sec:
        trail_dur = vad_result.duration_sec - last_seg.end_sec
        target_trail = min(trail_dur, edge_silence_trim_sec)
        timeline.append((last_seg.end_sec, vad_result.duration_sec, "trailing_silence", target_trail))

    return timeline


def splice_audio(
    audio: np.ndarray,
    vad_result: VADResult,
    crossfade_ms: float | None = None,
    snap_window_ms: float | None = None,
) -> SpliceResult:
    """Slice audio at zero-crossings, cap gaps per pause policy, and crossfade joins.

    Args:
        audio: Ingested 16 kHz mono float32 audio array.
        vad_result: VAD segmentation and gap analysis.
        crossfade_ms: Crossfade length in milliseconds (default from config: 15.0 ms).
        snap_window_ms: Snapping window in milliseconds (default from config: 15.0 ms).

    Returns:
        SpliceResult containing spliced audio array, join indices, and duration stats.
    """
    sr = vad_result.sample_rate
    cf_ms = crossfade_ms if crossfade_ms is not None else config.pause.crossfade_ms
    cf_samples = int((cf_ms / 1000.0) * sr)

    snap_ms = snap_window_ms if snap_window_ms is not None else config.audio.snap_window_ms
    snap_samples = int((snap_ms / 1000.0) * sr)

    min_gap_sec = config.pause.min_gap_to_shorten_ms / 1000.0
    target_gap_sec = config.pause.shortened_gap_target_ms / 1000.0
    edge_trim_sec = config.pause.edge_silence_trim_ms / 1000.0

    timeline = apply_pause_policy_to_gaps(
        vad_result,
        min_gap_to_shorten_sec=min_gap_sec,
        shortened_gap_sec=target_gap_sec,
        edge_silence_trim_sec=edge_trim_sec,
    )

    chunks: List[np.ndarray] = []
    join_indices: List[int] = []
    current_sample_pos = 0
    pauses_capped = 0

    ramp_len = min(cf_samples, int(0.010 * sr))  # 10 ms smooth boundary taper

    for item in timeline:
        orig_start_sec, orig_end_sec, item_type, target_dur = item

        if item_type == "speech":
            # Snap start and end sample to zero-crossings
            orig_start_sample = int(orig_start_sec * sr)
            orig_end_sample = int(orig_end_sec * sr)

            snapped_start = snap_to_zero_crossing(audio, orig_start_sample, snap_samples)
            snapped_end = snap_to_zero_crossing(audio, orig_end_sample, snap_samples)

            if snapped_end <= snapped_start:
                snapped_end = min(len(audio), snapped_start + int(0.05 * sr))

            chunk = audio[snapped_start:snapped_end].copy()

            # Smoothly taper first and last few ms with half-cosine to avoid clicks at transitions
            if len(chunk) >= 2 * ramp_len and ramp_len > 0:
                t = np.linspace(0, 1, ramp_len, dtype=np.float32)
                chunk[:ramp_len] *= np.sin(t * (np.pi / 2.0))
                chunk[-ramp_len:] *= np.cos(t * (np.pi / 2.0))

            if chunks:
                join_indices.append(current_sample_pos)

            chunks.append(chunk)
            current_sample_pos += len(chunk)

        else:
            # Silence item (leading, trailing, or pause)
            orig_gap_dur = orig_end_sec - orig_start_sec
            if orig_gap_dur > target_dur:
                pauses_capped += 1

            n_silence_samples = int(target_dur * sr)
            if n_silence_samples > 0:
                silence_chunk = np.zeros(n_silence_samples, dtype=np.float32)
                if chunks:
                    join_indices.append(current_sample_pos)
                chunks.append(silence_chunk)
                current_sample_pos += len(silence_chunk)

    if chunks:
        spliced = np.concatenate(chunks).astype(np.float32)
    else:
        spliced = np.zeros(0, dtype=np.float32)

    orig_dur = vad_result.duration_sec
    out_dur = float(len(spliced)) / float(sr)
    time_saved = max(0.0, orig_dur - out_dur)

    return SpliceResult(
        audio=spliced,
        join_indices=join_indices,
        original_duration_sec=orig_dur,
        output_duration_sec=out_dur,
        time_saved_sec=time_saved,
        pauses_capped=pauses_capped,
    )
