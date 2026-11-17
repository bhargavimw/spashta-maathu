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

# Classifier types are imported lazily at the type-checking level to avoid a
# hard dependency when the classifier module is not installed.  At runtime they
# are only needed when the caller passes classifier_input / classifier_output.
try:
    from backend.pipeline.classifier import ClassifierInput, ClassifierOutput
except ImportError:  # pragma: no cover
    ClassifierInput = None  # type: ignore[assignment,misc]
    ClassifierOutput = None  # type: ignore[assignment,misc]


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


def _match_classifier_segments_to_vad(
    classifier_input: "ClassifierInput",
    vad_result: VADResult,
    tolerance_sec: float = 0.05,
) -> Dict[int, SpeechSegment]:
    """Match classifier segments to VAD segments using timestamps."""

    matched: Dict[int, SpeechSegment] = {}
    used_vad_indices = set()

    for source in classifier_input.segments:
        candidates = [
            (i, vad_seg)
            for i, vad_seg in enumerate(vad_result.segments)
            if (
                abs(vad_seg.start_sec - source.start) <= tolerance_sec
                and abs(vad_seg.end_sec - source.end) <= tolerance_sec
            )
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Classifier segment {source.index} matched "
                f"{len(candidates)} VAD segments. Expected exactly one."
            )

        vad_index, vad_seg = candidates[0]

        if vad_index in used_vad_indices:
            raise ValueError(
                f"Multiple classifier segments map to VAD segment {vad_index}."
            )

        used_vad_indices.add(vad_index)
        matched[source.index] = vad_seg

    return matched


def _taper_audio_edges(
    chunk: np.ndarray,
    ramp_samples: int,
    fade_start: bool = True,
    fade_end: bool = True,
) -> np.ndarray:
    """Apply short half-cosine fades to prevent clicks."""

    chunk = chunk.copy()
    ramp = min(ramp_samples, len(chunk) // 2)

    if ramp <= 0:
        return chunk

    t = np.linspace(0.0, 1.0, ramp, dtype=np.float32)

    if fade_start:
        chunk[:ramp] *= np.sin(t * (np.pi / 2.0))

    if fade_end:
        chunk[-ramp:] *= np.cos(t * (np.pi / 2.0))

    return chunk


def _splice_classifier_guided(
    audio: np.ndarray,
    vad_result: VADResult,
    classifier_input: "ClassifierInput",
    classifier_output: "ClassifierOutput",
    crossfade_samples: int,
    snap_samples: int,
) -> SpliceResult:
    """Build output audio using retained segments and requested pauses."""

    classifier_output.validate_against_input(classifier_input)

    vad_by_index = _match_classifier_segments_to_vad(
        classifier_input,
        vad_result,
    )

    source_by_index = classifier_input.by_index()
    kept_indices = classifier_output.kept_segments

    if not kept_indices:
        raise ValueError("Classifier returned no retained speech segments.")

    sr = vad_result.sample_rate
    ramp_samples = min(crossfade_samples, int(0.010 * sr))

    speech_chunks = []

    for index in kept_indices:
        vad_seg = vad_by_index[index]

        start = snap_to_zero_crossing(
            audio, vad_seg.start_sample, snap_samples
        )
        end = snap_to_zero_crossing(
            audio, vad_seg.end_sample, snap_samples
        )

        if end <= start:
            raise ValueError(
                f"Invalid audio boundaries for retained segment {index}."
            )

        chunk = audio[start:end].astype(np.float32, copy=True)

        if len(chunk) == 0:
            raise ValueError(
                f"Retained segment {index} produced an empty audio chunk."
            )

        speech_chunks.append(chunk)

    chunks = [speech_chunks[0]]
    join_indices = []
    current_sample_pos = len(speech_chunks[0])
    pauses_capped = 0

    for i in range(len(speech_chunks) - 1):
        pause_sec = float(classifier_output.pause_after_kept[i])
        pause_samples = max(0, int(round(pause_sec * sr)))

        current_index = kept_indices[i]
        next_index = kept_indices[i + 1]

        available_gap = (
            source_by_index[next_index].start
            - source_by_index[current_index].end
        )

        if pause_sec < available_gap - 0.001:
            pauses_capped += 1

        next_chunk = speech_chunks[i + 1]

        if pause_samples > 0:
            chunks[-1] = _taper_audio_edges(
                chunks[-1], ramp_samples, fade_end=True
            )

            next_chunk = _taper_audio_edges(
                next_chunk, ramp_samples, fade_start=True
            )

            join_indices.append(current_sample_pos)

            silence = np.zeros(pause_samples, dtype=np.float32)
            chunks.append(silence)
            current_sample_pos += pause_samples

            join_indices.append(current_sample_pos)

            chunks.append(next_chunk)
            current_sample_pos += len(next_chunk)

        else:
            overlap = min(
                crossfade_samples,
                len(chunks[-1]),
                len(next_chunk),
            )

            if overlap > 0:
                join_indices.append(current_sample_pos - overlap)

                chunks[-1] = crossfade_equal_power(
                    chunks[-1], next_chunk, overlap
                )

                current_sample_pos += len(next_chunk) - overlap

            else:
                join_indices.append(current_sample_pos)
                chunks.append(next_chunk)
                current_sample_pos += len(next_chunk)

    spliced = np.concatenate(chunks).astype(np.float32)

    original_duration = vad_result.duration_sec
    output_duration = len(spliced) / sr

    return SpliceResult(
        audio=spliced,
        join_indices=join_indices,
        original_duration_sec=original_duration,
        output_duration_sec=output_duration,
        time_saved_sec=max(0.0, original_duration - output_duration),
        pauses_capped=pauses_capped,
    )


def splice_audio(
    audio: np.ndarray,
    vad_result: VADResult,
    crossfade_ms: float | None = None,
    snap_window_ms: float | None = None,
    classifier_input: Optional["ClassifierInput"] = None,
    classifier_output: Optional["ClassifierOutput"] = None,
) -> SpliceResult:
    """Slice audio at zero-crossings, cap gaps per pause policy, and crossfade joins.

    Args:
        audio: Ingested 16 kHz mono float32 audio array.
        vad_result: VAD segmentation and gap analysis.
        crossfade_ms: Crossfade length in milliseconds (default from config: 15.0 ms).
        snap_window_ms: Snapping window in milliseconds (default from config: 15.0 ms).
        classifier_input: Optional Module 3 classifier input (required when
            classifier_output is provided).
        classifier_output: Optional Module 3 classifier output.  When both
            classifier_input and classifier_output are supplied the
            classifier-guided splice path is used; otherwise the original
            VAD-only path runs.

    Returns:
        SpliceResult containing spliced audio array, join indices, and duration stats.
    """
    if audio.ndim != 1:
        raise ValueError("Audio must be a one-dimensional mono array.")

    if len(audio) != vad_result.total_samples:
        raise ValueError(
            f"Audio length ({len(audio)}) does not match "
            f"VAD sample count ({vad_result.total_samples})."
        )

    sr = vad_result.sample_rate

    cf_ms = (
        crossfade_ms
        if crossfade_ms is not None
        else config.pause.crossfade_ms
    )
    cf_samples = max(0, int((cf_ms / 1000.0) * sr))

    snap_ms = (
        snap_window_ms
        if snap_window_ms is not None
        else config.audio.snap_window_ms
    )
    snap_samples = max(0, int((snap_ms / 1000.0) * sr))

    if (classifier_input is None) != (classifier_output is None):
        raise ValueError(
            "classifier_input and classifier_output must both be "
            "provided or both be None."
        )

    if classifier_input is not None:
        return _splice_classifier_guided(
            audio=audio,
            vad_result=vad_result,
            classifier_input=classifier_input,
            classifier_output=classifier_output,
            crossfade_samples=cf_samples,
            snap_samples=snap_samples,
        )

    # Original VAD-only processing path.
    timeline = apply_pause_policy_to_gaps(
        vad_result,
        min_gap_to_shorten_sec=config.pause.min_gap_to_shorten_ms / 1000.0,
        shortened_gap_sec=config.pause.shortened_gap_target_ms / 1000.0,
        edge_silence_trim_sec=config.pause.edge_silence_trim_ms / 1000.0,
    )

    chunks: List[np.ndarray] = []
    join_indices: List[int] = []
    current_sample_pos = 0
    pauses_capped = 0

    ramp_len = min(cf_samples, int(0.010 * sr))  # 10 ms smooth boundary taper

    for orig_start_sec, orig_end_sec, item_type, target_dur in timeline:

        if item_type == "speech":
            # Snap start and end sample to zero-crossings
            orig_start_sample = int(orig_start_sec * sr)
            orig_end_sample = int(orig_end_sec * sr)

            snapped_start = snap_to_zero_crossing(
                audio, orig_start_sample, snap_samples
            )
            snapped_end = snap_to_zero_crossing(
                audio, orig_end_sample, snap_samples
            )

            if snapped_end <= snapped_start:
                snapped_end = min(
                    len(audio),
                    snapped_start + int(0.05 * sr),
                )

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
                silence_chunk = np.zeros(
                    n_silence_samples, dtype=np.float32
                )

                if chunks:
                    join_indices.append(current_sample_pos)

                chunks.append(silence_chunk)
                current_sample_pos += len(silence_chunk)

    spliced = (
        np.concatenate(chunks).astype(np.float32)
        if chunks
        else np.zeros(0, dtype=np.float32)
    )

    original_duration = vad_result.duration_sec
    output_duration = len(spliced) / sr

    return SpliceResult(
        audio=spliced,
        join_indices=join_indices,
        original_duration_sec=original_duration,
        output_duration_sec=output_duration,
        time_saved_sec=max(0.0, original_duration - output_duration),
        pauses_capped=pauses_capped,
    )
