from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.config import config
from backend.pipeline.vad import AudioGap, SpeechSegment, VADResult

logger = logging.getLogger("kannada_cleaner.splicer")

@dataclass
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
    """Determine target timeline of speech and adjusted gaps."""

    timeline: List[Tuple[float, float, str, float]] = []

    if not vad_result.segments:
        target_dur = min(vad_result.duration_sec, edge_silence_trim_sec)
        return [(0.0, vad_result.duration_sec, "leading_silence", target_dur)]

    total_segments = len(vad_result.segments)

    first_seg = vad_result.segments[0]

    if first_seg.start_sec > 0:
        target_lead = min(first_seg.start_sec, edge_silence_trim_sec)
        timeline.append(
            (0.0, first_seg.start_sec, "leading_silence", target_lead)
        )

    for i in range(total_segments):
        seg = vad_result.segments[i]

        timeline.append(
            (seg.start_sec, seg.end_sec, "speech", seg.duration_sec)
        )

        if i < total_segments - 1:
            next_seg = vad_result.segments[i + 1]
            gap_dur = next_seg.start_sec - seg.end_sec

            target_gap = (
                shortened_gap_sec
                if gap_dur > min_gap_to_shorten_sec
                else gap_dur
            )

            timeline.append(
                (seg.end_sec, next_seg.start_sec, "pause", target_gap)
            )

    last_seg = vad_result.segments[-1]

    if last_seg.end_sec < vad_result.duration_sec:
        trail_dur = vad_result.duration_sec - last_seg.end_sec
        target_trail = min(trail_dur, edge_silence_trim_sec)

        timeline.append(
            (last_seg.end_sec, vad_result.duration_sec,
             "trailing_silence", target_trail)
        )

    return timeline


def _match_classifier_segments_to_vad(
    classifier_input: ClassifierInput,
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
    classifier_input: ClassifierInput,
    classifier_output: ClassifierOutput,
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
    classifier_input: ClassifierInput | None = None,
    classifier_output: ClassifierOutput | None = None,
) -> SpliceResult:
    """Splice audio using VAD-only or classifier-guided processing."""

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

    chunks = []
    join_indices = []
    current_sample_pos = 0
    pauses_capped = 0

    ramp_len = min(cf_samples, int(0.010 * sr))

    for orig_start_sec, orig_end_sec, item_type, target_dur in timeline:

        if item_type == "speech":
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

            if len(chunk) >= 2 * ramp_len and ramp_len > 0:
                t = np.linspace(0, 1, ramp_len, dtype=np.float32)
                chunk[:ramp_len] *= np.sin(t * (np.pi / 2.0))
                chunk[-ramp_len:] *= np.cos(t * (np.pi / 2.0))

            if chunks:
                join_indices.append(current_sample_pos)

            chunks.append(chunk)
            current_sample_pos += len(chunk)

        else:
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
def snap_to_zero_crossing(
    audio: np.ndarray, index: int, window_samples: int = 240
) -> int:
    """Find the nearest zero-crossing or lowest-energy sample within a search window.

    Snapping cut points to zero-crossings or low-energy regions avoids clicks, pops,
      and mid-phoneme cut artifacts (see project plan).

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



