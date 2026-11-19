"""Stage 7: Cut and stitch audio using annotation tag spans.

Removes disfluent spans from audio and stitches the remaining speech with
zero-crossing snapping and equal-power crossfade (SPEC.md §7).

- Cut at word boundaries snapped to zero crossings
- Crossfade 10-30 ms at every join to avoid clicks
- Keep or insert a natural pause of 150-300 ms at sentence ends
- Do not apply generative enhancement to the output
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from backend.config import config as _global_config
from backend.pipeline.splicer import crossfade_equal_power, snap_to_zero_crossing
from backend.pipeline.tagger import AnnotationRow

logger = logging.getLogger("kannada_cleaner.stitch")

# Minimum pause at sentence ends (SPEC.md §7: 150-300 ms)
SENTENCE_PAUSE_MIN_MS = 150.0
SENTENCE_PAUSE_MAX_MS = 300.0
SENTENCE_PAUSE_TARGET_MS = 200.0


def _compute_keep_ranges(
    total_duration_sec: float,
    tags: List[AnnotationRow],
) -> List[Tuple[float, float]]:
    """Compute the time ranges to KEEP (everything not in a tag).

    Args:
        total_duration_sec: Total audio duration in seconds.
        tags: Annotation rows marking disfluency spans to remove.

    Returns:
        List of (start_s, end_s) keep ranges, sorted, non-overlapping.
    """
    if not tags:
        return [(0.0, total_duration_sec)]

    # Sort tags by start time
    sorted_tags = sorted(tags, key=lambda t: float(t.start_s))

    keep_ranges: List[Tuple[float, float]] = []
    cursor = 0.0

    for tag in sorted_tags:
        tag_start = float(tag.start_s)
        tag_end = float(tag.end_s)

        # Clamp tag to valid range
        tag_start = max(0.0, min(tag_start, total_duration_sec))
        tag_end = max(0.0, min(tag_end, total_duration_sec))

        if tag_end <= cursor:
            continue  # tag entirely before cursor, skip

        if tag_start > cursor:
            # There's audio to keep between cursor and tag start
            keep_ranges.append((cursor, tag_start))

        cursor = max(cursor, tag_end)

    # Add remaining audio after last tag
    if cursor < total_duration_sec:
        keep_ranges.append((cursor, total_duration_sec))

    # Merge overlapping/adjacent keep ranges
    merged: List[Tuple[float, float]] = []
    for start, end in keep_ranges:
        if merged and start <= merged[-1][1] + 0.001:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Filter out zero-length ranges
    merged = [(s, e) for s, e in merged if e - s > 0.01]

    return merged


def _add_sentence_pause(
    chunks: List[np.ndarray],
    sr: int,
    last_was_sentence_end: bool,
) -> Tuple[List[np.ndarray], bool]:
    """Optionally insert a natural pause at sentence ends.

    Args:
        chunks: Current list of audio chunks.
        sr: Sample rate.
        last_was_sentence_end: Whether the previous chunk ended a sentence.

    Returns:
        (updated chunks, whether current chunk ends a sentence)
    """
    if not chunks:
        return chunks, False

    # Insert a short pause between chunks (natural pacing)
    pause_ms = SENTENCE_PAUSE_TARGET_MS
    pause_samples = int(pause_ms / 1000.0 * sr)
    silence = np.zeros(pause_samples, dtype=np.float32)
    chunks.append(silence)

    return chunks, False


def stitch_audio_with_tags(
    audio: np.ndarray,
    tags: List[AnnotationRow],
    sample_rate: int = 16000,
    crossfade_ms: float = 15.0,
    snap_window_ms: float = 15.0,
) -> Tuple[np.ndarray, List[int], float, float]:
    """Stitch audio by removing tagged disfluency spans.

    Args:
        audio: 1-D float32 numpy array at sample_rate.
        tags: Annotation rows marking spans to remove.
        sample_rate: Audio sample rate in Hz.
        crossfade_ms: Equal-power crossfade duration in ms.
        snap_window_ms: Zero-crossing snap window in ms.

    Returns:
        (stitched_audio, join_indices, original_duration, clean_duration)
    """
    if audio.ndim != 1:
        raise ValueError(f"Audio must be 1-D mono; got shape {audio.shape}.")
    if audio.dtype != np.float32:
        raise ValueError(f"Audio must be float32; got dtype {audio.dtype}.")

    total_duration = len(audio) / sample_rate
    sr = sample_rate

    # Compute keep ranges (everything not tagged)
    keep_ranges = _compute_keep_ranges(total_duration, tags)

    if not keep_ranges:
        # Everything was tagged; return silence
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        return silence, [0], total_duration, 0.5

    crossfade_samples = max(0, int((crossfade_ms / 1000.0) * sr))
    snap_samples = max(0, int((snap_window_ms / 1000.0) * sr))

    chunks: List[np.ndarray] = []
    join_indices: List[int] = []
    current_sample_pos = 0

    for i, (start_sec, end_sec) in enumerate(keep_ranges):
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        # Snap to zero crossings
        snapped_start = snap_to_zero_crossing(audio, start_sample, snap_samples)
        snapped_end = snap_to_zero_crossing(audio, end_sample, snap_samples)

        if snapped_end <= snapped_start:
            snapped_end = min(len(audio), snapped_start + int(0.05 * sr))

        chunk = audio[snapped_start:snapped_end].copy()

        if len(chunk) == 0:
            continue

        # Apply short edge fades to prevent clicks
        ramp_len = min(crossfade_samples, int(0.010 * sr), len(chunk) // 2)
        if ramp_len > 0 and len(chunk) >= 2 * ramp_len:
            t = np.linspace(0, 1, ramp_len, dtype=np.float32)
            chunk[:ramp_len] *= np.sin(t * (np.pi / 2.0))
            chunk[-ramp_len:] *= np.cos(t * (np.pi / 2.0))

        if i == 0:
            chunks.append(chunk)
            current_sample_pos += len(chunk)
        else:
            # Crossfade with previous chunk
            overlap = min(crossfade_samples, len(chunks[-1]), len(chunk))
            if overlap > 0:
                join_indices.append(current_sample_pos - overlap)
                chunks[-1] = crossfade_equal_power(chunks[-1], chunk, overlap)
                current_sample_pos += len(chunk) - overlap
            else:
                join_indices.append(current_sample_pos)
                chunks.append(chunk)
                current_sample_pos += len(chunk)

    spliced = np.concatenate(chunks).astype(np.float32)
    clean_duration = len(spliced) / sr

    return spliced, join_indices, total_duration, clean_duration


def stitch_audio_with_tags_to_file(
    audio: np.ndarray,
    tags: List[AnnotationRow],
    output_path: str,
    sample_rate: int = 16000,
    crossfade_ms: float = 15.0,
    snap_window_ms: float = 15.0,
) -> str:
    """Stitch audio with tags and write to file.

    Returns the output path.
    """
    import soundfile as sf

    spliced, _, _, clean_duration = stitch_audio_with_tags(
        audio, tags, sample_rate, crossfade_ms, snap_window_ms
    )
    sf.write(output_path, spliced, sample_rate)
    return output_path
