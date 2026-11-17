"""Stage 1: Voice Activity Detection (VAD).

Uses Silero VAD to detect speech regions and measured gaps in 16 kHz mono float32 audio.
Strictly adheres to:
- plan.md §2 & §4 (Silero VAD on CPU)
- plan.md §7 M1 acceptance test: "VAD returns segments and gaps"
- AGENTS.md §8 (assert shapes and sample rates, pure functions for audio operations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from backend.config import config

logger = logging.getLogger("kannada_cleaner.vad")

_CACHED_VAD_MODEL: Optional[torch.nn.Module] = None


@dataclass(frozen=True)
class SpeechSegment:
    """Represents a continuous region of detected speech."""

    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        """Duration of speech segment in seconds."""
        return self.end_sec - self.start_sec

    @property
    def duration_samples(self) -> int:
        """Duration of speech segment in samples."""
        return self.end_sample - self.start_sample


@dataclass(frozen=True)
class AudioGap:
    """Represents a measured pause or silence between speech regions or edges."""

    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float

    @property
    def duration_ms(self) -> float:
        """Duration of gap in milliseconds."""
        return self.duration_sec * 1000.0


@dataclass(frozen=True)
class VADResult:
    """Encapsulates speech segments and non-speech gaps detected by VAD."""

    segments: List[SpeechSegment]
    gaps: List[AudioGap]
    sample_rate: int
    duration_sec: float
    total_samples: int


def get_vad_model() -> torch.nn.Module:
    """Load and cache the Silero VAD model on CPU."""
    global _CACHED_VAD_MODEL
    if _CACHED_VAD_MODEL is None:
        logger.info("Loading Silero VAD model onto CPU")
        model = load_silero_vad(onnx=False)
        _CACHED_VAD_MODEL = model
    return _CACHED_VAD_MODEL


def compute_gaps(
    segments: List[SpeechSegment], total_samples: int, sample_rate: int
) -> List[AudioGap]:
    """Compute non-speech intervals (gaps) from speech segments and audio boundaries.

    Pure deterministic function calculating:
    - Leading gap (0 to first speech start, if any)
    - Intermediate gaps (between speech segment i and i+1)
    - Trailing gap (last speech end to total_samples, if any)

    Args:
        segments: Sorted list of SpeechSegment objects.
        total_samples: Total number of audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        List of AudioGap objects.
    """
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    if total_samples <= 0:
        return []

    gaps: List[AudioGap] = []

    if not segments:
        # Entire clip is a single non-speech gap
        dur = float(total_samples) / float(sample_rate)
        return [
            AudioGap(
                start_sample=0,
                end_sample=total_samples,
                start_sec=0.0,
                end_sec=dur,
                duration_sec=dur,
            )
        ]

    # 1. Leading silence
    first_seg = segments[0]
    if first_seg.start_sample > 0:
        start_sec = 0.0
        end_sec = float(first_seg.start_sample) / float(sample_rate)
        gaps.append(
            AudioGap(
                start_sample=0,
                end_sample=first_seg.start_sample,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
            )
        )

    # 2. Inter-segment gaps
    for i in range(len(segments) - 1):
        prev_end = segments[i].end_sample
        next_start = segments[i + 1].start_sample
        if next_start > prev_end:
            start_sec = float(prev_end) / float(sample_rate)
            end_sec = float(next_start) / float(sample_rate)
            gaps.append(
                AudioGap(
                    start_sample=prev_end,
                    end_sample=next_start,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=end_sec - start_sec,
                )
            )

    # 3. Trailing silence
    last_seg = segments[-1]
    if last_seg.end_sample < total_samples:
        start_sec = float(last_seg.end_sample) / float(sample_rate)
        end_sec = float(total_samples) / float(sample_rate)
        gaps.append(
            AudioGap(
                start_sample=last_seg.end_sample,
                end_sample=total_samples,
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
            )
        )

    return gaps


def detect_speech_regions(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float | None = None,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> VADResult:
    """Run Silero VAD to detect speech regions and non-speech gaps.

    Args:
        audio: 1D mono float32 numpy array.
        sample_rate: Sample rate in Hz (must be 16000).
        threshold: VAD speech probability threshold (default from config: 0.5).
        min_speech_duration_ms: Minimum duration of a speech segment in ms.
        min_silence_duration_ms: Minimum duration of silence to split speech in ms.

    Returns:
        VADResult containing speech segments and measured gaps.

    Raises:
        ValueError: If audio invariants are violated.
    """
    # Strict audio invariant assertions
    if audio.ndim != 1:
        raise ValueError(f"Audio must be 1D mono, got shape {audio.shape}")
    if audio.dtype != np.float32:
        raise ValueError(f"Audio dtype must be float32, got {audio.dtype}")
    if sample_rate != config.audio.sample_rate:
        raise ValueError(
            f"VAD requires sample_rate={config.audio.sample_rate}, got {sample_rate}"
        )
    if audio.size == 0:
        raise ValueError("Audio array is empty.")

    vad_threshold = threshold if threshold is not None else config.models.vad_threshold
    model = get_vad_model()

    # Convert to torch tensor
    tensor = torch.from_numpy(audio)

    # Silero VAD returns a list of dicts: [{'start': int, 'end': int}, ...] in sample counts
    raw_timestamps = get_speech_timestamps(
        audio=tensor,
        model=model,
        sampling_rate=sample_rate,
        threshold=vad_threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    segments: List[SpeechSegment] = []
    for ts in raw_timestamps:
        start_samp = int(ts["start"])
        end_samp = int(ts["end"])
        segments.append(
            SpeechSegment(
                start_sample=start_samp,
                end_sample=end_samp,
                start_sec=float(start_samp) / float(sample_rate),
                end_sec=float(end_samp) / float(sample_rate),
            )
        )

    total_samples = len(audio)
    duration_sec = float(total_samples) / float(sample_rate)
    gaps = compute_gaps(segments, total_samples=total_samples, sample_rate=sample_rate)

    return VADResult(
        segments=segments,
        gaps=gaps,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        total_samples=total_samples,
    )
