"""Orchestrator and single pipeline entry point for Kannada Pause Remover & Stammer Cleaner.

Implements:
- process(audio_path, config) -> Result
Adheres strictly to plan.md and user specification:
- v1 pipeline: ingest -> VAD -> gap rules -> splicer.
- Result contains clean_audio_path, original_duration, clean_duration, removed_regions.
- Optional fields words, verbatim_text, clean_text, diff are None until M3-M5 exist.
- Never fakes or placeholders ASR, alignment, or LLM text.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
import soundfile as sf

from backend.config import AppConfig, config as default_config
from backend.pipeline.ingest import ingest_audio
from backend.pipeline.splicer import apply_pause_policy_to_gaps, splice_audio
from backend.pipeline.vad import detect_speech_regions

logger = logging.getLogger("kannada_cleaner.orchestrator")


@dataclass(frozen=True)
class RemovedRegion:
    """Represents a time interval removed from the original audio."""

    start: float
    end: float
    reason: str  # e.g., "disfluent_pause"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "reason": self.reason,
        }


@dataclass
class Result:
    """Result data contract for the single pipeline entry point."""

    clean_audio_path: str
    original_duration: float
    clean_duration: float
    removed_regions: List[Dict[str, Any]]
    words: Optional[List[Dict[str, Any]]] = None
    verbatim_text: Optional[str] = None
    clean_text: Optional[str] = None
    diff: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_audio_path": self.clean_audio_path,
            "original_duration": round(self.original_duration, 3),
            "clean_duration": round(self.clean_duration, 3),
            "removed_regions": self.removed_regions,
            "words": self.words,
            "verbatim_text": self.verbatim_text,
            "clean_text": self.clean_text,
            "diff": self.diff,
        }


def extract_removed_regions(
    timeline: List[tuple[float, float, str, float]]
) -> List[RemovedRegion]:
    """Calculate the exact original intervals removed under the pause policy.

    Args:
        timeline: List of (orig_start, orig_end, item_type, target_dur) tuples.

    Returns:
        List of RemovedRegion objects with coordinates in the original audio timeline.
    """
    removed: List[RemovedRegion] = []

    for orig_start, orig_end, item_type, target_dur in timeline:
        orig_dur = orig_end - orig_start
        if item_type != "speech" and orig_dur > target_dur:
            # The kept silence is at the beginning of the pause; the tail is cut
            cut_start = orig_start + target_dur
            cut_end = orig_end
            if cut_end > cut_start:
                removed.append(
                    RemovedRegion(
                        start=cut_start,
                        end=cut_end,
                        reason="disfluent_pause",
                    )
                )

    return removed


def process(
    audio_path: str | Path,
    config: AppConfig | None = None,
    output_path: str | Path | None = None,
) -> Result:
    """Single pipeline entry point for processing audio.

    Currently executes the v1 pipeline:
    Stage 0 (Ingest) -> Stage 1 (VAD) -> Gap rules -> Stage 5A (Splicer).

    Stages 2 (ASR), 3 (Alignment), and 4 (LLM Classifier) are deferred to M3-M5,
    so words, verbatim_text, clean_text, and diff remain strictly None.

    Args:
        audio_path: Path to the input audio file.
        config: Application configuration (defaults to global AppConfig).
        output_path: Optional destination path for the clean WAV.

    Returns:
        Result object adhering to data contracts.
    """
    cfg = config or default_config
    inp_path = Path(audio_path).resolve()

    if not inp_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {inp_path}")

    # Determine destination path for clean audio
    if output_path is not None:
        clean_path = Path(output_path).resolve()
    else:
        out_dir = cfg.paths.data_dir / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        clean_path = out_dir / f"{inp_path.stem}_{unique_id}_clean.wav"

    clean_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 0: Ingest (16 kHz mono float32 peak normalized)
    audio = ingest_audio(file_path=inp_path, target_sr=cfg.audio.sample_rate)
    sr = cfg.audio.sample_rate

    # Stage 1: VAD (Speech regions and pauses)
    vad_result = detect_speech_regions(
        audio,
        sample_rate=sr,
        threshold=cfg.models.vad_threshold,
    )

    # Gap rules: calculate timeline & removed regions
    timeline = apply_pause_policy_to_gaps(
        vad_result,
        min_gap_to_shorten_sec=cfg.pause.min_gap_to_shorten_ms / 1000.0,
        shortened_gap_sec=cfg.pause.shortened_gap_target_ms / 1000.0,
        edge_silence_trim_sec=cfg.pause.edge_silence_trim_ms / 1000.0,
    )
    removed_regions_list = [r.to_dict() for r in extract_removed_regions(timeline)]

    # Stage 5A: Splicer (Zero-crossing snap, pause capping, crossfade)
    splice_result = splice_audio(
        audio=audio,
        vad_result=vad_result,
        crossfade_ms=cfg.pause.crossfade_ms,
        snap_window_ms=cfg.audio.snap_window_ms,
    )

    # Write cleaned audio to disk
    sf.write(str(clean_path), splice_result.audio, sr)

    return Result(
        clean_audio_path=str(clean_path),
        original_duration=splice_result.original_duration_sec,
        clean_duration=splice_result.output_duration_sec,
        removed_regions=removed_regions_list,
        words=None,
        verbatim_text=None,
        clean_text=None,
        diff=None,
    )
