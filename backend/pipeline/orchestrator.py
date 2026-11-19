"""Orchestrator and single pipeline entry point for Kannada Pause Remover & Stammer Cleaner.

Implements:
- process(audio_path, config) -> Result          (v1: VAD-only pause removal, backward-compatible)
- process_full(audio_path, config) -> Result     (v2: ASR → Classifier → Splicer full pipeline)

Result contains all original v1 fields plus optional v2 fields:
  verbatim_text, clean_text, diff, words.

The v1 pipeline is preserved unchanged so all existing callers and tests continue to pass.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid

import numpy as np
import soundfile as sf

from backend.config import AppConfig, config as default_config
from backend.pipeline.clean_text import build_clean_text_from_segments
from backend.pipeline.ingest import ingest_audio
from backend.pipeline.splicer import apply_pause_policy_to_gaps, splice_audio
from backend.pipeline.stitch import stitch_audio_with_tags
from backend.pipeline.tagger import tag_all
from backend.pipeline.vad import detect_speech_regions

logger = logging.getLogger("kannada_cleaner.orchestrator")


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RemovedRegion:
    """Represents a time interval removed from the original audio."""

    start: float
    end: float
    reason: str  # e.g., "disfluent_pause", "stammer", "filler"
    segment_index: Optional[int] = None   # classifier segment index (v2 only)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "reason": self.reason,
        }
        if self.segment_index is not None:
            d["segment_index"] = self.segment_index
        return d


@dataclass
class SegmentDiff:
    """Single entry in the verbatim→clean diff (v2 only)."""

    index: int
    text: str
    action: str           # "keep" | "remove"
    reason: Optional[str] = None   # removal reason from classifier

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"index": self.index, "text": self.text, "action": self.action}
        if self.reason is not None:
            d["reason"] = self.reason
        return d


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
    diff: Optional[List[Dict[str, Any]]] = None

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_removed_regions(
    timeline: List[tuple[float, float, str, float]]
) -> List[RemovedRegion]:
    """Calculate the exact original intervals removed under the VAD pause policy.

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


def _make_output_path(
    cfg: AppConfig,
    inp_path: Path,
    output_path: Optional[str | Path],
) -> Path:
    """Resolve or auto-generate a destination path for the cleaned WAV."""
    if output_path is not None:
        clean_path = Path(output_path).resolve()
    else:
        out_dir = cfg.paths.data_dir / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        clean_path = out_dir / f"{inp_path.stem}_{unique_id}_clean.wav"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    return clean_path


# ---------------------------------------------------------------------------
# v1 pipeline – VAD-only pause removal (backward-compatible)
# ---------------------------------------------------------------------------

def process(
    audio_path: str | Path,
    config: AppConfig | None = None,
    output_path: str | Path | None = None,
) -> Result:
    """Single pipeline entry point – v1 (VAD-only pause removal).

    Executes: Ingest → VAD → Gap rules → Splicer.

    Existing callers and tests depend on this function.  Its behavior and
    return value are **unchanged** from previous milestones.

    Args:
        audio_path: Path to the input audio file.
        config: Application configuration (defaults to global AppConfig).
        output_path: Optional destination path for the clean WAV.

    Returns:
        Result with verbatim_text, clean_text, and diff all set to None.
    """
    cfg = config or default_config
    inp_path = Path(audio_path).resolve()

    if not inp_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {inp_path}")

    clean_path = _make_output_path(cfg, inp_path, output_path)

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


EMPTY_TRANSCRIPT_REASON = "empty_transcription"


def split_transcribed_segments(
    asr_segments: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate ASR segments with usable text from those with an empty transcript.

    The classifier rejects empty text, and inventing placeholder text would
    contaminate the transcripts, so empty segments are set aside explicitly
    (the caller reports them in ``removed_regions`` and ``diff``).

    ``gap_after`` of the usable segments is recomputed against the next *usable*
    segment, because the classifier checks it against the actual neighbour.
    Original segment indices and timestamps are preserved.

    Returns:
        ``(usable, untranscribed)`` – both in chronological order.
    """
    usable = [dict(s) for s in asr_segments if s["text"].strip()]
    untranscribed = [dict(s) for s in asr_segments if not s["text"].strip()]

    for cur, nxt in zip(usable, usable[1:]):
        cur["gap_after"] = round(max(0.0, nxt["start"] - cur["end"]), 6)
    if usable:
        usable[-1]["gap_after"] = 0.0

    return usable, untranscribed


# ---------------------------------------------------------------------------
# v2 pipeline – Full ASR → Classifier → Splicer integration (Module 5)
# ---------------------------------------------------------------------------

def process_full(
    audio_path: str | Path,
    config: AppConfig | None = None,
    output_path: str | Path | None = None,
    asr_transcribe_fn: Optional[Callable[[np.ndarray], str]] = None,
    gemini_api_key: Optional[str] = None,
) -> Result:
    """Full pipeline entry point – v2 (ASR → Classifier → Classifier-guided Splicer).

    Executes: Ingest → VAD → ASR → Classifier → Classifier-guided Splicer.

    Args:
        audio_path: Path to the input audio file.
        config: Application configuration (defaults to global AppConfig).
        output_path: Optional destination path for the clean WAV.
        asr_transcribe_fn: Optional callable ``(audio_chunk: np.ndarray) -> str``
            injected for testing.  When ``None`` the real Faster-Whisper model
            is loaded via :func:`~backend.pipeline.asr.get_asr_model`.
        gemini_api_key: Optional Gemini API key override (for tests / CI).
            When ``None`` the value from :pyattr:`config.models.gemini_api_key`
            or the ``GEMINI_API_KEY`` environment variable is used.

    Returns:
        Result with all fields populated.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        RuntimeError: If any pipeline stage fails without a usable fallback.
    """
    # Lazy imports keep this module importable even when heavy deps are absent.
    from backend.pipeline.asr import transcribe_vad_segments          # noqa: PLC0415
    from backend.pipeline.classifier import (                          # noqa: PLC0415
        ClassifierInput,
        classify_segments,
    )

    cfg = config or default_config
    inp_path = Path(audio_path).resolve()

    if not inp_path.is_file():
        raise FileNotFoundError(f"Input audio file not found: {inp_path}")

    clean_path = _make_output_path(cfg, inp_path, output_path)
    sr = cfg.audio.sample_rate

    # ── Stage 0: Ingest ────────────────────────────────────────────────────
    logger.info("Stage 0: Ingesting audio from %s", inp_path)
    audio = ingest_audio(file_path=inp_path, target_sr=sr)

    # ── Stage 1: VAD ───────────────────────────────────────────────────────
    logger.info("Stage 1: Running VAD")
    vad_result = detect_speech_regions(
        audio,
        sample_rate=sr,
        threshold=cfg.models.vad_threshold,
    )

    if not vad_result.segments:
        logger.warning("VAD found no speech segments; writing silence-trimmed audio.")
        splice_result = splice_audio(
            audio=audio,
            vad_result=vad_result,
            crossfade_ms=cfg.pause.crossfade_ms,
            snap_window_ms=cfg.audio.snap_window_ms,
        )
        sf.write(str(clean_path), splice_result.audio, sr)
        return Result(
            clean_audio_path=str(clean_path),
            original_duration=splice_result.original_duration_sec,
            clean_duration=splice_result.output_duration_sec,
            removed_regions=[],
            verbatim_text="",
            clean_text="",
            diff=[],
        )

    # ── Stage 2: ASR ───────────────────────────────────────────────────────
    logger.info("Stage 2: Transcribing %d VAD segments", len(vad_result.segments))
    try:
        asr_segments = transcribe_vad_segments(
            audio=audio,
            vad_result=vad_result,
            transcribe_fn=asr_transcribe_fn,
            backend=cfg.models.asr_backend,
            model_id=cfg.models.asr_model_id,
        )
    except ImportError:
        raise  # already carries an install hint (transformers missing)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Stage 2 (ASR) failed: {exc}") from exc

    usable_segments, untranscribed = split_transcribed_segments(asr_segments)
    if untranscribed:
        logger.warning(
            "ASR returned empty text for segment(s) %s; they are excluded from "
            "classification and reported as removed (%s).",
            [s["index"] for s in untranscribed],
            EMPTY_TRANSCRIPT_REASON,
        )
    if not usable_segments:
        raise RuntimeError(
            "Stage 2 (ASR) produced no transcribable text for any of the "
            f"{len(asr_segments)} VAD segment(s); cannot classify or splice."
        )

    # Verbatim transcript: every ASR segment that has text, in chronological order.
    verbatim_text = " ".join(s.get("text", "") for s in usable_segments)

    # ── Stage 4: Tagger ────────────────────────────────────────────────────
    # Tag disfluencies using word-level timestamps + VAD gaps (SPEC.md §4).
    logger.info(
        "Stage 4: Running disfluency tagger on %d segments", len(asr_segments)
    )
    all_words: List[dict] = []
    for seg in asr_segments:
        for w in seg.get("words", []):
            w_copy = dict(w)
            all_words.append(w_copy)

    tags = tag_all(all_words, vad_result, inp_path.name)

    # ── Stage 6: Clean text ────────────────────────────────────────────────
    # Build clean text deterministically from verbatim + tags (SPEC.md §6).
    logger.info("Stage 6: Building clean text from tags")
    clean_text = build_clean_text_from_segments(asr_segments, tags)

    # ── Stage 3: Classifier (LLM, for diff/comparison) ─────────────────────
    logger.info(
        "Stage 3: Running disfluency classifier on %d segments", len(usable_segments)
    )
    api_key = gemini_api_key or cfg.models.gemini_api_key
    try:
        classifier_input = ClassifierInput.from_list(usable_segments)
        classifier_output = classify_segments(
            raw_segments=usable_segments,
            api_key=api_key,
            model=cfg.models.gemini_model_id,
        )
        classifier_output.validate_against_input(classifier_input)
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise RuntimeError(f"Stage 3 (classifier) failed: {exc}") from exc

    # ── Stage 7: Stitch audio using tags ───────────────────────────────────
    # Remove tagged disfluency spans and stitch remaining speech (SPEC.md §7).
    logger.info(
        "Stage 7: Stitching audio – removing %d tagged spans",
        len(tags),
    )
    try:
        splice_result = stitch_audio_with_tags(
            audio=audio,
            tags=tags,
            sample_rate=sr,
            crossfade_ms=cfg.pause.crossfade_ms,
            snap_window_ms=cfg.audio.snap_window_ms,
        )
    except ValueError as exc:
        raise RuntimeError(f"Stage 7 (stitch) failed: {exc}") from exc

    # Write cleaned audio to disk
    sf.write(str(clean_path), splice_result.audio, sr)

    # ── Build removed_regions / diff (chronological, by segment index) ─────
    source_by_index = classifier_input.by_index()
    removed_reason_map = {r.index: r.reason.value for r in classifier_output.removed}
    kept_set = set(classifier_output.kept_segments)

    removed_entries: List[RemovedRegion] = []

    # Tag-based removed regions (time-based disfluency spans)
    for tag in tags:
        removed_entries.append(
            RemovedRegion(
                start=float(tag.start_s),
                end=float(tag.end_s),
                reason=tag.tag.lower(),
            )
        )

    # Classifier-based removed segments
    for removed_seg in classifier_output.removed:
        src = source_by_index[removed_seg.index]
        removed_entries.append(
            RemovedRegion(
                start=src.start,
                end=src.end,
                reason=removed_seg.reason.value,
                segment_index=removed_seg.index,
            )
        )
    for seg in untranscribed:
        removed_entries.append(
            RemovedRegion(
                start=seg["start"],
                end=seg["end"],
                reason=EMPTY_TRANSCRIPT_REASON,
                segment_index=seg["index"],
            )
        )
    removed_entries.sort(key=lambda r: r.start)
    removed_regions_list = [r.to_dict() for r in removed_entries]

    diff_entries: List[SegmentDiff] = []
    for seg in asr_segments:
        idx = seg["index"]
        if idx in kept_set:
            diff_entries.append(SegmentDiff(index=idx, text=seg["text"], action="keep"))
        elif idx in removed_reason_map:
            diff_entries.append(
                SegmentDiff(
                    index=idx,
                    text=seg["text"],
                    action="remove",
                    reason=removed_reason_map[idx],
                )
            )
        else:  # untranscribed segment (empty text)
            diff_entries.append(
                SegmentDiff(
                    index=idx,
                    text=seg["text"],
                    action="remove",
                    reason=EMPTY_TRANSCRIPT_REASON,
                )
            )

    # Flatten all word-level timestamps for Result.words
    # (Stage 3 ASR already adds segment start offset to word timestamps)
    all_words_with_offset: List[Dict[str, Any]] = []
    for seg in asr_segments:
        for w in seg.get("words", []):
            w_copy = dict(w)
            if "start_s" in w_copy:
                w_copy["start_s"] = round(float(w_copy["start_s"]), 6)
            if "end_s" in w_copy:
                w_copy["end_s"] = round(float(w_copy["end_s"]), 6)
            all_words_with_offset.append(w_copy)

    return Result(
        clean_audio_path=str(clean_path),
        original_duration=splice_result.original_duration_sec,
        clean_duration=splice_result.clean_duration,
        removed_regions=removed_regions_list,
        words=all_words_with_offset,
        verbatim_text=verbatim_text,
        clean_text=clean_text,
        diff=[d.to_dict() for d in diff_entries],
    )
