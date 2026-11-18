"""Stage 2: Segment-based Automatic Speech Recognition (ASR).

Transcribes each VAD speech segment independently using Faster-Whisper, preserving
original timestamps and producing output compatible with ClassifierInput.

Design principles:
- Load the model once and reuse it across all segments.
- Never transcribe the entire file as a single utterance.
- Never fabricate or silently discard segment text.
- Keep the model loader injectable for testing without downloading weights.
- CPU-friendly int8 quantisation by default.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import numpy as np

from backend.config import config as _global_config
from backend.pipeline.vad import VADResult

logger = logging.getLogger("kannada_cleaner.asr")

# Type alias for the callable that performs transcription of a single audio
# chunk.  Matching this signature allows unit tests to inject a mock without
# loading real model weights.
TranscribeFn = Callable[[np.ndarray], str]

# Module-level cached model reference.  Populated lazily on first call to
# get_asr_model() and cleared by reset_asr_model() (useful in tests).
_CACHED_ASR_MODEL: Optional[Any] = None
_CACHED_ASR_MODEL_ID: Optional[str] = None


def _load_faster_whisper_model(model_id: str) -> Any:
    """Load a Faster-Whisper model with CPU-friendly int8 quantisation.

    Raises:
        ImportError: When faster_whisper is not installed.
        RuntimeError: When the model cannot be loaded.
    """
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for ASR. "
            "Install it with: pip install faster-whisper"
        ) from exc

    logger.info("Loading Faster-Whisper model '%s' on CPU (int8)", model_id)
    try:
        model = WhisperModel(model_id, device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Faster-Whisper model '{model_id}': {exc}"
        ) from exc

    return model


def get_asr_model(model_id: Optional[str] = None) -> Any:
    """Return the cached Faster-Whisper model, loading it on first call.

    Args:
        model_id: Whisper model identifier (e.g. ``"large-v3"``).
                  Defaults to :pyattr:`config.models.asr_whisper_model_id`.

    Returns:
        A loaded ``WhisperModel`` instance.
    """
    global _CACHED_ASR_MODEL, _CACHED_ASR_MODEL_ID
    mid = model_id or _global_config.models.asr_whisper_model_id
    if _CACHED_ASR_MODEL is None or _CACHED_ASR_MODEL_ID != mid:
        _CACHED_ASR_MODEL = _load_faster_whisper_model(mid)
        _CACHED_ASR_MODEL_ID = mid
    return _CACHED_ASR_MODEL


def reset_asr_model() -> None:
    """Evict the cached ASR model (useful in tests)."""
    global _CACHED_ASR_MODEL, _CACHED_ASR_MODEL_ID
    _CACHED_ASR_MODEL = None
    _CACHED_ASR_MODEL_ID = None


def _transcribe_segment_with_model(model: Any, audio_chunk: np.ndarray) -> str:
    """Run Faster-Whisper on a single audio chunk and return the transcript text.

    Args:
        model: A loaded ``WhisperModel`` instance.
        audio_chunk: 1-D float32 numpy array at 16 kHz.

    Returns:
        Stripped transcript string.  Empty string if no speech detected.
    """
    segments_iter, _info = model.transcribe(
        audio_chunk,
        language="kn",          # Kannada
        beam_size=5,
        vad_filter=False,       # We already have VAD boundaries; skip internal VAD
        word_timestamps=False,
    )
    texts = [seg.text.strip() for seg in segments_iter if seg.text.strip()]
    return " ".join(texts)


def transcribe_vad_segments(
    audio: np.ndarray,
    vad_result: VADResult,
    transcribe_fn: Optional[TranscribeFn] = None,
    model_id: Optional[str] = None,
) -> List[dict]:
    """Transcribe each VAD speech segment independently and return classifier-ready dicts.

    Each returned dict matches the ``ClassifierInput`` segment schema:

    .. code-block:: python

        {
            "index": 0,
            "text": "ನಾನು",
            "start": 0.0,
            "end": 0.45,
            "gap_after": 0.30,
        }

    Args:
        audio: 1-D float32 numpy array at 16 kHz (the full ingested audio).
        vad_result: VAD output containing speech segments and sample rate.
        transcribe_fn: Optional callable ``(audio_chunk: np.ndarray) -> str``.
            When supplied it replaces the real Faster-Whisper inference, which
            makes the function fully testable without model weights.
        model_id: Whisper model identifier passed to :func:`get_asr_model` when
            *transcribe_fn* is ``None``.

    Returns:
        List of segment dicts in chronological order.  Empty list if
        ``vad_result.segments`` is empty.  A segment whose transcription is
        empty keeps ``"text": ""`` (no placeholder is invented); callers must
        handle it before building a ``ClassifierInput``.

    Raises:
        ValueError: If *audio* is not 1-D float32 or its length does not match
            ``vad_result.total_samples``.
        RuntimeError: If a segment transcription raises an unexpected error.
    """
    if audio.ndim != 1:
        raise ValueError(
            f"ASR requires a 1-D mono audio array; got shape {audio.shape}."
        )
    if audio.dtype != np.float32:
        raise ValueError(
            f"ASR requires float32 audio; got dtype {audio.dtype}."
        )
    if len(audio) != vad_result.total_samples:
        raise ValueError(
            f"Audio length ({len(audio)}) does not match "
            f"VAD total_samples ({vad_result.total_samples})."
        )

    segments = vad_result.segments
    if not segments:
        logger.info("ASR: VAD produced no speech segments; returning empty list.")
        return []

    # Build the inference callable once.
    if transcribe_fn is not None:
        infer = transcribe_fn
    else:
        model = get_asr_model(model_id)
        def infer(chunk: np.ndarray) -> str:  # noqa: E306
            return _transcribe_segment_with_model(model, chunk)

    results: List[dict] = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        audio_chunk = audio[seg.start_sample : seg.end_sample].copy()

        if len(audio_chunk) == 0:
            raise RuntimeError(
                f"ASR segment {idx}: audio chunk is empty "
                f"(start={seg.start_sample}, end={seg.end_sample})."
            )

        try:
            text = infer(audio_chunk)
        except Exception as exc:
            raise RuntimeError(
                f"ASR transcription failed for segment {idx} "
                f"({seg.start_sec:.3f}s – {seg.end_sec:.3f}s): {exc}"
            ) from exc

        text = (text or "").strip()
        if not text:
            # Never fabricate transcript text.  The segment is returned with an
            # empty ``text`` so the caller can handle it explicitly (the
            # orchestrator reports it and excludes it from the classifier,
            # which rejects empty text).
            logger.warning(
                "ASR segment %d (%.3f - %.3f s) produced an empty transcript.",
                idx,
                seg.start_sec,
                seg.end_sec,
            )

        # gap_after = time to next segment's start; 0.0 for the last segment
        if idx < total - 1:
            gap_after = round(segments[idx + 1].start_sec - seg.end_sec, 6)
        else:
            gap_after = 0.0

        results.append(
            {
                "index": idx,
                "text": text,
                "start": seg.start_sec,
                "end": seg.end_sec,
                "gap_after": gap_after,
            }
        )

        logger.debug(
            "ASR segment %d/%d [%.3f–%.3f s]: %r",
            idx + 1,
            total,
            seg.start_sec,
            seg.end_sec,
            text[:60],
        )

    return results
