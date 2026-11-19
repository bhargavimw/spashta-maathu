"""Stage 3: Verbatim transcript and word-level timestamps via MMS-1B + kan adapter.

Replaces Faster-Whisper ASR per SPEC.md §3: "Do not use Whisper: it silently
removes disfluencies and is weak on Kannada."

Uses facebook/mms-1b-all with the kan adapter (character-level CTC).
Word and character timestamps come from torchaudio.functional.forced_align
over the CTC posteriors.

Two modes:
  (a) Free ASR: greedy-decode the CTC emissions for the transcript, then
      forced_align that transcript against the SAME emissions (no second
      model pass). MMS does not output word timestamps natively.
  (b) Script mode: skip decoding and forced_align a known script directly.

Frame stride is 320 samples (20 ms at 16 kHz). Timestamps are converted from
frames using that stride. Runs per VAD speech segment from Stage 1, adding
each segment's start offset to its timestamps — never feeds a whole file in
one pass.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from backend.config import config as _global_config
from backend.pipeline.vad import VADResult

logger = logging.getLogger("kannada_cleaner.asr")

# MMS-1B frame stride: 320 samples = 20 ms at 16 kHz.
FRAME_STRIDE_SAMPLES = 320
SAMPLE_RATE = 16000

_CACHED_PROCESSOR: Optional[Any] = None
_CACHED_MODEL: Optional[Any] = None
_CACHED_BACKEND: Optional[str] = None


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "transformers is required for MMS ASR. "
            "Install it with: pip install transformers"
        ) from exc


def get_asr_model(
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load MMS-1B processor and model with the kan adapter.

    Returns (processor, model). The model is cached; call reset_asr_model()
    to free VRAM before loading the aligner or LLM (SPEC.md sequential stage
    rule, 8 GB VRAM budget).

    Runs fp16 on GPU when CUDA is available; CPU otherwise.
    """
    _require_transformers()

    global _CACHED_PROCESSOR, _CACHED_MODEL, _CACHED_BACKEND

    backend = backend or _global_config.models.asr_backend
    model_id = model_id or _global_config.models.asr_model_id

    if _CACHED_PROCESSOR is None or _CACHED_BACKEND != backend:
        _CACHED_PROCESSOR = None
        _CACHED_MODEL = None
        _CACHED_BACKEND = None

        if backend != "mms":
            raise ValueError(
                f"Unsupported ASR backend '{backend}'. Only 'mms' is supported."
            )

        logger.info(
            "Loading MMS-1B processor and model '%s' with kan adapter", model_id
        )
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        processor.tokenizer.set_target_lang("kan")

        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        model.load_adapter("kan")
        model.eval()

        # Startup assertion: kan adapter must be loaded and tokenizer must
        # contain Kannada characters; fail loudly if not.
        vocab = processor.tokenizer.get_vocab()
        kan_chars = [c for c in vocab if "\u0C80" <= c <= "\u0CFF"]
        if not kan_chars:
            raise RuntimeError(
                "Kannada adapter not loaded: tokenizer vocab has no Kannada "
                "characters (U+0C80–U+0CFF). Check that set_target_lang('kan') "
                "and load_adapter('kan') ran correctly for model '%s'."
                % model_id
            )
        logger.info(
            "Kannada adapter verified: %d Kannada characters in vocab",
            len(kan_chars),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            model = model.half()
        model = model.to(device)

        # Print emissions.max() once at startup: must be <= 0 after log_softmax.
        _dummy = torch.zeros(1, 1600, dtype=torch.float32, device=device)
        with torch.no_grad():
            _logits = model(_dummy).logits
            _em = torch.nn.functional.log_softmax(_logits, dim=-1)
        print(f"[ASR startup] emissions.max() = {float(_em.max()):.6f} (must be <= 0)")
        if float(_em.max()) > 0.0:
            raise RuntimeError(
                f"emissions.max() = {float(_em.max()):.6f} > 0; "
                "log_softmax not applied correctly."
            )

        _CACHED_PROCESSOR = processor
        _CACHED_MODEL = model
        _CACHED_BACKEND = backend

    return _CACHED_PROCESSOR, _CACHED_MODEL


def reset_asr_model() -> None:
    """Evict cached ASR model to free VRAM between pipeline stages."""
    global _CACHED_PROCESSOR, _CACHED_MODEL, _CACHED_BACKEND
    _CACHED_PROCESSOR = None
    _CACHED_MODEL = None
    _CACHED_BACKEND = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rms_per_frame(audio_chunk: np.ndarray, frame_stride: int = FRAME_STRIDE_SAMPLES) -> np.ndarray:
    """Compute RMS energy per 20 ms frame of the segment audio."""
    num_frames = len(audio_chunk) // frame_stride
    if num_frames == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        frame = audio_chunk[i * frame_stride:(i + 1) * frame_stride]
        rms[i] = float(np.sqrt(np.mean(frame ** 2)))
    return rms


def _extend_word_boundaries(
    spike_starts: List[int],
    spike_ends: List[int],
    rms: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int]]:
    """Grow each word into adjacent blank frames while 20 ms RMS > threshold.

    Never crosses into the neighbouring word's spike frames.
    threshold is relative to the segment median RMS (caller passes threshold already scaled).
    """
    num_frames = len(rms)
    extended: List[Tuple[int, int]] = []
    for i in range(len(spike_starts)):
        ws, we = spike_starts[i], spike_ends[i]

        # Extend start backwards into blank frames.
        es = ws
        prev_end = spike_ends[i - 1] if i > 0 else 0
        while es > prev_end and es > 0 and rms[es - 1] > threshold:
            es -= 1

        # Extend end forwards into blank frames.
        ee = we
        next_start = spike_starts[i + 1] if i < len(spike_starts) - 1 else num_frames
        while ee < next_start and ee < num_frames - 1 and rms[ee] > threshold:
            ee += 1

        extended.append((es, ee))
    return extended


def _check_unk_chars(processor: Any, text: str, context: str = "") -> List[str]:
    """Report characters that map to the tokenizer's UNK token."""
    vocab = processor.tokenizer.get_vocab()
    unk_chars = [ch for ch in text if ch.strip() and ch not in vocab]
    if unk_chars:
        logger.warning(
            "Characters map to UNK%s: %s",
            f" ({context})" if context else "",
            unk_chars,
        )
    return unk_chars


def _compute_emissions(
    model: Any, processor: Any, audio_chunk: np.ndarray, device: str
) -> torch.Tensor:
    """Run MMS-1B forward pass and return log-softmax emissions (T, vocab)."""
    inputs = processor(
        audio_chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        logits = model(input_values).logits  # (T, vocab_size)
        emissions = torch.nn.functional.log_softmax(logits, dim=-1)

    return emissions


def _greedy_decode(processor: Any, emissions: torch.Tensor) -> str:
    """Greedy CTC decode: argmax per frame, collapse repeats, remove blanks.

    Returns the transcript string with "|" word delimiters replaced by spaces.
    """
    # emissions may be (1, T, C) from batched model output; squeeze to (T, C).
    if emissions.dim() == 3:
        emissions = emissions.squeeze(0)
    token_ids = emissions.argmax(dim=-1)  # (T,)
    blank_id = processor.tokenizer.pad_token_id

    prev_id: Optional[int] = None
    text_chars: List[str] = []
    for tid in token_ids.tolist():
        tid = int(tid)
        if tid == blank_id:
            prev_id = None
            continue
        if tid != prev_id:
            text_chars.append(processor.tokenizer.convert_ids_to_tokens([tid])[0])
            prev_id = tid

    text = "".join(text_chars)
    text = text.replace("|", " ")
    return text


def _align_transcript(
    processor: Any,
    emissions: torch.Tensor,
    text: str,
    device: str,
    audio_chunk: Optional[np.ndarray] = None,
) -> List[Tuple[str, float, float, float, float, float, int, int]]:
    """Force-align a transcript string against CTC emissions.

    Uses torchaudio.functional.forced_align + merge_tokens per SPEC.md §3.
    Uses the MMS processor's tokenizer (not a hand-built vocab). The "|" word
    delimiter token preserves word boundaries through alignment. Before
    aligning, checks that no characters map to UNK.

    Returns list of (word, spike_start_s, spike_end_s, start_s, end_s,
    confidence, spike_start_frame, spike_end_frame).
    start_s/end_s are RMS-extended boundaries; spike_start_s/spike_end_s are
    raw CTC frame boundaries.
    """
    blank_id = processor.tokenizer.pad_token_id
    vocab = processor.tokenizer.get_vocab()
    word_delimiter_id = vocab.get("|")

    # Check for UNK characters before alignment.
    _check_unk_chars(processor, text)

    # Tokenize: text -> token IDs with "|" word delimiters between words.
    encoded = processor.tokenizer(text, return_tensors="pt")
    token_ids = encoded["input_ids"][0].tolist()

    # Tokens must be 2D (1, N) for forced_align.
    tokens = torch.tensor(token_ids, device=device).unsqueeze(0)

    # Force align: paths (1, T) = per-frame target token index,
    # scores (1, T) = per-frame log-prob.
    paths, scores = torchaudio.functional.forced_align(
        emissions,  # (1, T, C) log-probs
        tokens,     # (1, N) target token IDs
        blank=blank_id,
    )

    # Merge repeated tokens and blanks into TokenSpan(token, start, end, score).
    # start/end are frame indices.
    token_spans = torchaudio.functional.merge_tokens(
        paths[0], scores[0], blank=blank_id
    )

    # Group token spans into words using "|" delimiter.
    words: List[Tuple[str, float, float, float, float, float, int, int]] = []
    cur_spans: List[Any] = []

    for ts in token_spans:
        if ts.token == word_delimiter_id:
            if cur_spans:
                word_text = "".join(
                    processor.tokenizer.convert_ids_to_tokens([int(s.token)])[0]
                    for s in cur_spans
                )
                spike_start_frame = cur_spans[0].start
                spike_end_frame = cur_spans[-1].end
                spike_start_s = spike_start_frame * FRAME_STRIDE_SAMPLES / SAMPLE_RATE
                spike_end_s = spike_end_frame * FRAME_STRIDE_SAMPLES / SAMPLE_RATE
                confidence = float(np.mean([s.score for s in cur_spans]))
                confidence = float(np.exp(confidence))  # log-prob -> 0-1
                words.append(
                    (word_text, spike_start_s, spike_end_s, 0.0, 0.0,
                     confidence, spike_start_frame, spike_end_frame)
                )
                cur_spans = []
        else:
            cur_spans.append(ts)

    # Flush last word.
    if cur_spans:
        word_text = "".join(
            processor.tokenizer.convert_ids_to_tokens([int(s.token)])[0]
            for s in cur_spans
        )
        spike_start_frame = cur_spans[0].start
        spike_end_frame = cur_spans[-1].end
        spike_start_s = spike_start_frame * FRAME_STRIDE_SAMPLES / SAMPLE_RATE
        spike_end_s = spike_end_frame * FRAME_STRIDE_SAMPLES / SAMPLE_RATE
        confidence = float(np.mean([s.score for s in cur_spans]))
        confidence = float(np.exp(confidence))  # log-prob -> 0-1
        words.append(
            (word_text, spike_start_s, spike_end_s, 0.0, 0.0,
             confidence, spike_start_frame, spike_end_frame)
        )

    # Compute RMS-extended boundaries if audio provided.
    if audio_chunk is not None and words:
        rms = _rms_per_frame(audio_chunk)
        if len(rms) > 0:
            segment_median = float(np.median(rms))
            threshold = 0.3 * segment_median  # 30% of median RMS

            spike_starts = [w[6] for w in words]
            spike_ends = [w[7] for w in words]
            extended = _extend_word_boundaries(spike_starts, spike_ends, rms, threshold)

            # Rebuild words with extended boundaries.
            words_extended: List[Tuple[str, float, float, float, float, float, int, int]] = []
            for i, w in enumerate(words):
                es_frame, ee_frame = extended[i]
                es_s = es_frame * FRAME_STRIDE_SAMPLES / SAMPLE_RATE
                ee_s = ee_frame * FRAME_STRIDE_SAMPLES / SAMPLE_RATE
                words_extended.append(
                    (w[0], w[1], w[2], es_s, ee_s, w[5], w[6], w[7])
                )
            words = words_extended

    return words


def transcribe_vad_segments(
    audio: np.ndarray,
    vad_result: VADResult,
    transcribe_fn: Optional[Callable[[np.ndarray], str]] = None,
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
    script_texts: Optional[List[str]] = None,
) -> List[dict]:
    """
    Transcribe each VAD segment with word-level timestamps using MMS-1B + kan.

    Modes:
      (a) Free ASR: greedy-decode emissions -> transcript, then forced_align
          the transcript against the same emissions (single model pass).
      (b) Script mode: skip decoding; forced_align the given script_texts
          directly against the emissions.

    Args:
        audio: 1-D float32 numpy array at 16 kHz (full ingested audio).
        vad_result: VAD output with speech segments and gaps.
        transcribe_fn: Ignored for MMS backend; kept for backward compatibility.
        backend: ASR backend ("mms"). Defaults to config.models.asr_backend.
        model_id: Model identifier. Defaults to config.models.asr_model_id.
        script_texts: Optional known scripts, one per VAD segment (mode b).

    Returns:
        List of segment dicts:
            index, text, start, end, gap_after, words
        words: [{"word": str, "start_s": float, "end_s": float, "confidence": float}]
        Word timestamps include each VAD segment's start offset.
    """
    if audio.ndim != 1:
        raise ValueError(f"ASR requires 1-D mono audio; got shape {audio.shape}.")
    if audio.dtype != np.float32:
        raise ValueError(f"ASR requires float32 audio; got dtype {audio.dtype}.")
    if len(audio) != vad_result.total_samples:
        raise ValueError(
            f"Audio length ({len(audio)}) does not match "
            f"VAD total_samples ({vad_result.total_samples})."
        )

    segments = vad_result.segments
    if not segments:
        logger.info("ASR: VAD produced no speech segments; returning empty list.")
        return []

    backend = backend or _global_config.models.asr_backend

    # Load model once (cached across segments).
    processor, model = get_asr_model(backend=backend, model_id=model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results: List[dict] = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        seg_start_offset = seg.start_sec
        audio_chunk = audio[seg.start_sample : seg.end_sample].copy()

        if len(audio_chunk) == 0:
            raise RuntimeError(
                f"ASR segment {idx}: audio chunk is empty "
                f"(start={seg.start_sample}, end={seg.end_sample})."
            )

        # Compute emissions once per segment (shared by both modes).
        emissions = _compute_emissions(model, processor, audio_chunk, device)

        # Determine script for this segment (mode b) or None (mode a).
        script = None
        if script_texts is not None and idx < len(script_texts):
            script = script_texts[idx]

        if script is not None:
            # Mode (b): forced_align known script directly; no decoding.
            words = _align_transcript(processor, emissions, script, device, audio_chunk)
            text = script
        else:
            # Mode (a): free ASR - greedy decode then forced_align same emissions.
            text = _greedy_decode(processor, emissions)
            if text.strip():
                words = _align_transcript(processor, emissions, text, device, audio_chunk)
            else:
                words = []

        # gap_after: time to next segment's start; 0.0 for last.
        if idx < total - 1:
            gap_after = round(segments[idx + 1].start_sec - seg.end_sec, 6)
        else:
            gap_after = 0.0

        # Add segment start offset to word timestamps.
        # w tuple: (word, spike_start_s, spike_end_s, start_s, end_s, confidence, spike_start_frame, spike_end_frame)
        words_with_offset = [
            {
                "word": w[0],
                "spike_start_s": round(w[1] + seg_start_offset, 6),
                "spike_end_s": round(w[2] + seg_start_offset, 6),
                "start_s": round(w[3] + seg_start_offset, 6),
                "end_s": round(w[4] + seg_start_offset, 6),
                "confidence": w[5],
                "spike_start_frame": w[6],
                "spike_end_frame": w[7],
                "frames": max(
                    1,
                    round((w[4] - w[3]) * SAMPLE_RATE / FRAME_STRIDE_SAMPLES),
                ),
            }
            for w in words
        ]
        for w in words_with_offset:
            w["suspect"] = w["frames"] < 3  # < 3 CTC frames = 60 ms

        # Frame / token counts for this segment.
        num_frames = emissions.shape[1] if emissions.dim() == 3 else emissions.shape[0]
        num_tokens = len(words)

        results.append(
            {
                "index": idx,
                "text": text,
                "start": seg.start_sec,
                "end": seg.end_sec,
                "gap_after": gap_after,
                "words": words_with_offset,
                "num_frames": num_frames,
                "num_tokens": num_tokens,
            }
        )

        logger.debug(
            "ASR segment %d/%d [%.3f–%.3f s]: %r (%d words)",
            idx + 1,
            total,
            seg.start_sec,
            seg.end_sec,
            text[:60],
            len(words),
        )

    return results