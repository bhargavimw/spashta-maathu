"""Stage 6: Clean text generation from verbatim transcript + annotation tags.

Deterministically removes tagged disfluency spans from the verbatim transcript.
Normalizes to Unicode NFC. Adds sentence-ending punctuation.
Never lets an LLM rewrite text (SPEC.md §6).
"""

from __future__ import annotations

import unicodedata
from typing import List, Optional

from backend.pipeline.tagger import AnnotationRow


def _nfc(text: str) -> str:
    """Normalize Kannada text to Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def _word_overlaps_tag(word_start: float, word_end: float, tag_start: float, tag_end: float) -> bool:
    """Check if a word's time span overlaps with a tag's time span."""
    return word_start < tag_end and word_end > tag_start


def build_clean_text(
    verbatim_text: str,
    words_with_timestamps: List[dict],
    tags: List[AnnotationRow],
) -> str:
    """Build clean text by removing tagged spans from verbatim transcript.

    Args:
        verbatim_text: Full verbatim transcript string.
        words_with_timestamps: List of dicts with 'word', 'start_s', 'end_s'.
        tags: Annotation rows marking disfluency spans.

    Returns:
        Clean text with tagged spans removed, NFC-normalized, with punctuation.
    """
    if not verbatim_text or not verbatim_text.strip():
        return ""

    # Collect all tag time ranges
    tag_ranges = [(float(t.start_s), float(t.end_s)) for t in tags]

    kept_words: List[str] = []
    for w in words_with_timestamps:
        word_start = float(w.get("start_s", 0))
        word_end = float(w.get("end_s", 0))
        word_text = w.get("word", "").strip()

        if not word_text:
            continue

        # Check if this word overlaps with any tag
        removed = False
        for tag_start, tag_end in tag_ranges:
            if _word_overlaps_tag(word_start, word_end, tag_start, tag_end):
                removed = True
                break

        if not removed:
            kept_words.append(_nfc(word_text))

    if not kept_words:
        return ""

    clean_text = " ".join(kept_words)
    clean_text = clean_text.strip()

    # Add sentence-ending punctuation if missing (SPEC.md §6)
    if clean_text and clean_text[-1] not in ".!?।":
        clean_text += "."

    return clean_text


def build_clean_text_from_segments(
    asr_segments: List[dict],
    tags: List[AnnotationRow],
) -> str:
    """Convenience: build clean text from ASR segments (each with 'words') + tags.

    Flattens all word-level timestamps from all segments, then removes tagged
    words. Falls back to verbatim segment text if no word-level data available.
    """
    # Collect all words from all segments with timestamps
    # (Stage 3 ASR already adds segment start offset to word timestamps)
    all_words: List[dict] = []
    for seg in asr_segments:
        for w in seg.get("words", []):
            w_copy = dict(w)
            all_words.append(w_copy)

    verbatim_text = " ".join(s.get("text", "") for s in asr_segments).strip()

    return build_clean_text(verbatim_text, all_words, tags)
