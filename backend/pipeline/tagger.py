"""Stage 4: Disfluency tagging over aligned word list.

Rule-based tagger per SPEC.md §4 tag taxonomy:

| Tag | Meaning | Detection signal |
|-----|---------|------------------|
| PAUSE | Silence gap in speech | Silero VAD gaps; light 1-3 s, medium 3-5 s, heavy over 5 s |
| FILLER | Hesitation sounds (ಉಹ್, ಮ್ಮ್, ಆಂ) | Lexicon match in verbatim transcript |
| REP | Repeated sound/syllable/word/n-gram | Prefix-of-next-word, or identical word/n-gram twice |
| PROL | Prolonged sound | Word duration z-score vs speaker's own median |
| BLOCK | Silent block inside/before a word | Silence inside word, or abnormal gap before onset |
| FALSESTART | Incomplete fragment, then restart | Fragment followed by restart with same opening |

Core rule: keep the last complete version of any repeated or restarted unit,
and delete the earlier fragments.

CSV schema: file,tag,start_s,end_s,text
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from backend.pipeline.vad import AudioGap, VADResult

PAUSE_TAG = "PAUSE"
FILLER_TAG = "FILLER"
REP_TAG = "REP"
FALSESTART_TAG = "FALSESTART"
PROL_TAG = "PROL"
BLOCK_TAG = "BLOCK"

# Duration thresholds in seconds (SPEC.md §3).
LIGHT_MIN = 1.0
MEDIUM_MIN = 3.0
HEAVY_MIN = 5.0

# Kannada filler-word lexicon (SPEC.md §4: "Lexicon match in verbatim transcript").
FILLER_LEXICON = {
    "ಉಹ್",   # uh
    "ಮ್ಮ್",   # hmm
    "ಆಂ",    # ahm
    "ಅ",     # a
    "ಓಹ್",  # oh
    "ಹ್ಮ್",  # hm
    "ಒ",    # o
    "ಅ ಓಹ್",  # a oh
}

# Z-score threshold for prolongation detection (SPEC.md §4).
PROL_Z_THRESHOLD = 2.0

# Abnormal gap before word onset counts as BLOCK (seconds).
BLOCK_ABNORMAL_GAP_S = 0.5

# False-start fragment max length (chars).
FALSESTART_MAX_CHARS = 6


@dataclass(frozen=True)
class AnnotationRow:
    """Single row in the annotation CSV."""

    file: str
    tag: str
    start_s: float
    end_s: float
    text: str


# ---------------------------------------------------------------------------
# PAUSE (Stage 4a) — from VAD gaps
# ---------------------------------------------------------------------------

def _pause_severity(duration_sec: float) -> Optional[str]:
    """Return severity label for a gap duration, or None if below 1 s threshold."""
    if duration_sec < LIGHT_MIN:
        return None
    if duration_sec < MEDIUM_MIN:
        return "light"
    if duration_sec < HEAVY_MIN:
        return "medium"
    return "heavy"


def tag_pauses(vad_result: VADResult, audio_filename: str) -> List[AnnotationRow]:
    """Tag VAD gaps as PAUSE with severity based on duration.

    Gaps under 1 s are not tagged (returned as empty).
    """
    rows: List[AnnotationRow] = []
    for gap in vad_result.gaps:
        severity = _pause_severity(gap.duration_sec)
        if severity is None:
            continue
        rows.append(
            AnnotationRow(
                file=audio_filename,
                tag=PAUSE_TAG,
                start_s=round(gap.start_sec, 6),
                end_s=round(gap.end_sec, 6),
                text=severity,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# FILLER — from lexicon match
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Normalize a token for comparison: strip whitespace, casefold."""
    return text.strip().casefold()


def tag_fillers(words: List[dict], audio_filename: str) -> List[AnnotationRow]:
    """Tag words that match the Kannada filler lexicon.

    SPEC.md §4: "FILLER — Hesitation sounds (ಉಹ್, ಮ್ಮ್, ಆಂ)
    Detection: Lexicon match in verbatim transcript, plus short voiced-segment
    acoustic check."
    """
    rows: List[AnnotationRow] = []
    for w in words:
        token = _norm(w["word"])
        if token and token in FILLER_LEXICON:
            rows.append(
                AnnotationRow(
                    file=audio_filename,
                    tag=FILLER_TAG,
                    start_s=round(float(w["start_s"]), 6),
                    end_s=round(float(w["end_s"]), 6),
                    text=w["word"],
                )
            )
    return rows


# ---------------------------------------------------------------------------
# REP — prefix-of-next-word or identical word/n-gram twice
# ---------------------------------------------------------------------------

def _is_prefix(short: str, long: str) -> bool:
    """True if short is a proper prefix of long."""
    return bool(short) and len(short) < len(long) and long.startswith(short)


def tag_reps(words: List[dict], audio_filename: str) -> List[AnnotationRow]:
    """Tag repetitions per SPEC.md §4:

    REP — Repeated sound, syllable, word or n-gram.
    Detection: a fragment that is a prefix of the next word
    (ನಾ-ನಾ-ನಾನು), or the same word or n-gram twice in a row.

    Core rule: keep the last complete version, delete earlier fragments.
    Here we tag the earlier fragment(s).
    """
    rows: List[AnnotationRow] = []
    i = 0
    n = len(words)
    while i < n - 1:
        w1 = words[i]
        w2 = words[i + 1]
        t1 = _norm(w1["word"])
        t2 = _norm(w2["word"])

        if not t1 or not t2:
            i += 1
            continue

        # Identical word twice in a row -> tag the first occurrence.
        if t1 == t2:
            rows.append(
                AnnotationRow(
                    file=audio_filename,
                    tag=REP_TAG,
                    start_s=round(float(w1["start_s"]), 6),
                    end_s=round(float(w1["end_s"]), 6),
                    text=w1["word"],
                )
            )
            i += 2
            continue

        # Prefix-of-next-word (fragment of a stammered word) -> tag fragment.
        if _is_prefix(t1, t2):
            rows.append(
                AnnotationRow(
                    file=audio_filename,
                    tag=REP_TAG,
                    start_s=round(float(w1["start_s"]), 6),
                    end_s=round(float(w1["end_s"]), 6),
                    text=w1["word"],
                )
            )
            i += 2
            continue

        i += 1
    return rows


# ---------------------------------------------------------------------------
# FALSESTART — incomplete fragment followed by restart with same opening
# ---------------------------------------------------------------------------

def tag_false_starts(words: List[dict], audio_filename: str) -> List[AnnotationRow]:
    """Tag false starts per SPEC.md §4:

    FALSESTART — Incomplete fragment, then restart.
    Detection: fragment followed by a restart with the same opening
    (ನನ್ನ… ನನ್ನ ಮನೆ). Mark the earlier fragment.

    Heuristic: a short fragment (1-3 chars) immediately followed by a word
    that begins with the same characters.
    """
    rows: List[AnnotationRow] = []
    i = 0
    n = len(words)
    while i < n - 1:
        w1 = words[i]
        w2 = words[i + 1]
        t1 = _norm(w1["word"])
        t2 = _norm(w2["word"])

        if not t1 or not t2:
            i += 1
            continue

        # Fragment is short, and the next word starts with the same chars.
        if len(t1) <= FALSESTART_MAX_CHARS and t2.startswith(t1):
            rows.append(
                AnnotationRow(
                    file=audio_filename,
                    tag=FALSESTART_TAG,
                    start_s=round(float(w1["start_s"]), 6),
                    end_s=round(float(w1["end_s"]), 6),
                    text=w1["word"],
                )
            )
            i += 2
            continue

        i += 1
    return rows


# ---------------------------------------------------------------------------
# PROL — z-score of word duration vs speaker's own median
# ---------------------------------------------------------------------------

def tag_prolongations(
    words: List[dict], audio_filename: str, z_threshold: float = PROL_Z_THRESHOLD
) -> List[AnnotationRow]:
    """Tag prolonged words per SPEC.md §4:

    PROL — Prolonged sound.
    Detection: phoneme or word duration far above the speaker's norm
    (forced alignment). Uses z-score of word duration relative to the
    speaker's own median duration.
    """
    durations = [
        float(w["end_s"]) - float(w["start_s"])
        for w in words
        if _norm(w["word"])
    ]
    if len(durations) < 3:
        return []

    median = statistics.median(durations)
    stdev = statistics.stdev(durations)
    if stdev == 0:
        return []

    rows: List[AnnotationRow] = []
    for w in words:
        token = _norm(w["word"])
        if not token:
            continue
        dur = float(w["end_s"]) - float(w["start_s"])
        z = (dur - median) / stdev
        if z > z_threshold:
            rows.append(
                AnnotationRow(
                    file=audio_filename,
                    tag=PROL_TAG,
                    start_s=round(float(w["start_s"]), 6),
                    end_s=round(float(w["end_s"]), 6),
                    text=f"{w['word']} (z={z:.1f})",
                )
            )
    return rows


# ---------------------------------------------------------------------------
# BLOCK — silence inside a word, or abnormal gap before onset
# ---------------------------------------------------------------------------

def tag_blocks(
    words: List[dict], vad_result: VADResult, audio_filename: str
) -> List[AnnotationRow]:
    """Tag blocks per SPEC.md §4:

    BLOCK — Silent block inside or before a word.
    Detection: silence inside a word, or a long gap just before word onset.

    Uses VAD gaps from Stage 1. A gap overlapping the interior of a word is
    a block; a gap > BLOCK_ABNORMAL_GAP_S immediately before a word onset is
    also a block.
    """
    rows: List[AnnotationRow] = []
    gaps = [(float(g.start_sec), float(g.end_sec)) for g in vad_result.gaps]

    for w in words:
        token = _norm(w["word"])
        if not token:
            continue
        w_start = float(w["start_s"])
        w_end = float(w["end_s"])

        for g_start, g_end in gaps:
            # Silence fully inside the word interior.
            if g_start > w_start and g_end < w_end:
                rows.append(
                    AnnotationRow(
                        file=audio_filename,
                        tag=BLOCK_TAG,
                        start_s=round(g_start, 6),
                        end_s=round(g_end, 6),
                        text="silence_inside_word",
                    )
                )
                break

            # Abnormal gap immediately before word onset.
            if abs(g_end - w_start) < 0.01 and (g_end - g_start) > BLOCK_ABNORMAL_GAP_S:
                rows.append(
                    AnnotationRow(
                        file=audio_filename,
                        tag=BLOCK_TAG,
                        start_s=round(g_start, 6),
                        end_s=round(g_end, 6),
                        text="abnormal_onset_gap",
                    )
                )
                break

    return rows


# ---------------------------------------------------------------------------
# Combined tagger
# ---------------------------------------------------------------------------

def tag_all(
    words: List[dict],
    vad_result: VADResult,
    audio_filename: str,
) -> List[AnnotationRow]:
    """Run all taggers (PAUSE, FILLER, REP, FALSESTART, PROL, BLOCK) and
    return combined, time-sorted annotation rows."""
    rows: List[AnnotationRow] = []
    rows.extend(tag_pauses(vad_result, audio_filename))
    rows.extend(tag_fillers(words, audio_filename))
    rows.extend(tag_reps(words, audio_filename))
    rows.extend(tag_false_starts(words, audio_filename))
    rows.extend(tag_prolongations(words, audio_filename))
    rows.extend(tag_blocks(words, vad_result, audio_filename))
    rows.sort(key=lambda r: (r.start_s, r.end_s))
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_annotation_csv(
    rows: List[AnnotationRow],
    output_path: Path,
) -> None:
    """Write annotation rows to CSV with header file,tag,start_s,end_s,text."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "tag", "start_s", "end_s", "text"])
        for row in rows:
            writer.writerow(
                [row.file, row.tag, f"{row.start_s:.6f}", f"{row.end_s:.6f}", row.text]
            )


def tag_and_write_all(
    words: List[dict],
    vad_result: VADResult,
    audio_filename: str,
    output_path: Path,
) -> List[AnnotationRow]:
    """Convenience: tag all disfluencies and write annotation CSV."""
    rows = tag_all(words, vad_result, audio_filename)
    write_annotation_csv(rows, output_path)
    return rows
