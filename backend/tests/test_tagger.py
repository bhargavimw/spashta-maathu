"""Tests for Stage 4 tagger (PAUSE, FILLER, REP, FALSESTART, PROL, BLOCK)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from backend.pipeline.tagger import (
    BLOCK_TAG,
    FALSESTART_TAG,
    FILLER_TAG,
    PROL_TAG,
    REP_TAG,
    PAUSE_TAG,
    tag_all,
    tag_blocks,
    tag_false_starts,
    tag_fillers,
    tag_prolongations,
    tag_reps,
)
from backend.pipeline.vad import AudioGap, SpeechSegment, VADResult

SR = 16000


def _make_vad(
    segments: list[tuple[float, float]],
    gaps: list[tuple[float, float, float]],
    total_sec: float,
) -> VADResult:
    segs = [
        SpeechSegment(
            start_sample=int(s * SR),
            end_sample=int(e * SR),
            start_sec=s,
            end_sec=e,
        )
        for s, e in segments
    ]
    gap_objs = [
        AudioGap(
            start_sample=int(s * SR),
            end_sample=int(e * SR),
            start_sec=s,
            end_sec=e,
            duration_sec=d,
        )
        for s, e, d in gaps
    ]
    return VADResult(
        segments=segs,
        gaps=gap_objs,
        sample_rate=SR,
        duration_sec=total_sec,
        total_samples=int(total_sec * SR),
    )


def _word(start_s: float, end_s: float, text: str) -> dict:
    return {
        "word": text,
        "start_s": start_s,
        "end_s": end_s,
        "confidence": 0.9,
    }


# ---------------------------------------------------------------------------
# PAUSE
# ---------------------------------------------------------------------------


def test_pause_light_medium_heavy():
    vad = _make_vad(
        [(0.0, 1.0), (3.0, 4.0), (7.0, 8.0), (9.0, 10.0)],
        [(1.0, 3.0, 2.0), (4.0, 7.0, 3.0), (8.0, 15.0, 7.0)],
        16.0,
    )
    rows = tag_all([], vad, "t.wav")
    tags = [(r.start_s, r.end_s, r.text) for r in rows if r.tag == PAUSE_TAG]
    assert (1.0, 3.0, "light") in tags
    assert (4.0, 7.0, "medium") in tags
    assert (8.0, 15.0, "heavy") in tags


def test_pause_under_one_sec_not_tagged():
    vad = _make_vad([(0.0, 1.0), (1.5, 2.5)], [(1.0, 1.5, 0.5)], 3.0)
    rows = tag_all([], vad, "t.wav")
    assert all(r.tag != PAUSE_TAG for r in rows)


# ---------------------------------------------------------------------------
# FILLER
# ---------------------------------------------------------------------------


def test_filler_lexicon():
    words = [
        _word(0.0, 0.5, "ಉಹ್"),
        _word(0.6, 1.0, "ನಾನು"),
        _word(1.1, 1.6, "ಮ್ಮ್"),
    ]
    rows = tag_fillers(words, "t.wav")
    assert len(rows) == 2
    assert all(r.tag == FILLER_TAG for r in rows)
    assert rows[0].text == "ಉಹ್"
    assert rows[1].text == "ಮ್ಮ್"


# ---------------------------------------------------------------------------
# REP
# ---------------------------------------------------------------------------


def test_rep_identical_word_twice():
    words = [
        _word(0.0, 0.5, "ನಾನು"),
        _word(0.6, 1.1, "ನಾನು"),
        _word(1.2, 1.7, "ಹೋಗುತ್ತೇನೆ"),
    ]
    rows = tag_reps(words, "t.wav")
    assert len(rows) == 1
    assert rows[0].tag == REP_TAG
    assert rows[0].text == "ನಾನು"
    assert rows[0].start_s == 0.0


def test_rep_prefix_of_next():
    words = [
        _word(0.0, 0.3, "ನಾ"),
        _word(0.4, 0.8, "ನಾನು"),
    ]
    rows = tag_reps(words, "t.wav")
    assert len(rows) == 1
    assert rows[0].tag == REP_TAG
    assert rows[0].text == "ನಾ"


# ---------------------------------------------------------------------------
# FALSESTART
# ---------------------------------------------------------------------------


def test_false_start_fragment_restart():
    words = [
        _word(0.0, 0.3, "ನನ್ನ"),
        _word(0.4, 1.0, "ನನ್ನ ಮನೆ"),
    ]
    rows = tag_false_starts(words, "t.wav")
    assert len(rows) == 1
    assert rows[0].tag == FALSESTART_TAG
    assert rows[0].text == "ನನ್ನ"


# ---------------------------------------------------------------------------
# PROL
# ---------------------------------------------------------------------------


def test_prolongation_detected():
    # Six short words + one very long word -> z-score > 2 for the long one.
    words = [
        _word(0.0, 0.2, "ಅ"),
        _word(0.3, 0.5, "ಆ"),
        _word(0.6, 0.8, "ಇ"),
        _word(0.9, 1.1, "ಈ"),
        _word(1.2, 1.4, "ಉ"),
        _word(1.5, 1.7, "ಊ"),
        _word(1.8, 4.3, "ಓōōōōō"),  # prolonged: 2.5s vs 0.2s median
    ]
    rows = tag_prolongations(words, "t.wav")
    assert len(rows) == 1
    assert rows[0].tag == PROL_TAG
    assert "ಓōōōōō" in rows[0].text
    assert rows[0].text.endswith(")")  # z-score annotation present


def test_prolongation_not_enough_words():
    words = [_word(0.0, 0.5, "ನಾನು"), _word(0.6, 1.0, "ಹೇ")]
    rows = tag_prolongations(words, "t.wav")
    assert rows == []


# ---------------------------------------------------------------------------
# BLOCK
# ---------------------------------------------------------------------------


def test_block_silence_inside_word():
    vad = _make_vad(
        [(0.0, 1.0)],
        [(0.3, 0.7, 0.4)],  # gap inside the word 0.0-1.0
        2.0,
    )
    words = [_word(0.0, 1.0, "ನಾನು")]
    rows = tag_blocks(words, vad, "t.wav")
    assert len(rows) == 1
    assert rows[0].tag == BLOCK_TAG
    assert rows[0].text == "silence_inside_word"


def test_block_abnormal_onset_gap():
    vad = _make_vad(
        [(0.0, 1.0), (2.0, 3.0)],
        [(1.0, 2.0, 1.0)],  # 1s gap before word onset at 2.0
        4.0,
    )
    words = [_word(0.0, 1.0, "ನಾನು"), _word(2.0, 3.0, "ಹೇ")]
    rows = tag_blocks(words, vad, "t.wav")
    assert len(rows) == 1
    assert rows[0].tag == BLOCK_TAG
    assert rows[0].text == "abnormal_onset_gap"


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def test_tag_all_sorted_by_time():
    vad = _make_vad(
        [(0.0, 1.0), (2.5, 3.5)],
        [(1.0, 2.5, 1.5)],
        5.0,
    )
    words = [
        _word(0.0, 0.5, "ಉಹ್"),
        _word(3.0, 3.5, "ನಾನು"),
    ]
    rows = tag_all(words, vad, "t.wav")
    assert len(rows) >= 2
    # Sorted by start time
    times = [r.start_s for r in rows]
    assert times == sorted(times)


def test_write_annotation_csv(tmp_path):
    from backend.pipeline.tagger import AnnotationRow, write_annotation_csv

    rows = [
        AnnotationRow("t.wav", PAUSE_TAG, 1.0, 2.5, "light"),
        AnnotationRow("t.wav", FILLER_TAG, 3.0, 3.5, "ಉಹ್"),
    ]
    out = tmp_path / "annotations.csv"
    write_annotation_csv(rows, out)
    content = out.read_text(encoding="utf-8")
    assert "file,tag,start_s,end_s,text" in content
    assert "t.wav,PAUSE,1.000000,2.500000,light" in content
    assert "t.wav,FILLER,3.000000,3.500000,ಉಹ್" in content
