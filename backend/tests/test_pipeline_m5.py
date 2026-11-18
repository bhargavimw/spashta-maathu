"""Tests for Module 5: ASR → Classifier → Splicer pipeline integration.

All tests use mocks so no model weights are downloaded and no real API calls
are made.  The existing VAD-only tests (test_splicer.py, test_orchestrator.py)
are unaffected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from backend.pipeline.asr import reset_asr_model, transcribe_vad_segments
from backend.pipeline.classifier import (
    ClassifierInput,
    ClassifierOutput,
    LLMClassification,
    RemovedSegment,
    RemovalReason,
)
from backend.pipeline.orchestrator import (
    Result,
    RemovedRegion,
    SegmentDiff,
    extract_removed_regions,
    process_full,
)
from backend.pipeline.splicer import splice_audio
from backend.pipeline.vad import AudioGap, SpeechSegment, VADResult

SR = 16000


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _sine(duration_sec: float, f0: float = 200.0, sr: int = SR) -> np.ndarray:
    n = int(sr * duration_sec)
    t = np.arange(n, dtype=np.float32) / sr
    return (0.4 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def _build_vad_result(
    seg_specs: list[tuple[float, float]],
    total_sec: float,
) -> tuple[np.ndarray, VADResult]:
    """Return (audio, vad_result) for the given segment specs."""
    n_total = int(total_sec * SR)
    audio = np.zeros(n_total, dtype=np.float32)
    segs: List[SpeechSegment] = []

    for i, (start_sec, end_sec) in enumerate(seg_specs):
        s, e = int(start_sec * SR), int(end_sec * SR)
        n_seg = e - s
        t = np.arange(n_seg, dtype=np.float32) / SR
        audio[s:e] = (0.4 * np.sin(2 * np.pi * (100 + i * 50) * t)).astype(np.float32)
        segs.append(SpeechSegment(
            start_sample=s, end_sample=e,
            start_sec=start_sec, end_sec=end_sec,
        ))

    vad = VADResult(
        segments=segs,
        gaps=[],
        sample_rate=SR,
        duration_sec=total_sec,
        total_samples=n_total,
    )
    return audio, vad


def _constant_transcribe(texts: list[str]):
    """Return a transcribe_fn that cycles through the given texts."""
    call_count = [0]

    def fn(chunk: np.ndarray) -> str:
        idx = call_count[0] % len(texts)
        call_count[0] += 1
        return texts[idx]

    return fn


# ---------------------------------------------------------------------------
# 1. ASR transcribes VAD segments independently
# ---------------------------------------------------------------------------

def test_asr_transcribes_each_segment_separately() -> None:
    """transcribe_vad_segments calls the transcribe_fn once per VAD segment."""
    seg_specs = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    audio, vad = _build_vad_result(seg_specs, total_sec=6.0)

    call_args: list[np.ndarray] = []

    def spy_fn(chunk: np.ndarray) -> str:
        call_args.append(chunk.copy())
        return "ನಾನು"

    result = transcribe_vad_segments(audio, vad, transcribe_fn=spy_fn)

    assert len(call_args) == 3, "Should call transcribe_fn once per segment"
    # Each chunk must correspond to its VAD segment's sample count
    for i, (start_sec, end_sec) in enumerate(seg_specs):
        expected_len = int((end_sec - start_sec) * SR)
        assert len(call_args[i]) == expected_len, (
            f"Chunk {i} has wrong length: {len(call_args[i])} vs {expected_len}"
        )


# ---------------------------------------------------------------------------
# 2. ASR preserves segment indices and timestamps
# ---------------------------------------------------------------------------

def test_asr_preserves_indices_and_timestamps() -> None:
    """Each returned dict has the correct index, start, and end matching the VAD segment."""
    seg_specs = [(0.5, 1.5), (2.0, 3.0)]
    audio, vad = _build_vad_result(seg_specs, total_sec=4.0)

    texts = ["ಹಲೋ", "ಜಗತ್ತು"]
    result = transcribe_vad_segments(audio, vad, transcribe_fn=_constant_transcribe(texts))

    assert len(result) == 2
    for i, (start_sec, end_sec) in enumerate(seg_specs):
        assert result[i]["index"] == i
        assert result[i]["start"] == pytest.approx(start_sec)
        assert result[i]["end"] == pytest.approx(end_sec)
        assert result[i]["text"] == texts[i]


# ---------------------------------------------------------------------------
# 3. ASR calculates gap_after correctly
# ---------------------------------------------------------------------------

def test_asr_calculates_gap_after() -> None:
    """gap_after equals the gap to the next segment's start; 0.0 for the last."""
    seg_specs = [(0.0, 1.0), (1.5, 2.5), (3.2, 4.0)]
    audio, vad = _build_vad_result(seg_specs, total_sec=5.0)

    result = transcribe_vad_segments(
        audio, vad, transcribe_fn=_constant_transcribe(["a", "b", "c"])
    )

    assert result[0]["gap_after"] == pytest.approx(0.5, abs=1e-5)   # 1.5 - 1.0
    assert result[1]["gap_after"] == pytest.approx(0.7, abs=1e-5)   # 3.2 - 2.5
    assert result[2]["gap_after"] == pytest.approx(0.0, abs=1e-5)   # last segment


# ---------------------------------------------------------------------------
# 4. Empty VAD output is handled safely
# ---------------------------------------------------------------------------

def test_asr_empty_vad_returns_empty_list() -> None:
    """When VAD finds no segments, transcribe_vad_segments returns []."""
    n = int(2.0 * SR)
    silence = np.zeros(n, dtype=np.float32)
    vad = VADResult(segments=[], gaps=[], sample_rate=SR, duration_sec=2.0, total_samples=n)

    # transcribe_fn should never be called
    called = [False]

    def should_not_be_called(chunk):
        called[0] = True
        return "oops"

    result = transcribe_vad_segments(silence, vad, transcribe_fn=should_not_be_called)
    assert result == []
    assert not called[0]


# ---------------------------------------------------------------------------
# 5. ASR output converts correctly to ClassifierInput
# ---------------------------------------------------------------------------

def test_asr_output_converts_to_classifier_input() -> None:
    """ASR dicts round-trip into a valid ClassifierInput without validation errors."""
    seg_specs = [(0.0, 1.0), (1.8, 2.8)]
    audio, vad = _build_vad_result(seg_specs, total_sec=3.5)

    asr_segs = transcribe_vad_segments(
        audio, vad, transcribe_fn=_constant_transcribe(["ಸಂಪೂರ್ಣ", "ಸರಿ"])
    )

    # Should not raise
    ci = ClassifierInput.from_list(asr_segs)
    assert len(ci.segments) == 2
    assert ci.segments[0].index == 0
    assert ci.segments[1].index == 1
    assert ci.segments[0].text == "ಸಂಪೂರ್ಣ"
    assert ci.segments[1].text == "ಸರಿ"
    assert ci.segments[0].gap_after == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# 6. Classifier output passes to classifier-guided splicer
# ---------------------------------------------------------------------------

def test_classifier_output_passes_to_splicer() -> None:
    """When a ClassifierOutput is supplied, splice_audio uses the classifier-guided path."""
    seg_specs = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
    audio, vad = _build_vad_result(seg_specs, total_sec=5.0)

    asr_segs = transcribe_vad_segments(
        audio, vad, transcribe_fn=_constant_transcribe(["ಒಂದು", "ಎರಡು", "ಮೂರು"])
    )
    ci = ClassifierInput.from_list(asr_segs)

    # Keep all three segments with explicit pauses
    llm = LLMClassification(
        kept_segments=[0, 1, 2],
        pause_after_kept=[0.20, 0.20],
        removed=[],
    )
    co = ClassifierOutput.from_llm(llm, ci)

    result = splice_audio(
        audio, vad,
        crossfade_ms=0.0,
        snap_window_ms=0.0,
        classifier_input=ci,
        classifier_output=co,
    )

    assert result.output_duration_sec > 0
    assert np.all(np.isfinite(result.audio))
    # Three 1-second segments + two 0.20-second pauses = ~3.40 s
    assert result.output_duration_sec == pytest.approx(3.40, abs=0.05)


# ---------------------------------------------------------------------------
# 7. Verbatim transcript contains all ASR segments in order
# ---------------------------------------------------------------------------

def test_verbatim_text_contains_all_segments_in_order() -> None:
    """verbatim_text is the concatenation of all ASR segment texts in order."""
    seg_specs = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
    audio, vad = _build_vad_result(seg_specs, total_sec=5.0)

    texts = ["ಮೊದಲ", "ಎರಡನೇ", "ಮೂರನೇ"]
    asr_segs = transcribe_vad_segments(
        audio, vad, transcribe_fn=_constant_transcribe(texts)
    )

    verbatim = " ".join(s["text"] for s in asr_segs)
    assert verbatim == "ಮೊದಲ ಎರಡನೇ ಮೂರನೇ"
    # Order: index 0 before 1 before 2
    for i in range(len(asr_segs) - 1):
        assert asr_segs[i]["index"] < asr_segs[i + 1]["index"]


# ---------------------------------------------------------------------------
# 8. Clean transcript matches the classifier output
# ---------------------------------------------------------------------------

def test_clean_text_matches_classifier_output() -> None:
    """clean_text from ClassifierOutput equals kept segment texts joined by spaces."""
    seg_specs = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
    audio, vad = _build_vad_result(seg_specs, total_sec=5.0)

    asr_segs = transcribe_vad_segments(
        audio, vad,
        transcribe_fn=_constant_transcribe(["ಒಂದು", "ಎರಡು", "ಮೂರು"])
    )
    ci = ClassifierInput.from_list(asr_segs)

    # Classifier removes segment 1 ("ಎರಡು")
    llm = LLMClassification(
        kept_segments=[0, 2],
        pause_after_kept=[0.20],
        removed=[RemovedSegment(index=1, reason=RemovalReason.FILLER)],
    )
    co = ClassifierOutput.from_llm(llm, ci)

    assert co.clean_text == "ಒಂದು ಮೂರು"


# ---------------------------------------------------------------------------
# 9. Removed segment information is represented correctly
# ---------------------------------------------------------------------------

def test_removed_segments_represented_correctly() -> None:
    """The diff list marks kept segments as 'keep' and removed ones as 'remove'."""
    seg_specs = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
    audio, vad = _build_vad_result(seg_specs, total_sec=5.0)

    asr_segs = transcribe_vad_segments(
        audio, vad,
        transcribe_fn=_constant_transcribe(["ಒಂದು", "ಎರಡು", "ಮೂರು"])
    )
    ci = ClassifierInput.from_list(asr_segs)

    llm = LLMClassification(
        kept_segments=[0, 2],
        pause_after_kept=[0.20],
        removed=[RemovedSegment(index=1, reason=RemovalReason.STAMMER)],
    )
    co = ClassifierOutput.from_llm(llm, ci)

    source_by_index = ci.by_index()
    removed_regions = []
    removed_reason_map = {r.index: r.reason.value for r in co.removed}
    kept_set = set(co.kept_segments)

    diff = []
    for seg in asr_segs:
        idx = seg["index"]
        if idx in kept_set:
            diff.append(SegmentDiff(index=idx, text=seg["text"], action="keep").to_dict())
        else:
            diff.append(SegmentDiff(
                index=idx, text=seg["text"], action="remove",
                reason=removed_reason_map.get(idx),
            ).to_dict())

        if idx not in kept_set:
            src = source_by_index[idx]
            removed_regions.append(RemovedRegion(
                start=src.start, end=src.end,
                reason=removed_reason_map[idx],
                segment_index=idx,
            ).to_dict())

    keep_entries = [d for d in diff if d["action"] == "keep"]
    remove_entries = [d for d in diff if d["action"] == "remove"]

    assert len(keep_entries) == 2
    assert len(remove_entries) == 1
    assert remove_entries[0]["index"] == 1
    assert remove_entries[0]["reason"] == "stammer"
    assert remove_entries[0]["text"] == "ಎರಡು"

    assert len(removed_regions) == 1
    assert removed_regions[0]["segment_index"] == 1
    assert removed_regions[0]["start"] == pytest.approx(1.5)
    assert removed_regions[0]["end"] == pytest.approx(2.5)
    assert removed_regions[0]["reason"] == "stammer"


# ---------------------------------------------------------------------------
# 10. Integrated process_full produces the expected audio result
# ---------------------------------------------------------------------------

def test_process_full_end_to_end_with_mocks() -> None:
    """process_full returns a valid Result with verbatim and clean text using mocks."""
    sr = 16000
    # 0.5s silence | 1s speech | 1s gap | 1s speech | 0.5s silence = 4s
    audio = np.concatenate([
        np.zeros(int(0.5 * sr), dtype=np.float32),
        _sine(1.0, f0=150.0),
        np.zeros(int(1.0 * sr), dtype=np.float32),
        _sine(1.0, f0=250.0),
        np.zeros(int(0.5 * sr), dtype=np.float32),
    ])

    # Build a mock VAD result matching the audio structure
    segs = [
        SpeechSegment(
            start_sample=int(0.5 * sr), end_sample=int(1.5 * sr),
            start_sec=0.5, end_sec=1.5,
        ),
        SpeechSegment(
            start_sample=int(2.5 * sr), end_sample=int(3.5 * sr),
            start_sec=2.5, end_sec=3.5,
        ),
    ]
    mock_vad_result = VADResult(
        segments=segs, gaps=[], sample_rate=sr, duration_sec=4.0, total_samples=len(audio)
    )

    # Classifier output: keep both segments, 0.20 s pause
    asr_texts = ["ನಾನು ಹೇಳಿದ", "ಸರಿ ಆಯಿತು"]
    mock_co = ClassifierOutput(
        kept_segments=[0, 1],
        pause_after_kept=[0.20],
        removed=[],
        clean_text="ನಾನು ಹೇಳಿದ ಸರಿ ಆಯಿತು",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / "test_input.wav"
        output_file = Path(tmp_dir) / "test_full_clean.wav"
        sf.write(str(input_file), audio, sr)

        asr_segs = [
            {"index": 0, "text": asr_texts[0], "start": 0.5, "end": 1.5, "gap_after": 1.0},
            {"index": 1, "text": asr_texts[1], "start": 2.5, "end": 3.5, "gap_after": 0.0},
        ]

        with (
            patch("backend.pipeline.orchestrator.ingest_audio", return_value=audio),
            patch("backend.pipeline.orchestrator.detect_speech_regions", return_value=mock_vad_result),
            # patch at source because process_full uses lazy local imports
            patch("backend.pipeline.asr.transcribe_vad_segments", return_value=asr_segs),
            patch("backend.pipeline.classifier.classify_segments", return_value=mock_co),
        ):
            res: Result = process_full(input_file, output_path=output_file)

        assert isinstance(res, Result)
        assert Path(res.clean_audio_path).is_file()
        assert res.verbatim_text == "ನಾನು ಹೇಳಿದ ಸರಿ ಆಯಿತು"
        assert res.clean_text == "ನಾನು ಹೇಳಿದ ಸರಿ ಆಯಿತು"
        assert res.original_duration > 0
        assert res.clean_duration > 0
        assert isinstance(res.diff, list)
        assert len(res.diff) == 2
        assert all(d["action"] == "keep" for d in res.diff)
        assert res.words is None   # alignment not yet implemented


# ---------------------------------------------------------------------------
# 11. Existing VAD-only process() path is unaffected
# ---------------------------------------------------------------------------

def test_process_v1_still_returns_none_text_fields() -> None:
    """The existing VAD-only process() path still sets text fields to None."""
    sr = 16000
    audio = np.concatenate([
        np.zeros(int(0.5 * sr), dtype=np.float32),
        _sine(1.0, f0=180.0),
        np.zeros(int(0.5 * sr), dtype=np.float32),
    ])

    mock_vad = VADResult(
        segments=[SpeechSegment(
            start_sample=int(0.5 * sr), end_sample=int(1.5 * sr),
            start_sec=0.5, end_sec=1.5,
        )],
        gaps=[],
        sample_rate=sr,
        duration_sec=2.0,
        total_samples=len(audio),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / "v1_input.wav"
        output_file = Path(tmp_dir) / "v1_clean.wav"
        sf.write(str(input_file), audio, sr)

        with (
            patch("backend.pipeline.orchestrator.ingest_audio", return_value=audio),
            patch("backend.pipeline.orchestrator.detect_speech_regions", return_value=mock_vad),
        ):
            from backend.pipeline.orchestrator import process
            res = process(input_file, output_path=output_file)

        assert res.verbatim_text is None
        assert res.clean_text is None
        assert res.diff is None
        assert res.words is None
        assert Path(res.clean_audio_path).is_file()


# ---------------------------------------------------------------------------
# 12. Existing Module 3 classifier tests: ClassifierInput validation intact
# ---------------------------------------------------------------------------

def test_classifier_input_validation_still_enforced() -> None:
    """ClassifierInput rejects overlapping timestamps – existing contract unchanged."""
    with pytest.raises(Exception):
        ClassifierInput.from_list([
            {"index": 0, "text": "ಒಂದು", "start": 0.0, "end": 2.0, "gap_after": 0.5},
            # Starts before previous end (1.0 < 2.0 - 0.05 tolerance)
            {"index": 1, "text": "ಎರಡು", "start": 1.0, "end": 3.0, "gap_after": 0.0},
        ])


# ---------------------------------------------------------------------------
# 13. ASR bad input raises ValueError
# ---------------------------------------------------------------------------

def test_asr_rejects_non_1d_audio() -> None:
    """transcribe_vad_segments raises ValueError for non-1D audio."""
    stereo = np.zeros((2, SR), dtype=np.float32)
    vad = VADResult(segments=[], gaps=[], sample_rate=SR, duration_sec=1.0, total_samples=SR)
    with pytest.raises(ValueError, match="1-D"):
        transcribe_vad_segments(stereo, vad)


def test_asr_rejects_mismatched_length() -> None:
    """transcribe_vad_segments raises ValueError when audio length != total_samples."""
    audio = np.zeros(100, dtype=np.float32)
    vad = VADResult(segments=[], gaps=[], sample_rate=SR, duration_sec=1.0, total_samples=SR)
    with pytest.raises(ValueError, match="total_samples"):
        transcribe_vad_segments(audio, vad)


# ---------------------------------------------------------------------------
# 14. process_full handles empty VAD (no speech)
# ---------------------------------------------------------------------------

def test_process_full_no_speech_returns_empty_texts() -> None:
    """process_full with no VAD segments returns empty verbatim and clean text."""
    sr = 16000
    silence = np.zeros(int(2.0 * sr), dtype=np.float32)
    mock_vad = VADResult(
        segments=[], gaps=[], sample_rate=sr, duration_sec=2.0, total_samples=len(silence)
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / "silence.wav"
        output_file = Path(tmp_dir) / "silence_clean.wav"
        sf.write(str(input_file), silence, sr)

        with (
            patch("backend.pipeline.orchestrator.ingest_audio", return_value=silence),
            patch("backend.pipeline.orchestrator.detect_speech_regions", return_value=mock_vad),
        ):
            res = process_full(input_file, output_path=output_file)

        assert res.verbatim_text == ""
        assert res.clean_text == ""
        assert res.diff == []
        assert res.removed_regions == []
        assert Path(res.clean_audio_path).is_file()


# ---------------------------------------------------------------------------
# 15. reset_asr_model evicts the cache
# ---------------------------------------------------------------------------

def test_reset_asr_model_clears_cache() -> None:
    """reset_asr_model() sets the cached model reference back to None."""
    import backend.pipeline.asr as asr_module

    # Inject a fake model into the cache
    asr_module._CACHED_ASR_MODEL = object()
    assert asr_module._CACHED_ASR_MODEL is not None
    reset_asr_model()
    assert asr_module._CACHED_ASR_MODEL is None


# ===========================================================================
# Additional Module 5 tests: failure handling, empty transcripts, model reuse,
# exact-pause splicing, rule-based fallback and the CLI entry point.
# ===========================================================================

import dataclasses
from backend.config import AppConfig, ModelConfig
from backend.pipeline.orchestrator import EMPTY_TRANSCRIPT_REASON, split_transcribed_segments


def _three_segment_audio() -> tuple[np.ndarray, VADResult]:
    """0.5 s lead | seg0 1 s | 1 s gap | seg1 1 s | 1 s gap | seg2 1 s | 0.5 s tail."""
    return _build_vad_result([(0.5, 1.5), (2.5, 3.5), (4.5, 5.5)], total_sec=6.0)


def _run_full(audio, vad, tmp_path, transcribe_fn, classify_fn=None, cfg=None):
    """Run process_full with ingest/VAD/(optionally classifier) mocked."""
    input_file = tmp_path / "in.wav"
    sf.write(str(input_file), audio, SR)
    patches = [
        patch("backend.pipeline.orchestrator.ingest_audio", return_value=audio),
        patch("backend.pipeline.orchestrator.detect_speech_regions", return_value=vad),
    ]
    if classify_fn is not None:
        patches.append(patch("backend.pipeline.classifier.classify_segments", side_effect=classify_fn))
    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return process_full(
            input_file,
            config=cfg,
            output_path=tmp_path / "out.wav",
            asr_transcribe_fn=transcribe_fn,
        )


def test_asr_empty_transcript_is_not_fabricated() -> None:
    """An empty transcription keeps text == '' (no placeholder text is invented)."""
    audio, vad = _build_vad_result([(0.1, 0.6), (1.0, 1.5)], total_sec=2.0)
    texts = iter(["ನಾನು", "   "])
    out = transcribe_vad_segments(audio, vad, transcribe_fn=lambda c: next(texts))
    assert [s["text"] for s in out] == ["ನಾನು", ""]
    assert len(out) == 2  # nothing silently dropped


def test_asr_failure_names_the_segment() -> None:
    audio, vad = _build_vad_result([(0.1, 0.6), (1.0, 1.5)], total_sec=2.0)
    calls = [0]

    def boom(chunk: np.ndarray) -> str:
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("decoder exploded")
        return "ok"

    with pytest.raises(RuntimeError, match=r"segment 1.*decoder exploded"):
        transcribe_vad_segments(audio, vad, transcribe_fn=boom)


def test_asr_model_loaded_once_and_used_per_segment() -> None:
    """The Faster-Whisper model is loaded once and queried once per VAD segment."""
    reset_asr_model()
    audio, vad = _three_segment_audio()

    seg = MagicMock()
    seg.text = " ನಮಸ್ಕಾರ "
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = lambda *a, **k: (iter([seg]), MagicMock())

    with patch("backend.pipeline.asr._load_faster_whisper_model", return_value=fake_model) as loader:
        out = transcribe_vad_segments(audio, vad, model_id="tiny")

    try:
        assert loader.call_count == 1
        assert fake_model.transcribe.call_count == 3
        for call in fake_model.transcribe.call_args_list:
            assert call.kwargs["language"] == "kn"
            assert call.kwargs["vad_filter"] is False
            assert len(call.args[0]) == SR  # each 1 s chunk, not the whole file
        assert [s["text"] for s in out] == ["ನಮಸ್ಕಾರ"] * 3
    finally:
        reset_asr_model()


def test_asr_model_load_failure_is_understandable() -> None:
    from backend.pipeline.asr import _load_faster_whisper_model

    fake_module = MagicMock()
    fake_module.WhisperModel.side_effect = OSError("no such file")
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        with pytest.raises(RuntimeError, match="Failed to load Faster-Whisper model 'tiny'"):
            _load_faster_whisper_model("tiny")


def test_asr_missing_dependency_has_install_hint() -> None:
    from backend.pipeline.asr import _load_faster_whisper_model

    with patch.dict("sys.modules", {"faster_whisper": None}):
        with pytest.raises(ImportError, match="pip install faster-whisper"):
            _load_faster_whisper_model("tiny")


def test_split_transcribed_segments_recomputes_gap_after() -> None:
    segs = [
        {"index": 0, "text": "a", "start": 0.5, "end": 1.5, "gap_after": 1.0},
        {"index": 1, "text": "", "start": 2.5, "end": 3.5, "gap_after": 1.0},
        {"index": 2, "text": "c", "start": 4.5, "end": 5.5, "gap_after": 0.0},
    ]
    usable, untranscribed = split_transcribed_segments(segs)
    assert [s["index"] for s in usable] == [0, 2]
    assert usable[0]["gap_after"] == pytest.approx(3.0)  # to segment 2, not 1
    assert usable[1]["gap_after"] == 0.0
    assert [s["index"] for s in untranscribed] == [1]
    ClassifierInput.from_list(usable)  # passes classifier validation
    assert segs[0]["gap_after"] == 1.0  # input not mutated


def test_process_full_reports_empty_transcript_segment(tmp_path: Path) -> None:
    audio, vad = _three_segment_audio()
    texts = iter(["ಮೊದಲು", "", "ಕೊನೆ"])
    seen = {}

    def classify(raw_segments, api_key=None, model=""):
        seen["indices"] = [s["index"] for s in raw_segments]
        return ClassifierOutput(
            kept_segments=[0, 2],
            pause_after_kept=[0.5],
            removed=[],
            clean_text="ಮೊದಲು ಕೊನೆ",
        )

    res = _run_full(audio, vad, tmp_path, lambda c: next(texts), classify)

    assert seen["indices"] == [0, 2]  # empty segment never reaches the classifier
    assert res.verbatim_text == "ಮೊದಲು ಕೊನೆ"
    assert res.clean_text == "ಮೊದಲು ಕೊನೆ"
    assert [d["index"] for d in res.diff] == [0, 1, 2]
    assert res.diff[1] == {
        "index": 1, "text": "", "action": "remove", "reason": EMPTY_TRANSCRIPT_REASON,
    }
    assert res.removed_regions == [
        {"start": 2.5, "end": 3.5, "reason": EMPTY_TRANSCRIPT_REASON, "segment_index": 1}
    ]
    # 1 s + 0.5 s requested pause + 1 s
    assert res.clean_duration == pytest.approx(2.5, abs=0.05)


def test_process_full_all_segments_empty_raises(tmp_path: Path) -> None:
    audio, vad = _three_segment_audio()
    with pytest.raises(RuntimeError, match="no transcribable text"):
        _run_full(audio, vad, tmp_path, lambda c: "")


def test_process_full_asr_failure_is_wrapped(tmp_path: Path) -> None:
    audio, vad = _three_segment_audio()

    def boom(chunk):
        raise ValueError("bad chunk")

    with pytest.raises(RuntimeError, match="Stage 2 \\(ASR\\) failed"):
        _run_full(audio, vad, tmp_path, boom)


def test_process_full_rejects_classifier_output_inconsistent_with_input(tmp_path: Path) -> None:
    """Output that cites an unknown segment is stopped before the splicer runs."""
    audio, vad = _three_segment_audio()
    texts = iter(["a", "b", "c"])

    def bad_classify(raw_segments, api_key=None, model=""):
        return ClassifierOutput(
            kept_segments=[0, 1, 7], pause_after_kept=[0.2, 0.2], removed=[], clean_text="a b",
        )

    with patch("backend.pipeline.orchestrator.splice_audio") as spl:
        with pytest.raises(RuntimeError, match="Stage 3 \\(classifier\\) failed"):
            _run_full(audio, vad, tmp_path, lambda c: next(texts), bad_classify)
        spl.assert_not_called()


def test_process_full_uses_exact_classifier_pause_not_vad_policy(tmp_path: Path) -> None:
    """A 0.6 s pause is honoured; the VAD policy would have shortened it to 0.2 s."""
    audio, vad = _build_vad_result([(0.5, 1.5), (2.5, 3.5)], total_sec=4.0)
    texts = iter(["ಒಂದು", "ಎರಡು"])

    def classify(raw_segments, api_key=None, model=""):
        return ClassifierOutput(
            kept_segments=[0, 1], pause_after_kept=[0.6], removed=[], clean_text="ಒಂದು ಎರಡು",
        )

    res = _run_full(audio, vad, tmp_path, lambda c: next(texts), classify)
    assert res.clean_duration == pytest.approx(2.6, abs=0.05)


def test_process_full_rule_based_fallback_without_api_key(tmp_path: Path, monkeypatch) -> None:
    """No Gemini key -> the classifier's own rule-based fallback runs; no network."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = AppConfig(models=dataclasses.replace(ModelConfig(), gemini_api_key=None))
    audio, vad = _three_segment_audio()
    texts = iter(["ನಾನು", "ನಾನು", "ಹೋದೆ"])  # adjacent exact repetition

    with patch("backend.pipeline.classifier.load_dotenv"):
        with pytest.warns(UserWarning, match="rule-based fallback"):
            res = _run_full(audio, vad, tmp_path, lambda c: next(texts), cfg=cfg)

    assert res.verbatim_text == "ನಾನು ನಾನು ಹೋದೆ"
    assert res.clean_text == "ನಾನು ಹೋದೆ"
    assert [(d["index"], d["action"], d.get("reason")) for d in res.diff] == [
        (0, "keep", None), (1, "remove", "repetition"), (2, "keep", None),
    ]


def test_cli_clean_speech_success_and_failure(tmp_path: Path, capsys) -> None:
    from backend import cli

    wav = tmp_path / "in.wav"
    sf.write(str(wav), _sine(1.0), SR)
    fake = Result(
        clean_audio_path=str(tmp_path / "o.wav"), original_duration=1.0, clean_duration=0.8,
        removed_regions=[], verbatim_text="a b", clean_text="a",
        diff=[{"index": 0, "text": "a", "action": "keep"},
              {"index": 1, "text": "b", "action": "remove", "reason": "repetition"}],
    )
    with patch("backend.pipeline.orchestrator.process_full", return_value=fake):
        assert cli.main(["clean-speech", "-i", str(wav), "-o", str(tmp_path / "o.wav")]) == 0
    assert "repetition" in capsys.readouterr().out

    with patch("backend.pipeline.orchestrator.process_full", side_effect=RuntimeError("Stage 2 (ASR) failed: x")):
        assert cli.main(["clean-speech", "-i", str(wav)]) == 1
    assert "Stage 2 (ASR) failed" in capsys.readouterr().err

    assert cli.main(["clean-speech", "-i", str(tmp_path / "missing.wav")]) == 1
