"""Unit tests for Evaluation Runner and Metrics (backend/eval/run_eval.py)."""

import unicodedata
import numpy as np
import pytest

from backend.config import config
from backend.eval.run_eval import (
    compute_cer,
    evaluate_disfluency_detection,
    levenshtein_distance,
    run_evaluation_on_clips,
)


def test_levenshtein_distance() -> None:
    """Levenshtein distance calculation is accurate."""
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("ಕನ್ನಡ", "ಕನ್ನಡ") == 0
    assert levenshtein_distance("ಕನ್ನಡ", "ಕನ್ನಡಿಗ") == 2
    assert levenshtein_distance("abc", "def") == 3


def test_compute_cer_unicode_nfc() -> None:
    """CER computation normalizes Unicode to NFC before comparison."""
    ref = "ನಮಸ್ಕಾರ"
    # Create NFD decomposed version
    ref_nfd = unicodedata.normalize("NFD", ref)

    # When compared, both normalize to NFC so CER must be 0.0
    assert compute_cer(ref, ref_nfd) == 0.0
    assert compute_cer(ref, ref) == 0.0

    # With 1 edit
    hyp = "ನಮಸ್ಕಾರಗಳು"
    cer = compute_cer(ref, hyp)
    assert cer > 0.0


def test_evaluate_disfluency_detection_metrics() -> None:
    """Disfluency detection correctly calculates recall, precision, and false-removal rate."""
    gt_labels = [
        {"start_sec": 1.0, "end_sec": 1.4, "type": "disfluent_pause"},
        {"start_sec": 3.0, "end_sec": 3.5, "type": "repetition"},
        {"start_sec": 5.0, "end_sec": 5.2, "type": "filler"},
    ]

    # Two detected regions matching first two, plus one false positive
    detected = [
        {"start": 1.05, "end": 1.38, "reason": "disfluent_pause"},
        {"start": 2.95, "end": 3.48, "reason": "repetition"},
        {"start": 7.00, "end": 7.30, "reason": "disfluent_pause"},  # False positive
    ]

    metrics = evaluate_disfluency_detection(detected, gt_labels, overlap_tolerance_sec=0.15)

    assert metrics["true_positives"] == 2.0
    assert metrics["false_negatives"] == 1.0
    assert metrics["false_positives"] == 1.0
    assert metrics["recall"] == pytest.approx(2.0 / 3.0, abs=0.01)
    assert metrics["precision"] == pytest.approx(2.0 / 3.0, abs=0.01)


def test_run_evaluation_on_clips_smoke() -> None:
    """Evaluation runs over clips and returns structured metrics."""
    sr = 16000
    # Two simple clips
    clip1 = {
        "audio_array": np.zeros(sr * 3, dtype=np.float32),
        "sr": sr,
        "labels": [],
    }
    clip2 = {
        "audio_array": np.zeros(sr * 2, dtype=np.float32),
        "sr": sr,
        "labels": [],
    }

    metrics = run_evaluation_on_clips([clip1, clip2], config)
    assert metrics.num_clips == 2
    assert metrics.avg_original_duration_sec == pytest.approx(2.5)
    assert metrics.total_join_clicks == 0
