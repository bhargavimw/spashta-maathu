"""Unit tests for Synthetic Disfluency Injector (backend/synth/inject.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.synth.inject import (
    add_pitch_and_jitter,
    create_clean_kannada_sentence,
    generate_evaluation_dataset,
    inject_disfluencies_into_sentence,
)


def test_add_pitch_and_jitter() -> None:
    """Pitch and jitter modulation preserves shape and finiteness."""
    sr = 16000
    audio = np.sin(np.linspace(0, 10, 1600, dtype=np.float32))
    modulated = add_pitch_and_jitter(audio, sr=sr)

    assert len(modulated) == len(audio)
    assert modulated.dtype == np.float32
    assert np.all(np.isfinite(modulated))
    # Not bit-identical (realism rule in plan.md §6)
    assert not np.array_equal(audio, modulated)


def test_create_clean_kannada_sentence() -> None:
    """Sentence generation produces 16 kHz audio and accurate word metadata."""
    words = ["ನಮಸ್ಕಾರ", "ಬೆಂಗಳೂರು", "ಸ್ವಾಗತ"]
    audio, meta = create_clean_kannada_sentence(words, sr=16000)

    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(meta) == len(words)
    assert meta[0]["text"] == "ನಮಸ್ಕಾರ"
    assert meta[0]["start"] < meta[0]["end"]


def test_inject_disfluencies_into_sentence() -> None:
    """Disfluency injection produces hesitant audio with ground truth labels."""
    words = ["ನಾನು", "ಕನ್ನಡ", "ಭಾಷೆ", "ಕಲಿಯುತ್ತಿದ್ದೇನೆ"]
    clean_audio, meta = create_clean_kannada_sentence(words, sr=16000)
    hesitant_audio, clean_text, hesitant_text, labels = inject_disfluencies_into_sentence(
        words, clean_audio, meta, sr=16000, rng_seed=42
    )

    assert len(hesitant_audio) > 0
    assert len(clean_text) > 0
    assert len(hesitant_text) >= len(clean_text)
    assert len(labels) > 0

    for l in labels:
        assert l.type in ("repetition", "false_start", "filler", "disfluent_pause")
        assert 0.0 <= l.start_sec < l.end_sec
        assert l.duration_sec > 0.0


def test_generate_evaluation_dataset_files() -> None:
    """Dataset generation writes paired WAVs and valid JSON labels to disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "synth_eval"
        manifest = generate_evaluation_dataset(out_dir, num_clips=3, sr=16000)

        assert len(manifest) == 3
        for item in manifest:
            cid = item["clip_id"]
            clean_file = out_dir / f"{cid}_clean.wav"
            hesitant_file = out_dir / f"{cid}_hesitant.wav"
            labels_file = out_dir / f"{cid}_labels.json"

            assert clean_file.is_file()
            assert hesitant_file.is_file()
            assert labels_file.is_file()
            assert item["hesitant_duration"] > 0
