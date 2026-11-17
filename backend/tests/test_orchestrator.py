"""Unit tests for single pipeline entry point process(audio_path, config) -> Result."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.pipeline import Result, process
from backend.pipeline.orchestrator import extract_removed_regions


def _create_synthetic_burst(duration_sec: float, sr: int = 16000, f0: float = 180.0) -> np.ndarray:
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * f0 * t).astype(np.float32)
    return sig


def test_extract_removed_regions() -> None:
    """Verify calculation of removed time intervals."""
    # Timeline:
    # 0.0 to 1.0: leading silence, target 0.1s -> removed: 0.1 to 1.0 (dur 0.9s)
    # 1.0 to 3.0: speech, target 2.0s -> nothing removed
    # 3.0 to 5.0: pause, target 0.2s -> removed: 3.2 to 5.0 (dur 1.8s)
    # 5.0 to 7.0: speech, target 2.0s -> nothing removed
    # 7.0 to 8.0: trailing silence, target 0.1s -> removed: 7.1 to 8.0 (dur 0.9s)
    timeline = [
        (0.0, 1.0, "leading_silence", 0.1),
        (1.0, 3.0, "speech", 2.0),
        (3.0, 5.0, "pause", 0.2),
        (5.0, 7.0, "speech", 2.0),
        (7.0, 8.0, "trailing_silence", 0.1),
    ]

    regions = extract_removed_regions(timeline)
    assert len(regions) == 3

    # Leading cut
    assert regions[0].start == pytest.approx(0.1)
    assert regions[0].end == pytest.approx(1.0)
    assert regions[0].reason == "disfluent_pause"

    # Inter-pause cut
    assert regions[1].start == pytest.approx(3.2)
    assert regions[1].end == pytest.approx(5.0)
    assert regions[1].reason == "disfluent_pause"

    # Trailing cut
    assert regions[2].start == pytest.approx(7.1)
    assert regions[2].end == pytest.approx(8.0)
    assert regions[2].reason == "disfluent_pause"


def test_process_v1_pipeline_e2e() -> None:
    """Verify single pipeline entry point process() returns contract conforming Result."""
    sr = 16000
    # 0.5s silence + 1.5s speech + 1.5s pause + 1.5s speech + 0.5s silence = 5.5s
    audio = np.concatenate([
        np.zeros(int(0.5 * sr), dtype=np.float32),
        _create_synthetic_burst(1.5, sr=sr, f0=180.0),
        np.zeros(int(1.5 * sr), dtype=np.float32),
        _create_synthetic_burst(1.5, sr=sr, f0=220.0),
        np.zeros(int(0.5 * sr), dtype=np.float32),
    ])

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_file = Path(tmp_dir) / "test_input.wav"
        output_file = Path(tmp_dir) / "test_clean.wav"
        sf.write(str(input_file), audio, sr)

        res: Result = process(input_file, output_path=output_file)

        # Invariants & data contract checks
        assert isinstance(res, Result)
        assert Path(res.clean_audio_path).is_file()
        assert res.original_duration == pytest.approx(5.5, abs=0.05)
        assert res.clean_duration < res.original_duration
        assert len(res.removed_regions) >= 1
        for region in res.removed_regions:
            assert "start" in region
            assert "end" in region
            assert "reason" in region
            assert region["start"] < region["end"]

        # Missing stages MUST return None (never placeholder strings)
        assert res.words is None
        assert res.verbatim_text is None
        assert res.clean_text is None
        assert res.diff is None
