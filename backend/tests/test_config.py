"""Unit tests for Milestone M0: repo scaffold, config, and test harness."""

import os
from unittest.mock import patch

from backend.config import (
    AppConfig,
    AudioConfig,
    HardwareConfig,
    ModelConfig,
    PausePolicyConfig,
    TextConfig,
    get_config,
    setup_logging,
)


def test_trivial_pass() -> None:
    """Trivial passing test to verify pytest environment is operational."""
    assert True


def test_config_loads_with_plan_defaults() -> None:
    """Verify that config.py loads and adheres to all defaults specified in plan.md."""
    cfg = get_config()

    # Audio invariants (plan.md §2, AGENTS.md §8)
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.channels == 1
    assert cfg.audio.dtype == "float32"
    assert cfg.audio.snap_window_ms == 15.0

    # Pause policy defaults (plan.md §3)
    assert cfg.pause.min_gap_to_shorten_ms == 250.0
    assert cfg.pause.shortened_gap_min_ms == 150.0
    assert cfg.pause.shortened_gap_max_ms == 250.0
    assert cfg.pause.clause_boundary_max_gap_ms == 500.0
    assert cfg.pause.edge_silence_trim_ms == 100.0
    assert 10.0 <= cfg.pause.crossfade_ms <= 20.0
    assert cfg.pause.crossfade_curve == "equal-power"

    # Hardware budget & sequential execution (plan.md §4)
    assert cfg.hardware.gpu_vram_budget_gb == 8.0
    assert cfg.hardware.sequential_gpu_execution is True

    # Kannada text rules (plan.md §3, §5, AGENTS.md §8)
    assert cfg.text.unicode_normalization_form == "NFC"
    assert cfg.text.prefer_last_repetition is True
    assert set(cfg.text.valid_removal_reasons) == {
        "repetition",
        "false_start",
        "filler",
        "disfluent_pause",
    }

    # Model IDs defaults (plan.md §4)
    assert "flash" in cfg.models.gemini_model_id.lower()
    assert "conformer" in cfg.models.asr_model_id.lower() or "whisper" in cfg.models.asr_model_id.lower()
    assert "mms" in cfg.models.aligner_model_id.lower()
    assert "silero" in cfg.models.vad_model_id.lower()


def test_config_env_overrides() -> None:
    """Verify that model IDs and tunables are read from environment variables."""
    overrides = {
        "CROSSFADE_MS": "20.0",
        "MIN_GAP_TO_SHORTEN_MS": "300.0",
        "GEMINI_MODEL_ID": "custom-gemini-model",
        "ASR_MODEL_ID": "custom-indic-asr",
        "VAD_THRESHOLD": "0.65",
        "GPU_VRAM_BUDGET_GB": "12.0",
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict(os.environ, overrides, clear=False):
        cfg = get_config()
        assert cfg.pause.crossfade_ms == 20.0
        assert cfg.pause.min_gap_to_shorten_ms == 300.0
        assert cfg.models.gemini_model_id == "custom-gemini-model"
        assert cfg.models.asr_model_id == "custom-indic-asr"
        assert cfg.models.vad_threshold == 0.65
        assert cfg.hardware.gpu_vram_budget_gb == 12.0
        assert cfg.logging.log_level == "DEBUG"


def test_logging_setup() -> None:
    """Verify logger configuration and initialization."""
    logger = setup_logging()
    assert logger.name == "kannada_cleaner"
