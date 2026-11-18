"""Central configuration module for Kannada Pause Remover & Stammer Cleaner.

All model IDs, thresholds, crossfade lengths, and pause policies are defined here.
Values are read from environment variables with safe, documented defaults.
Hardcoding parameters in pipeline logic is prohibited (AGENTS.md).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class AudioConfig:
    """Audio signal constraints and parameters (plan.md §2, AGENTS.md §8)."""

    sample_rate: int = 16000
    channels: int = 1  # mono
    dtype: str = "float32"
    snap_window_ms: float = 15.0  # zero-crossing / low-energy snap window (+/- 15 ms)
    target_peak_norm_db: float = -1.0  # target peak amplitude for ingest normalization


@dataclass(frozen=True)
class PausePolicyConfig:
    """Configurable pause and crossfade policy tunables (plan.md §3)."""

    # Within-clause gaps exceeding this threshold are shortened
    min_gap_to_shorten_ms: float = 250.0

    # Target shortened gap range
    shortened_gap_target_ms: float = 200.0
    shortened_gap_min_ms: float = 150.0
    shortened_gap_max_ms: float = 250.0

    # Clause-boundary gap maximum duration
    clause_boundary_max_gap_ms: float = 500.0

    # Trim leading and trailing silence
    edge_silence_trim_ms: float = 100.0

    # Equal-power crossfade duration (10-20 ms)
    crossfade_ms: float = 15.0
    crossfade_curve: str = "equal-power"


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and inference parameters read from env (plan.md §4)."""

    # Cloud LLM (Gemini)
    gemini_model_id: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash")
    )
    gemini_api_key: str | None = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )

    # Local LLM Fallback (Ollama)
    local_llm_model_id: str = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_MODEL_ID", "gemma2:9b")
    )
    local_llm_host: str = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_HOST", "http://localhost:11434")
    )

    # ASR (Whole-file Kannada transcription — legacy IndicConformer path)
    asr_model_id: str = field(
        default_factory=lambda: os.getenv(
            "ASR_MODEL_ID", "ai4bharat/indicconformer_stt_kn_hybrid"
        )
    )
    asr_backend: str = field(
        default_factory=lambda: os.getenv("ASR_BACKEND", "indicconformer")
    )

    # ASR — Faster-Whisper segment-based transcription (Module 5)
    asr_whisper_model_id: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_ID", "large-v3")
    )

    # Forced Alignment
    aligner_model_id: str = field(
        default_factory=lambda: os.getenv("ALIGNER_MODEL_ID", "facebook/mms-1b-all")
    )

    # Voice Activity Detection (VAD)
    vad_model_id: str = field(
        default_factory=lambda: os.getenv("VAD_MODEL_ID", "silero_vad")
    )
    vad_threshold: float = field(
        default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.5"))
    )


@dataclass(frozen=True)
class HardwareConfig:
    """GPU memory management and device placement (plan.md §4, AGENTS.md §8)."""

    gpu_vram_budget_gb: float = field(
        default_factory=lambda: float(os.getenv("GPU_VRAM_BUDGET_GB", "8.0"))
    )
    sequential_gpu_execution: bool = field(
        default_factory=lambda: os.getenv(
            "SEQUENTIAL_GPU_EXECUTION", "true"
        ).lower()
        in ("1", "true", "yes")
    )
    device: str = field(
        default_factory=lambda: os.getenv("DEVICE", "cuda")
    )


@dataclass(frozen=True)
class TextConfig:
    """Kannada text processing standards and disfluency definitions (plan.md §3, §5)."""

    # Always normalize Kannada text to Unicode NFC before any comparison
    unicode_normalization_form: str = "NFC"

    # In repetitions, prefer keeping the last (most fluent) instance
    prefer_last_repetition: bool = True

    # Valid disfluency removal reason tags
    valid_removal_reasons: Sequence[str] = (
        "repetition",
        "false_start",
        "filler",
        "disfluent_pause",
    )

    # Default list of common Kannada filler words
    kannada_filler_words: Sequence[str] = (
        "ಅಂದರೆ",
        "ಅ...",
        "ಹ್ಮ್...",
        "ಆ...",
        "ಹಾಗೆ",
    )


@dataclass(frozen=True)
class FallbackPolicyConfig:
    """Rule-based fallback policy when LLM output is invalid or fails (plan.md §5)."""

    drop_consecutive_duplicates: bool = True
    drop_known_fillers: bool = True
    cap_gaps: bool = True


@dataclass(frozen=True)
class PathsConfig:
    """Project directory paths (plan.md §8)."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data"
    )
    raw_data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw"
    )
    synthetic_data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "synthetic"
    )
    holdout_data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "holdout"
    )


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass(frozen=True)
class AppConfig:
    """Root configuration holding all sub-configurations."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    pause: PausePolicyConfig = field(
        default_factory=lambda: PausePolicyConfig(
            crossfade_ms=float(os.getenv("CROSSFADE_MS", "15.0")),
            min_gap_to_shorten_ms=float(os.getenv("MIN_GAP_TO_SHORTEN_MS", "250.0")),
            clause_boundary_max_gap_ms=float(os.getenv("CLAUSE_BOUNDARY_MAX_GAP_MS", "500.0")),
            edge_silence_trim_ms=float(os.getenv("EDGE_SILENCE_TRIM_MS", "100.0")),
        )
    )
    models: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    text: TextConfig = field(default_factory=TextConfig)
    fallback: FallbackPolicyConfig = field(default_factory=FallbackPolicyConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def get_config() -> AppConfig:
    """Instantiate and return the primary application configuration."""
    return AppConfig()


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Set up structured application logging."""
    cfg = config or LoggingConfig()
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=cfg.log_format,
        force=True,
    )
    logger = logging.getLogger("kannada_cleaner")
    logger.setLevel(level)
    return logger


# Default global configuration instance
config: AppConfig = get_config()
