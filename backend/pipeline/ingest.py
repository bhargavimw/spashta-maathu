
"""Stage 0: Audio Ingest and Normalization.

Ingests user audio, standardizes to 16 kHz mono float32, and performs peak normalization.

Audio invariants:
- Audio: 16 kHz mono float32 internally.
- Resampling is performed explicitly when input and target rates differ.
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from backend.config import config

logger = logging.getLogger("kannada_cleaner.ingest")


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to single-channel float32."""

    if audio.size == 0:
        raise ValueError("Audio array is empty.")

    if audio.ndim == 1:
        return audio.astype(np.float32)

    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            logger.info(
                "Averaging %d audio channels (shape %s) to mono",
                audio.shape[0],
                audio.shape,
            )
            mono = np.mean(audio, axis=0)
        else:
            logger.info(
                "Averaging %d audio channels (shape %s) to mono",
                audio.shape[1],
                audio.shape,
            )
            mono = np.mean(audio, axis=1)

        return mono.astype(np.float32)

    raise ValueError(
        f"Audio array has unsupported shape {audio.shape}; "
        "expected 1D or 2D array."
    )


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample 1D mono audio to the target sample rate using SciPy."""

    if audio.ndim != 1:
        raise ValueError(
            f"Expected 1D mono audio array, got shape {audio.shape}"
        )

    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(
            f"Sample rates must be positive: "
            f"orig_sr={orig_sr}, target_sr={target_sr}"
        )

    audio = audio.astype(np.float32, copy=False)

    if orig_sr == target_sr:
        return audio

    if audio.size == 0:
        return audio.copy()

    logger.info(
        "Resampling audio from %d Hz to %d Hz",
        orig_sr,
        target_sr,
    )

    # Reduce the sample-rate ratio to its smallest integer form.
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor

    resampled_audio = resample_poly(
        audio,
        up=up,
        down=down,
    )

    return resampled_audio.astype(np.float32)


def peak_normalize(
    audio: np.ndarray,
    target_db: float = -1.0,
) -> np.ndarray:
    """Perform peak amplitude normalization on float32 audio."""

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite values.")

    if audio.size == 0:
        return audio.astype(np.float32)

    peak = float(np.max(np.abs(audio)))

    if peak == 0.0:
        return audio.astype(np.float32)

    target_amplitude = 10.0 ** (target_db / 20.0)
    gain = target_amplitude / peak
    normalized = audio * gain

    normalized = np.clip(
        normalized,
        -1.0,
        1.0,
    ).astype(np.float32)

    return normalized


def load_audio_file(
    file_path: str | Path,
) -> Tuple[np.ndarray, int]:
    """Load an audio file into a float32 NumPy array and sample rate."""

    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    try:
        data, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read audio file '{path}': {exc}"
        ) from exc

    return data, sr


def ingest_audio(
    file_path: str | Path | None = None,
    audio_array: np.ndarray | None = None,
    sr: int | None = None,
    target_sr: int = 16000,
    target_db: float = -1.0,
) -> np.ndarray:
    """Standardize input audio to 16 kHz mono float32 with peak normalization.

    Either file_path or both audio_array and sr must be provided.
    """

    if file_path is None and audio_array is None:
        raise ValueError(
            "Either file_path or audio_array must be provided."
        )

    if file_path is not None:
        raw_audio, input_sr = load_audio_file(file_path)
    else:
        if sr is None:
            raise ValueError(
                "sr (sample rate) must be provided when audio_array "
                "is specified."
            )

        raw_audio = audio_array
        input_sr = sr

    if not isinstance(raw_audio, np.ndarray):
        raw_audio = np.asarray(
            raw_audio,
            dtype=np.float32,
        )

    # 1. Convert to mono float32.
    mono_audio = to_mono(raw_audio)

    # 2. Resample if necessary.
    if input_sr != target_sr:
        resampled_audio = resample_audio(
            mono_audio,
            orig_sr=input_sr,
            target_sr=target_sr,
        )
    else:
        resampled_audio = mono_audio

    # 3. Peak normalize.
    normalized_audio = peak_normalize(
        resampled_audio,
        target_db=target_db,
    )

    # Strict invariant assertions.
    assert normalized_audio.ndim == 1, (
        f"Expected 1D audio, got ndim={normalized_audio.ndim}"
    )

    assert normalized_audio.dtype == np.float32, (
        f"Expected float32 dtype, got {normalized_audio.dtype}"
    )

    assert np.all(np.isfinite(normalized_audio)), (
        "Ingested audio contains NaN or Inf"
    )

    assert target_sr == config.audio.sample_rate, (
        f"Ingest target_sr ({target_sr}) does not match "
        f"config ({config.audio.sample_rate})"
    )

    return normalized_audio
