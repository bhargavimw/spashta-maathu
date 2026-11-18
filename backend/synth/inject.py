"""Synthetic Disfluency Injector (plan.md §6).

Generates realistic hesitant Kannada speech from clean audio by injecting:
1. Repetitions: repeated word/syllable bursts with pitch shift and jitter so repeats are not bit-identical copies.
2. False starts: truncated word start (~150 ms) followed by brief hesitation before full word.
3. Fillers: natural acoustic filler sounds (e.g., 'ಅಂದರೆ', 'ಹ್ಮ್', 'ಆ').
4. Disfluent pauses: 200–400 ms unnatural pauses inside phrases.

Outputs: (hesitant.wav, clean.wav, hesitant_text, clean_text, labels.json)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

import numpy as np
import soundfile as sf

from backend.config import config

logger = logging.getLogger("kannada_cleaner.synth")


@dataclass(frozen=True)
class InjectedDisfluency:
    """Ground truth record of an injected disfluency."""

    type: str  # "repetition", "false_start", "filler", "disfluent_pause"
    word_index: int
    text: str
    start_sec: float
    end_sec: float
    duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "word_index": self.word_index,
            "text": self.text,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
        }


def add_pitch_and_jitter(
    audio: np.ndarray,
    sr: int = 16000,
    pitch_mod_cents: float = 25.0,
    jitter_amp: float = 0.02,
) -> np.ndarray:
    """Apply micro-pitch modulation and amplitude jitter so repeats are realistic and non-identical."""
    if len(audio) == 0:
        return audio

    t = np.arange(len(audio), dtype=np.float32) / float(sr)
    # Slow vibrato-like pitch drift
    vibrato = 1.0 + (pitch_mod_cents / 1200.0) * np.sin(2 * np.pi * 5.0 * t).astype(np.float32)
    # Amplitude jitter
    jitter = 1.0 + jitter_amp * np.sin(2 * np.pi * 3.3 * t).astype(np.float32)

    return (audio * vibrato * jitter).astype(np.float32)


def generate_kannada_speech_unit(
    word: str,
    duration_sec: float = 0.5,
    sr: int = 16000,
    f0: float = 160.0,
) -> np.ndarray:
    """Synthesize a rich harmonic vowel/consonant acoustic unit with syllable envelope."""
    n = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)

    # Intonation contour (slight natural downward drift)
    pitch_contour = f0 * (1.0 - 0.08 * (t / max(duration_sec, 0.01)))
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sr

    # Vocal cord harmonics
    signal = (
        0.55 * np.sin(phase)
        + 0.28 * np.sin(2 * phase)
        + 0.16 * np.sin(3 * phase)
        + 0.08 * np.sin(4 * phase)
    ).astype(np.float32)

    # Syllabic envelope with smooth attack and release
    envelope = np.ones(n, dtype=np.float32)
    ramp_len = min(int(0.04 * sr), n // 4)
    if ramp_len > 0:
        envelope[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
        envelope[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)

    return (signal * envelope).astype(np.float32)


def create_clean_kannada_sentence(
    words: List[str],
    sr: int = 16000,
    base_f0: float = 165.0,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Assemble a clean Kannada spoken sentence with word boundaries and natural inter-word pacing."""
    audio_chunks: List[np.ndarray] = []
    word_metadata: List[Dict[str, Any]] = []
    current_time = 0.10  # 100 ms leading natural room tone / silence

    # Add 100 ms leading silence
    audio_chunks.append(np.zeros(int(0.10 * sr), dtype=np.float32))

    for idx, w in enumerate(words):
        dur = max(0.35, min(0.75, 0.12 * len(w)))
        # Slight pitch variation per word
        word_f0 = base_f0 + 5.0 * np.sin(idx * 1.2)
        unit = generate_kannada_speech_unit(w, duration_sec=dur, sr=sr, f0=word_f0)

        start_sec = current_time
        end_sec = current_time + dur
        word_metadata.append({
            "i": idx,
            "text": unicodedata.normalize("NFC", w),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(dur, 3),
        })

        audio_chunks.append(unit)
        current_time += dur

        # Natural inter-word spacing (80–120 ms)
        if idx < len(words) - 1:
            pace_sec = 0.09
            audio_chunks.append(np.zeros(int(pace_sec * sr), dtype=np.float32))
            current_time += pace_sec

    # Add 100 ms trailing silence
    audio_chunks.append(np.zeros(int(0.10 * sr), dtype=np.float32))

    full_audio = np.concatenate(audio_chunks).astype(np.float32)
    return full_audio, word_metadata


def inject_disfluencies_into_sentence(
    words: List[str],
    clean_audio: np.ndarray,
    word_meta: List[Dict[str, Any]],
    sr: int = 16000,
    rng_seed: Optional[int] = None,
) -> Tuple[np.ndarray, str, str, List[InjectedDisfluency]]:
    """Inject realistic Kannada disfluencies (repetitions, false starts, fillers, pauses).

    Returns:
        (hesitant_audio, clean_text, hesitant_text, ground_truth_labels)
    """
    rng = random.Random(rng_seed)
    hesitant_chunks: List[np.ndarray] = []
    hesitant_tokens: List[str] = []
    labels: List[InjectedDisfluency] = []
    current_time = 0.0

    # 1. Natural leading silence (can sometimes be long / disfluent)
    lead_silence_dur = rng.choice([0.15, 0.45, 0.70])
    lead_silence = np.zeros(int(lead_silence_dur * sr), dtype=np.float32)
    hesitant_chunks.append(lead_silence)
    if lead_silence_dur > 0.25:
        labels.append(
            InjectedDisfluency(
                type="disfluent_pause",
                word_index=0,
                text="[leading_silence]",
                start_sec=current_time + 0.10,
                end_sec=current_time + lead_silence_dur,
                duration_sec=lead_silence_dur - 0.10,
            )
        )
    current_time += lead_silence_dur

    for idx, w in enumerate(words):
        dur = max(0.35, min(0.75, 0.12 * len(w)))
        unit = generate_kannada_speech_unit(w, duration_sec=dur, sr=sr)

        # Decide whether to inject a disfluency at this word
        dice = rng.random()

        if dice < 0.25 and idx > 0:
            # Injection 1: Repetition (e.g. "ನಾನು" -> "ನಾನು ನಾನು")
            rep_unit = add_pitch_and_jitter(unit, sr=sr, pitch_mod_cents=35.0)
            rep_start = current_time
            hesitant_chunks.append(rep_unit)
            current_time += dur
            labels.append(
                InjectedDisfluency(
                    type="repetition",
                    word_index=idx,
                    text=w,
                    start_sec=rep_start,
                    end_sec=current_time,
                    duration_sec=dur,
                )
            )
            hesitant_tokens.append(w)

            # Tiny pause between repeats
            rep_gap = 0.06
            hesitant_chunks.append(np.zeros(int(rep_gap * sr), dtype=np.float32))
            current_time += rep_gap

        elif dice < 0.45:
            # Injection 2: False start (first 150 ms of word)
            fs_dur = min(0.15, dur * 0.4)
            fs_samples = int(fs_dur * sr)
            fs_unit = unit[:fs_samples].copy()
            # Cosine decay on false start cut
            decay_len = min(int(0.03 * sr), fs_samples // 2)
            if decay_len > 0:
                t = np.linspace(0, 1, decay_len, dtype=np.float32)
                fs_unit[-decay_len:] *= np.cos(t * (np.pi / 2.0))

            fs_start = current_time
            hesitant_chunks.append(fs_unit)
            current_time += fs_dur
            labels.append(
                InjectedDisfluency(
                    type="false_start",
                    word_index=idx,
                    text=w[:max(1, len(w) // 2)],
                    start_sec=fs_start,
                    end_sec=current_time,
                    duration_sec=fs_dur,
                )
            )
            hesitant_tokens.append(w[:max(1, len(w) // 2)])

            # Hesitation micro-pause after false start (120 ms)
            fs_gap = 0.12
            hesitant_chunks.append(np.zeros(int(fs_gap * sr), dtype=np.float32))
            current_time += fs_gap

        elif dice < 0.65 and idx > 0:
            # Injection 3: Filler (e.g. "ಅಂದರೆ" or "ಹ್ಮ್")
            filler_text = rng.choice(["ಅಂದರೆ", "ಹ್ಮ್...", "ಆ..."])
            filler_dur = 0.35
            filler_unit = generate_kannada_speech_unit(filler_text, duration_sec=filler_dur, sr=sr, f0=145.0)

            fil_start = current_time
            hesitant_chunks.append(filler_unit)
            current_time += filler_dur
            labels.append(
                InjectedDisfluency(
                    type="filler",
                    word_index=idx,
                    text=filler_text,
                    start_sec=fil_start,
                    end_sec=current_time,
                    duration_sec=filler_dur,
                )
            )
            hesitant_tokens.append(filler_text)

            fil_gap = 0.08
            hesitant_chunks.append(np.zeros(int(fil_gap * sr), dtype=np.float32))
            current_time += fil_gap

        # Add the intended clean word
        hesitant_chunks.append(unit)
        hesitant_tokens.append(w)
        current_time += dur

        # Inter-word pause injection (disfluent pause between words: 300–600 ms)
        if idx < len(words) - 1:
            if rng.random() < 0.35:
                # Disfluent long pause (> 250 ms, e.g. 450 ms)
                long_pause_dur = rng.uniform(0.35, 0.65)
                hesitant_chunks.append(np.zeros(int(long_pause_dur * sr), dtype=np.float32))
                labels.append(
                    InjectedDisfluency(
                        type="disfluent_pause",
                        word_index=idx,
                        text="[pause]",
                        start_sec=current_time,
                        end_sec=current_time + long_pause_dur,
                        duration_sec=long_pause_dur,
                    )
                )
                current_time += long_pause_dur
            else:
                # Normal fluent inter-word gap (90 ms)
                norm_gap = 0.09
                hesitant_chunks.append(np.zeros(int(norm_gap * sr), dtype=np.float32))
                current_time += norm_gap

    # Add trailing silence
    trail_dur = rng.choice([0.15, 0.50])
    hesitant_chunks.append(np.zeros(int(trail_dur * sr), dtype=np.float32))
    if trail_dur > 0.25:
        labels.append(
            InjectedDisfluency(
                type="disfluent_pause",
                word_index=len(words) - 1,
                text="[trailing_silence]",
                start_sec=current_time + 0.10,
                end_sec=current_time + trail_dur,
                duration_sec=trail_dur - 0.10,
            )
        )

    hesitant_audio = np.concatenate(hesitant_chunks).astype(np.float32)
    clean_text = unicodedata.normalize("NFC", " ".join(words))
    hesitant_text = unicodedata.normalize("NFC", " ".join(hesitant_tokens))

    return hesitant_audio, clean_text, hesitant_text, labels


# Benchmark Kannada sentences for synthetic evaluation
KANNADA_BENCHMARK_CORPUS: List[List[str]] = [
    ["ನಮಸ್ಕಾರ", "ಇಂದು", "ಬೆಂಗಳೂರಿನಲ್ಲಿ", "ಹವಾಮಾನ", "ಬಹಳ", "ಉತ್ತಮವಾಗಿದೆ"],
    ["ನಾನು", "ಕನ್ನಡ", "ಭಾಷೆಯನ್ನು", "ಪ್ರೀತಿಸುತ್ತೇನೆ", "ಮತ್ತು", "ಕಲಿಯುತ್ತಿದ್ದೇನೆ"],
    ["ಕರ್ನಾಟಕದ", "ಇತಿಹಾಸ", "ಮತ್ತು", "ಸಂಸ್ಕೃತಿ", "ತುಂಬಾ", "ಶ್ರೀಮಂತವಾಗಿದೆ"],
    ["ಮುಂದಿನ", "ವಾರ", "ನಾವು", "ಮೈಸೂರು", "ಅರಮನೆಗೆ", "ಪ್ರವಾಸ", "ಹೋಗುತ್ತಿದ್ದೇವೆ"],
    ["ಕೃತಕ", "ಬುದ್ಧಿಮತ್ತೆ", "ತಂತ್ರಜ್ಞಾನವು", "ದಿನದಿಂದ", "ದಿನಕ್ಕೆ", "ಬೆಳೆಯುತ್ತಿದೆ"],
    ["ಕಾವೇರಿ", "ನದಿ", "ಕರ್ನಾಟಕದ", "ಜೀವನಾಡಿಯಾಗಿದೆ", "ಮತ್ತು", "ಪವಿತ್ರವಾಗಿದೆ"],
    ["ವಿದ್ಯಾರ್ಥಿಗಳು", "ಪರೀಕ್ಷೆಗೆ", "ಶ್ರದ್ಧೆಯಿಂದ", "ಸಿದ್ಧತೆ", "ಮಾಡಿಕೊಳ್ಳುತ್ತಿದ್ದಾರೆ"],
    ["ಕನ್ನಡ", "ಸಾಹಿತ್ಯಕ್ಕೆ", "ಎಂಟು", "ಜ್ಞಾನಪೀಠ", "ಪ್ರಶಸ್ತಿಗಳು", "ಲಭಿಸಿವೆ"],
    ["ನಮ್ಮ", "ಊರಿನಲ್ಲಿ", "ಹಬ್ಬದ", "ಸಂಭ್ರಮ", "ಮನೆಮಾಡಿದೆ"],
    ["ಆರೋಗ್ಯವೇ", "ಭಾಗ್ಯ", "ಎಂಬುದು", "ಹಿರಿಯರ", "ಗಾದೆ", "ಮಾತಾಗಿದೆ"],
]


def generate_evaluation_dataset(
    output_dir: Path,
    num_clips: int = 50,
    sr: int = 16000,
) -> List[Dict[str, Any]]:
    """Generate and persist evaluation audio clips with ground-truth disfluency labels.

    Args:
        output_dir: Directory to store generated datasets.
        num_clips: Number of clips to generate.
        sr: Audio sample rate.

    Returns:
        List of dataset entry summaries.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    for clip_idx in range(num_clips):
        sentence_words = KANNADA_BENCHMARK_CORPUS[clip_idx % len(KANNADA_BENCHMARK_CORPUS)]
        # Add slight variations for higher indices
        if clip_idx >= len(KANNADA_BENCHMARK_CORPUS):
            sentence_words = sentence_words.copy()
            random.Random(clip_idx).shuffle(sentence_words)

        clean_audio, word_meta = create_clean_kannada_sentence(sentence_words, sr=sr)
        hesitant_audio, clean_text, hesitant_text, labels = inject_disfluencies_into_sentence(
            sentence_words, clean_audio, word_meta, sr=sr, rng_seed=1000 + clip_idx
        )

        clip_id = f"clip_{clip_idx:03d}"
        clean_path = output_dir / f"{clip_id}_clean.wav"
        hesitant_path = output_dir / f"{clip_id}_hesitant.wav"
        labels_path = output_dir / f"{clip_id}_labels.json"

        sf.write(str(clean_path), clean_audio, sr)
        sf.write(str(hesitant_path), hesitant_audio, sr)

        labels_data = {
            "clip_id": clip_id,
            "clean_text": clean_text,
            "hesitant_text": hesitant_text,
            "clean_duration": round(len(clean_audio) / float(sr), 3),
            "hesitant_duration": round(len(hesitant_audio) / float(sr), 3),
            "disfluencies": [l.to_dict() for l in labels],
        }

        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, ensure_ascii=False, indent=2)

        manifest.append(labels_data)

    logger.info("Generated %d synthetic evaluation clips in %s", len(manifest), output_dir)
    return manifest
