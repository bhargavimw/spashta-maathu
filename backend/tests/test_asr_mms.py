"""Tests for Stage 3: MMS-1B ASR + forced alignment (verbatim transcript + word timestamps).

Uses mocks so no model weights are downloaded and no GPU is required.
Verifies both free-ASR mode and script mode, plus segment start-offset
application and multi-segment handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchaudio

from backend.pipeline.asr import (
    FRAME_STRIDE_SAMPLES,
    reset_asr_model,
    transcribe_vad_segments,
)
from backend.pipeline.vad import SpeechSegment, VADResult

SR = 16000


def _make_audio_with_segments(
    seg_specs: list[tuple[float, float]], total_sec: float
) -> tuple[np.ndarray, VADResult]:
    """Build synthetic audio with sine waves in speech segments, silence elsewhere."""
    n_total = int(total_sec * SR)
    audio = np.zeros(n_total, dtype=np.float32)
    segs: list[SpeechSegment] = []
    for i, (start_sec, end_sec) in enumerate(seg_specs):
        s, e = int(start_sec * SR), int(end_sec * SR)
        t = np.arange(e - s, dtype=np.float32) / SR
        audio[s:e] = (
            0.4 * np.sin(2 * np.pi * (200 + i * 50) * t)
        ).astype(np.float32)
        segs.append(
            SpeechSegment(
                start_sample=s,
                end_sample=e,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
    vad = VADResult(
        segments=segs,
        gaps=[],
        sample_rate=SR,
        duration_sec=total_sec,
        total_samples=n_total,
    )
    return audio, vad


def _make_mock_processor_model(
    vocab_chars: list[str],
) -> tuple[MagicMock, MagicMock]:
    """Create mock processor/model with a controllable char-level vocab.

    Vocab: <pad>=0 (CTC blank), <unk>=1, |=2 (word delimiter), then chars.
    """
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1, "|": 2}
    for i, ch in enumerate(vocab_chars):
        vocab[ch] = 3 + i

    id_to_token = {v: k for k, v in vocab.items()}
    pad_token_id = vocab["<pad>"]
    word_delimiter_id = vocab["|"]

    processor = MagicMock()
    processor.tokenizer.get_vocab.return_value = vocab
    processor.tokenizer.pad_token_id = pad_token_id
    processor.tokenizer.__getitem__ = MagicMock(
        side_effect=lambda ch: vocab.get(ch, vocab["<unk>"])
    )
    processor.tokenizer.convert_ids_to_tokens.side_effect = (
        lambda tids: [id_to_token.get(int(t), "x") for t in tids]
    )

    def fake_tokenize(text, return_tensors=None):
        ids = []
        for ch in text:
            if ch == " ":
                ids.append(word_delimiter_id)
            else:
                ids.append(vocab.get(ch, vocab["<unk>"]))
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    processor.tokenizer.side_effect = fake_tokenize
    processor.tokenizer.set_target_lang = MagicMock()

    def fake_processor_call(audio_chunk, sampling_rate, return_tensors, padding):
        return {"input_values": torch.tensor([audio_chunk], dtype=torch.float32)}

    processor.side_effect = fake_processor_call

    model = MagicMock()
    model.freeze = MagicMock()
    model.half = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    def fake_forward(input_values):
        audio_len = input_values.shape[-1]
        T = max(1, audio_len // FRAME_STRIDE_SAMPLES)
        logits = torch.zeros(1, T, len(vocab))
        mock_result = MagicMock()
        mock_result.logits = logits
        return mock_result

    model.side_effect = fake_forward

    return processor, model


# ---------------------------------------------------------------------------
# Mode (a): Free ASR
# ---------------------------------------------------------------------------


def test_free_asr_mode_returns_words_with_timestamps():
    """Free ASR: greedy decode + forced_align, returns word timestamps."""
    audio, vad = _make_audio_with_segments([(0.0, 1.0)], total_sec=2.0)

    processor, model = _make_mock_processor_model(["ನ", "ಾ", "ನು"])

    with patch(
        "backend.pipeline.asr.get_asr_model", return_value=(processor, model)
    ):
        result = transcribe_vad_segments(audio, vad, backend="mms")

    assert len(result) == 1
    seg = result[0]
    assert seg["index"] == 0
    assert isinstance(seg["text"], str)
    assert isinstance(seg["words"], list)
    for w in seg["words"]:
        assert "word" in w
        assert "start_s" in w
        assert "end_s" in w
        assert "confidence" in w
        assert w["start_s"] >= 0.0
        assert w["end_s"] >= w["start_s"]


def test_free_asr_multiple_segments_independent_offsets():
    """Each VAD segment processed independently; word timestamps include segment offset."""
    audio, vad = _make_audio_with_segments(
        [(0.0, 1.0), (2.5, 3.5)], total_sec=5.0
    )

    processor, model = _make_mock_processor_model(["ನ", "ಾ", "ನು", " ", "ಹ"])

    with patch(
        "backend.pipeline.asr.get_asr_model", return_value=(processor, model)
    ):
        result = transcribe_vad_segments(audio, vad, backend="mms")

    assert len(result) == 2
    assert result[0]["index"] == 0
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 1.0
    assert result[1]["index"] == 1
    assert result[1]["start"] == 2.5
    assert result[1]["end"] == 3.5

    # Word timestamps in segment 1 should be offset by 2.5s
    for w in result[1]["words"]:
        assert w["start_s"] >= 2.5
        assert w["end_s"] >= 2.5


def test_free_asr_gap_after_computed():
    """gap_after is time to next segment start; 0.0 for last segment."""
    audio, vad = _make_audio_with_segments(
        [(0.0, 1.0), (1.5, 2.5)], total_sec=3.0
    )

    processor, model = _make_mock_processor_model(["ನ", "ಾ", "ನು"])

    with patch(
        "backend.pipeline.asr.get_asr_model", return_value=(processor, model)
    ):
        result = transcribe_vad_segments(audio, vad, backend="mms")

    assert result[0]["gap_after"] == pytest.approx(0.5, abs=1e-5)  # 1.5 - 1.0
    assert result[1]["gap_after"] == 0.0  # last segment


# ---------------------------------------------------------------------------
# Mode (b): Script mode
# ---------------------------------------------------------------------------


def test_script_mode_uses_given_script_verbatim():
    """Script mode: forced_align uses provided script, not greedy decode."""
    audio, vad = _make_audio_with_segments([(0.0, 1.0)], total_sec=2.0)

    processor, model = _make_mock_processor_model(["ನ", "ಾ", "ನು", " ", "ಹ"])

    with patch(
        "backend.pipeline.asr.get_asr_model", return_value=(processor, model)
    ):
        result = transcribe_vad_segments(
            audio, vad, backend="mms", script_texts=["ನಾನು ಹ"]
        )

    assert len(result) == 1
    seg = result[0]
    assert seg["text"] == "ನಾನು ಹ"  # script used verbatim
    assert isinstance(seg["words"], list)


def test_script_mode_longer_than_audio_segment():
    """Script mode works when script spans the whole segment."""
    audio, vad = _make_audio_with_segments([(0.0, 2.0)], total_sec=3.0)

    processor, model = _make_mock_processor_model(
        ["ನ", "ಾ", "ನು", " ", "ಹ", "'oiseau"]  # 'o' will be UNK
    )

    with patch(
        "backend.pipeline.asr.get_asr_model", return_value=(processor, model)
    ):
        result = transcribe_vad_segments(
            audio, vad, backend="mms", script_texts=["ನಾನು"]
        )

    assert len(result) == 1
    assert result[0]["text"] == "ನಾನು"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_vad_returns_empty_list():
    """VAD with no segments returns empty list."""
    n = int(2.0 * SR)
    silence = np.zeros(n, dtype=np.float32)
    vad = VADResult(
        segments=[], gaps=[], sample_rate=SR, duration_sec=2.0, total_samples=n
    )

    processor, model = _make_mock_processor_model(["ನ"])

    with patch(
        "backend.pipeline.asr.get_asr_model", return_value=(processor, model)
    ):
        result = transcribe_vad_segments(audio=silence, vad_result=vad, backend="mms")

    assert result == []


def test_audio_validation_errors():
    """Invalid audio raises ValueError."""
    vad = VADResult(
        segments=[], gaps=[], sample_rate=SR, duration_sec=1.0, total_samples=SR
    )

    # Non-1D audio
    stereo = np.zeros((2, SR), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        transcribe_vad_segments(stereo, vad, backend="mms")

    # Wrong dtype
    int16 = np.zeros(SR, dtype=np.int16)
    with pytest.raises(ValueError, match="float32"):
        transcribe_vad_segments(int16, vad, backend="mms")

    # Length mismatch
    audio_wrong = np.zeros(100, dtype=np.float32)
    with pytest.raises(ValueError, match="total_samples"):
        transcribe_vad_segments(audio_wrong, vad, backend="mms")


def test_reset_asr_model_clears_cache():
    """reset_asr_model evicts cached processor and model."""
    import backend.pipeline.asr as asr_module

    asr_module._CACHED_PROCESSOR = object()
    asr_module._CACHED_MODEL = object()
    asr_module._CACHED_BACKEND = "mms"
    assert asr_module._CACHED_PROCESSOR is not None

    reset_asr_model()

    assert asr_module._CACHED_PROCESSOR is None
    assert asr_module._CACHED_MODEL is None
    assert asr_module._CACHED_BACKEND is None


def test_frame_stride_is_320_samples():
    """FRAME_STRIDE_SAMPLES is 320 (20 ms at 16 kHz)."""
    assert FRAME_STRIDE_SAMPLES == 320


def test_forced_align_available():
    """torchaudio.functional.forced_align and merge_tokens exist (SPEC.md §3)."""
    assert hasattr(torchaudio.functional, "forced_align")
    assert hasattr(torchaudio.functional, "merge_tokens")