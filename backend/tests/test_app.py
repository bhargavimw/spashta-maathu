"""FastAPI endpoint tests using TestClient.

Updated for Module 6: POST /process now runs process_full() (full pipeline),
and GET /health reports component readiness via the 'components' key.
"""

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def _generate_wav_bytes(duration_sec: float = 4.0, sr: int = 16000) -> bytes:
    """Generate in-memory WAV file bytes with speech bursts and silence."""
    n_speech = int(1.5 * sr)
    t = np.linspace(0, 1.5, n_speech, endpoint=False, dtype=np.float32)
    burst = 0.5 * np.sin(2 * np.pi * 200.0 * t).astype(np.float32)

    # 0.5s silence + 1.5s speech + 1.5s pause + 0.5s silence = 4.0s
    audio = np.concatenate([
        np.zeros(int(0.5 * sr), dtype=np.float32),
        burst,
        np.zeros(int(1.5 * sr), dtype=np.float32),
        np.zeros(int(0.5 * sr), dtype=np.float32),
    ])

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _make_process_full_result(tmp_path: Path, sr: int = 16000):
    """Build a mock Result with a real WAV on disk."""
    from backend.pipeline.orchestrator import Result
    from backend.app import PROCESSED_DIR

    # Write a real clean WAV so the /audio/ endpoint can serve it
    clean_wav = PROCESSED_DIR / "clean_testapplegacy.wav"
    sf.write(str(clean_wav), np.zeros(sr, dtype=np.float32), sr)

    return Result(
        clean_audio_path=str(clean_wav),
        original_duration=4.0,
        clean_duration=2.5,
        removed_regions=[{"start": 2.5, "end": 4.0, "reason": "disfluent_pause"}],
        verbatim_text="ನಾನು ಹೋದೆ",
        clean_text="ನಾನು ಹೋದೆ",
        diff=[{"index": 0, "text": "ನಾನು ಹೋದೆ", "action": "keep"}],
        words=None,
    ), clean_wav


def test_get_health() -> None:
    """GET /health reports status and component readiness (Module 6 schema)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data

    # Module 6: components key with vad, asr, classifier readiness
    components = data["components"]
    assert "vad" in components
    assert "asr" in components
    assert "classifier" in components

    # Each value must be a boolean
    for key in ("vad", "asr", "classifier"):
        assert isinstance(components[key], bool)


def test_post_process_audio_upload(tmp_path) -> None:
    """POST /process uploads audio and returns valid JSON with all Module 6 fields."""
    result, clean_wav = _make_process_full_result(tmp_path)

    try:
        wav_bytes = _generate_wav_bytes(duration_sec=4.0)
        files = {"file": ("test_clip.wav", wav_bytes, "audio/wav")}

        with patch("backend.app.process_full", return_value=result):
            response = client.post("/process", files=files)

        assert response.status_code == 200
        data = response.json()

        # Validate Result contract
        assert "clean_audio_url" in data
        assert "clean_audio_path" in data
        assert data["clean_audio_url"].startswith("/audio/")
        assert data["original_duration"] == pytest.approx(4.0, abs=0.05)
        assert data["clean_duration"] < data["original_duration"]
        assert data["time_saved"] > 0

        # Removed regions
        assert isinstance(data["removed_regions"], list)
        assert len(data["removed_regions"]) >= 1
        for region in data["removed_regions"]:
            assert "start" in region
            assert "end" in region
            assert "reason" in region

        # Module 6 transcript fields (populated by process_full)
        assert data["verbatim_text"] is not None
        assert data["clean_text"] is not None

    finally:
        if clean_wav.exists():
            clean_wav.unlink(missing_ok=True)


def test_get_audio_not_found() -> None:
    """GET /audio/{nonexistent} returns 404."""
    response = client.get("/audio/nonexistent_audio_file.wav")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_root_serves_frontend() -> None:
    """GET / serves the built React single-page application."""
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text or "Kannada" in response.text
