"""FastAPI endpoint tests using TestClient."""

import io
from pathlib import Path

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


def test_get_health() -> None:
    """GET /health reports status and stage availability."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data

    stages = data["stages"]
    # v1 stages are available
    assert stages["ingest"] is True
    assert stages["vad"] is True
    assert stages["gap_rules"] is True
    assert stages["splicer"] is True

    # v2/v3 stages are deferred and report False
    assert stages["asr"] is False
    assert stages["align"] is False
    assert stages["classifier"] is False


def test_post_process_audio_upload() -> None:
    """POST /process uploads audio, processes via v1, and returns valid JSON and URLs."""
    wav_bytes = _generate_wav_bytes(duration_sec=4.0)

    files = {"file": ("test_clip.wav", wav_bytes, "audio/wav")}
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
        assert region["reason"] == "disfluent_pause"

    # Missing stages MUST be None (null in JSON)
    assert data["words"] is None
    assert data["verbatim_text"] is None
    assert data["clean_text"] is None
    assert data["diff"] is None

    # Retrieve processed audio from clean_audio_url
    audio_resp = client.get(data["clean_audio_url"])
    assert audio_resp.status_code == 200
    assert audio_resp.headers["content-type"] == "audio/wav"
    assert len(audio_resp.content) > 0


def test_get_audio_not_found() -> None:
    """GET /audio/{nonexistent} returns 404."""
    response = client.get("/audio/nonexistent_audio_file.wav")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
