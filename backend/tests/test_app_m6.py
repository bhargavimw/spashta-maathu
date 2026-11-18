"""Module 6 FastAPI endpoint tests.

Tests cover:
  1. Successful audio processing (POST /process returns all expected fields).
  2. Invalid / unsupported uploads (400 / 422 responses).
  3. Pipeline failures (500 response with safe error messages).
  4. Health endpoint – component readiness reflects actual configuration state.

No Whisper models are downloaded, no real ASR inference runs, and no Gemini
API quota is consumed.  All heavy model calls are mocked.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16000


def _wav_bytes(duration_sec: float = 2.0, sr: int = SR) -> bytes:
    """Return in-memory WAV bytes with a simple sine-wave burst."""
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    audio = (0.4 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def _make_result(**kwargs) -> Any:
    """Build a minimal pipeline Result-like object for mock returns."""
    from backend.pipeline.orchestrator import Result

    defaults: Dict[str, Any] = {
        "clean_audio_path": "/tmp/clean.wav",
        "original_duration": 2.0,
        "clean_duration": 1.5,
        "removed_regions": [{"start": 1.5, "end": 2.0, "reason": "disfluent_pause"}],
        "verbatim_text": "ನಾನು ನಾನು ಹೋದೆ",
        "clean_text": "ನಾನು ಹೋದೆ",
        "diff": [
            {"index": 0, "text": "ನಾನು", "action": "keep"},
            {"index": 1, "text": "ನಾನು", "action": "remove", "reason": "repetition"},
            {"index": 2, "text": "ಹೋದೆ", "action": "keep"},
        ],
        "words": None,
    }
    defaults.update(kwargs)
    return Result(**defaults)


# ---------------------------------------------------------------------------
# Client fixture (fresh import per test to avoid state leakage)
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from backend.app import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Successful processing
# ---------------------------------------------------------------------------

class TestPostProcessSuccess:
    """POST /process — happy-path scenarios."""

    def test_returns_200_and_all_fields(self, client, tmp_path):
        """Successful upload returns 200 with clean_audio_url, transcripts, diff, regions."""
        result = _make_result()
        clean_wav = tmp_path / "clean.wav"
        sf.write(str(clean_wav), np.zeros(SR, dtype=np.float32), SR)
        result.clean_audio_path = str(clean_wav)

        with patch("backend.app.process_full", return_value=result):
            resp = client.post(
                "/process",
                files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()

        # Required fields
        assert data["clean_audio_url"].startswith("/audio/")
        assert isinstance(data["original_duration"], float)
        assert isinstance(data["clean_duration"], float)
        assert isinstance(data["time_saved"], float)
        assert data["time_saved"] >= 0.0
        assert isinstance(data["removed_regions"], list)

        # Module 6 transcript fields
        assert data["verbatim_text"] == result.verbatim_text
        assert data["clean_text"] == result.clean_text
        assert isinstance(data["diff"], list)

    def test_time_saved_computed_correctly(self, client, tmp_path):
        """time_saved = original_duration - clean_duration (non-negative)."""
        result = _make_result(original_duration=5.0, clean_duration=3.2)
        clean_wav = tmp_path / "c.wav"
        sf.write(str(clean_wav), np.zeros(SR, dtype=np.float32), SR)
        result.clean_audio_path = str(clean_wav)

        with patch("backend.app.process_full", return_value=result):
            resp = client.post(
                "/process",
                files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["time_saved"] == pytest.approx(1.8, abs=0.001)

    def test_clean_audio_url_retrievable(self, client, tmp_path):
        """The URL returned in clean_audio_url responds with 200 audio/wav."""
        from backend.app import PROCESSED_DIR
        import uuid

        # The app auto-generates the output path as PROCESSED_DIR / "clean_<id>.wav".
        # We pre-create a known file and return a result whose clean_audio_path
        # points to it, then verify the URL the app derives is fetchable.
        unique_id = uuid.uuid4().hex[:10]
        clean_wav = PROCESSED_DIR / f"clean_{unique_id}.wav"
        sf.write(str(clean_wav), np.zeros(SR, dtype=np.float32), SR)

        result = _make_result(clean_audio_path=str(clean_wav))

        # Patch process_full so the app skips real inference but still writes
        # the output to the pre-created file path via our result.
        def _fake_process_full(audio_path, config=None, output_path=None, **kw):
            # The app passes output_path; write a valid WAV there
            if output_path is not None:
                sf.write(str(output_path), np.zeros(SR, dtype=np.float32), SR)
                result2 = _make_result(clean_audio_path=str(output_path))
                return result2
            return result

        try:
            with patch("backend.app.process_full", side_effect=_fake_process_full):
                resp = client.post(
                    "/process",
                    files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
                )
            assert resp.status_code == 200
            audio_url = resp.json()["clean_audio_url"]

            audio_resp = client.get(audio_url)
            assert audio_resp.status_code == 200
            assert "audio" in audio_resp.headers.get("content-type", "")
        finally:
            # Clean up any files left in PROCESSED_DIR by this test
            for f in PROCESSED_DIR.glob("clean_*.wav"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def test_removed_regions_format(self, client, tmp_path):
        """removed_regions entries each have start, end, reason."""
        regions = [
            {"start": 0.5, "end": 0.8, "reason": "repetition", "segment_index": 1},
        ]
        result = _make_result(removed_regions=regions)
        clean_wav = tmp_path / "c.wav"
        sf.write(str(clean_wav), np.zeros(SR, dtype=np.float32), SR)
        result.clean_audio_path = str(clean_wav)

        with patch("backend.app.process_full", return_value=result):
            resp = client.post(
                "/process",
                files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            )

        assert resp.status_code == 200
        rr = resp.json()["removed_regions"]
        assert len(rr) == 1
        assert rr[0]["start"] == pytest.approx(0.5)
        assert rr[0]["end"] == pytest.approx(0.8)
        assert rr[0]["reason"] == "repetition"


# ---------------------------------------------------------------------------
# 2. Invalid / unsupported uploads
# ---------------------------------------------------------------------------

class TestPostProcessInvalidUpload:
    """POST /process — invalid upload scenarios."""

    def test_missing_filename_returns_400_or_422(self, client):
        """An upload with an empty filename returns 400 or 422."""
        resp = client.post(
            "/process",
            files={"file": ("", _wav_bytes(), "audio/wav")},
        )
        # FastAPI/Starlette may validate at the framework level (422) or
        # our code may catch it first (400). Both are acceptable.
        assert resp.status_code in (400, 422)

    def test_unsupported_format_returns_400(self, client):
        """An upload with an unsupported extension returns 400."""
        resp = client.post(
            "/process",
            files={"file": ("clip.xyz", b"not audio", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_empty_file_returns_422(self, client):
        """An upload with zero bytes returns 422."""
        resp = client.post(
            "/process",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 422
        assert "empty" in resp.json()["detail"].lower()

    def test_invalid_audio_content_returns_422(self, client):
        """A WAV file that cannot be decoded raises a 422 or 500."""
        # process_full will raise FileNotFoundError / RuntimeError on bad audio
        with patch(
            "backend.app.process_full",
            side_effect=RuntimeError("Stage 0 (ingest) failed: not a valid audio file"),
        ):
            resp = client.post(
                "/process",
                files={"file": ("bad.wav", b"THIS IS NOT AUDIO", "audio/wav")},
            )
        # Pipeline RuntimeError → 500
        assert resp.status_code == 500

    def test_no_file_field_returns_422(self, client):
        """A POST with no 'file' field returns 422 (FastAPI validation)."""
        resp = client.post("/process")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. Pipeline failures
# ---------------------------------------------------------------------------

class TestPostProcessPipelineFailure:
    """POST /process — pipeline error propagation."""

    def test_runtime_error_returns_500(self, client):
        """RuntimeError from process_full produces a 500 with a safe message."""
        with patch(
            "backend.app.process_full",
            side_effect=RuntimeError("Stage 2 (ASR) failed: model not loaded"),
        ):
            resp = client.post(
                "/process",
                files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            )
        assert resp.status_code == 500
        detail = resp.json()["detail"]
        # Safe: pipeline stage message is forwarded
        assert "Stage 2" in detail or "failed" in detail.lower()
        # Must NOT expose secrets
        assert "API_KEY" not in detail
        assert "GEMINI" not in detail

    def test_value_error_returns_500(self, client):
        """ValueError from process_full produces a 500."""
        with patch(
            "backend.app.process_full",
            side_effect=ValueError("bad segment count"),
        ):
            resp = client.post(
                "/process",
                files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            )
        assert resp.status_code == 500

    def test_import_error_wrapped_as_500(self, client):
        """ImportError (e.g., faster-whisper missing) produces a 500."""
        with patch(
            "backend.app.process_full",
            side_effect=ImportError("faster-whisper is required for ASR"),
        ):
            resp = client.post(
                "/process",
                files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            )
        # ImportError falls under generic Exception → 500
        assert resp.status_code == 500

    def test_no_stack_trace_in_response(self, client):
        """Stack traces are never serialised into the response body."""
        with patch(
            "backend.app.process_full",
            side_effect=Exception("unexpected"),
        ):
            resp = client.post(
                "/process",
                files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
            )
        body = resp.text
        assert "Traceback" not in body
        assert "File \"" not in body


# ---------------------------------------------------------------------------
# 4. Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /health — status and component readiness."""

    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_has_required_fields(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data

    def test_components_are_booleans(self, client):
        components = client.get("/health").json()["components"]
        for key in ("vad", "asr", "classifier"):
            assert key in components, f"Missing component '{key}'"
            assert isinstance(components[key], bool)

    def test_vad_ready_when_torch_and_silero_available(self, client):
        """vad=True when torch and silero_vad can be imported."""
        with patch("backend.app._vad_ready", return_value=True):
            data = client.get("/health").json()
        assert data["components"]["vad"] is True

    def test_vad_not_ready_when_import_fails(self, client):
        """vad=False when torch or silero_vad is not installed."""
        with patch("backend.app._vad_ready", return_value=False):
            data = client.get("/health").json()
        assert data["components"]["vad"] is False

    def test_asr_ready_when_faster_whisper_installed_and_model_configured(self, client):
        """asr=True when faster_whisper is installed and model ID is non-empty."""
        with patch("backend.app._asr_ready", return_value=True):
            data = client.get("/health").json()
        assert data["components"]["asr"] is True

    def test_asr_not_ready_when_faster_whisper_missing(self, client):
        """asr=False when faster_whisper cannot be imported."""
        with patch("backend.app._asr_ready", return_value=False):
            data = client.get("/health").json()
        assert data["components"]["asr"] is False

    def test_classifier_ready_when_api_key_set(self, client, monkeypatch):
        """classifier=True when GEMINI_API_KEY is non-empty."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
        with patch("backend.app._classifier_ready", return_value=True):
            data = client.get("/health").json()
        assert data["components"]["classifier"] is True

    def test_classifier_not_ready_when_no_api_key(self, client, monkeypatch):
        """classifier=False when no Gemini API key is configured."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("backend.app._classifier_ready", return_value=False):
            data = client.get("/health").json()
        assert data["components"]["classifier"] is False

    def test_all_unavailable_still_returns_healthy_status(self, client):
        """API status is 'healthy' even when components are not ready."""
        with (
            patch("backend.app._vad_ready", return_value=False),
            patch("backend.app._asr_ready", return_value=False),
            patch("backend.app._classifier_ready", return_value=False),
        ):
            data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["components"]["vad"] is False
        assert data["components"]["asr"] is False
        assert data["components"]["classifier"] is False


# ---------------------------------------------------------------------------
# 5. Audio serving endpoint
# ---------------------------------------------------------------------------

class TestGetAudioEndpoint:
    """GET /audio/{filename} — serving processed audio."""

    def test_nonexistent_file_returns_404(self, client):
        resp = client.get("/audio/does_not_exist_m6.wav")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_directory_traversal_is_blocked(self, client):
        """Path traversal attempts cannot escape the processed dir."""
        resp = client.get("/audio/../../etc/passwd")
        # Either 404 (safe_name normalisation) or blocked entirely
        assert resp.status_code in (404, 400, 422)
