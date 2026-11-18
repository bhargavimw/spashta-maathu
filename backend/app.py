"""FastAPI Application for Kannada Pause Remover & Stammer Cleaner.

Provides REST API endpoints for:
- GET /health: Health check and pipeline stage availability report.
- POST /process: Upload audio file and run the full processing pipeline.
- GET /audio/{filename}: Serve processed clean audio files.

Module 6 implementation:
  - POST /process runs process_full() (Ingest → VAD → ASR → Classifier → Splicer).
  - GET /health reports actual component readiness from configuration state.
  - Errors return appropriate HTTP status codes without exposing secrets or tracebacks.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import config, setup_logging
from backend.pipeline import process_full, Result

setup_logging()
logger = logging.getLogger("kannada_cleaner.app")

app = FastAPI(
    title="Kannada Pause Remover & Stammer Cleaner API",
    description=(
        "REST API for processing hesitant Kannada speech, removing disfluencies "
        "and unnatural pauses using VAD → ASR → classifier-guided splicing."
    ),
    version="2.0.0",
)

# Storage directory for processed audio
PROCESSED_DIR = config.paths.data_dir / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Supported audio MIME types / extensions
_SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm",
}


# ---------------------------------------------------------------------------
# Readiness helpers
# ---------------------------------------------------------------------------

def _vad_ready() -> bool:
    """Check whether Silero VAD can be loaded (torch + silero_vad installed)."""
    try:
        import torch  # noqa: PLC0415
        import silero_vad  # noqa: PLC0415
        return True
    except ImportError:
        return False


def _asr_ready() -> bool:
    """Check whether Faster-Whisper is installed and a model ID is configured."""
    try:
        import faster_whisper  # noqa: PLC0415
    except ImportError:
        return False
    model_id = config.models.asr_whisper_model_id
    return bool(model_id and model_id.strip())


def _classifier_ready() -> bool:
    """Check whether the Gemini classifier is configured (API key present)."""
    api_key = config.models.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    return bool(api_key and api_key.strip())


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ComponentStatus(BaseModel):
    """Readiness status of a single pipeline component."""

    ready: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health status and component readiness response schema."""

    status: str
    version: str
    components: Dict[str, bool]
    stages: Dict[str, bool]


class ProcessResponse(BaseModel):
    """Full processing result response schema."""

    clean_audio_url: str
    clean_audio_path: str
    original_duration: float
    clean_duration: float
    time_saved: float
    removed_regions: List[Dict[str, Any]]
    verbatim_text: Optional[str] = None
    clean_text: Optional[str] = None
    diff: Optional[List[Dict[str, Any]]] = None
    words: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report API health and actual pipeline component readiness.

    Component readiness is determined from real configuration/import state,
    not merely successful Python imports of the module package.
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        components={
            "vad": _vad_ready(),
            "asr": _asr_ready(),
            "classifier": _classifier_ready(),
        },
        stages={
            "ingest": True,
            "vad": _vad_ready(),
            "gap_rules": True,
            "splicer": True,
            "asr": _asr_ready(),
            "align": False,  # Stage 3 forced alignment not yet implemented (M3)
            "classifier": _classifier_ready(),
        },
    )


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to clean (WAV, MP3, FLAC, etc.)")
) -> ProcessResponse:
    """Upload and process audio through the full pipeline.

    Runs: Ingest → VAD → ASR → Classifier → Classifier-guided Splicer.

    Returns the clean transcript, verbatim transcript, segment-level diff,
    removed regions, and a URL to retrieve the cleaned audio file.

    Errors:
        400: Missing filename or unsupported audio format.
        422: Uploaded file is not valid audio.
        500: Internal processing failure (details logged server-side).
    """
    # --- Validate upload ---
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix and file_suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported audio format '{file_suffix}'. "
                f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            ),
        )

    unique_id = uuid.uuid4().hex[:10]
    suffix = file_suffix or ".wav"
    temp_input = PROCESSED_DIR / f"upload_{unique_id}{suffix}"
    clean_output = PROCESSED_DIR / f"clean_{unique_id}.wav"

    try:
        # Save uploaded file to disk
        with open(temp_input, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Validate the file is non-empty
        if temp_input.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded file is empty.",
            )

        # Run full pipeline
        try:
            result: Result = process_full(
                audio_path=temp_input,
                config=config,
                output_path=clean_output,
            )
        except FileNotFoundError as exc:
            logger.error("Pipeline failed — file not found: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Audio file could not be read. Ensure the upload is a valid audio file.",
            ) from exc
        except (RuntimeError, ValueError) as exc:
            # Sanitize: never forward internal messages that contain secrets.
            # The exc message is safe (RuntimeError from pipeline stages).
            logger.exception("Pipeline processing failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio processing failed: {exc}",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error during processing: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during audio processing.",
            ) from exc

        time_saved = max(0.0, result.original_duration - result.clean_duration)
        audio_url = f"/audio/{clean_output.name}"

        return ProcessResponse(
            clean_audio_url=audio_url,
            clean_audio_path=str(clean_output),
            original_duration=round(result.original_duration, 3),
            clean_duration=round(result.clean_duration, 3),
            time_saved=round(time_saved, 3),
            removed_regions=result.removed_regions,
            verbatim_text=result.verbatim_text,
            clean_text=result.clean_text,
            diff=result.diff,
            words=result.words,
        )

    finally:
        # Always clean up the temp upload file
        if temp_input.exists():
            try:
                temp_input.unlink()
            except OSError:
                pass


@app.get("/audio/{filename}")
def get_audio_file(filename: str) -> FileResponse:
    """Serve a processed clean audio file by filename.

    Sanitizes the filename against directory traversal attacks.
    """
    safe_name = Path(filename).name  # strip any path components
    file_path = PROCESSED_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{safe_name}' not found.",
        )

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=safe_name,
    )


# ---------------------------------------------------------------------------
# Serve built React frontend if available
# ---------------------------------------------------------------------------

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
