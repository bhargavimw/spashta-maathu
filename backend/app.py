"""FastAPI Application for Kannada Pause Remover & Stammer Cleaner.

Provides REST API endpoints for:
- GET /health: Health check and pipeline stage availability report.
- POST /process: Upload audio file and run processing pipeline.
- GET /audio/{filename}: Serve processed clean audio files.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import config, setup_logging
from backend.pipeline import Result, process

setup_logging()
logger = logging.getLogger("kannada_cleaner.app")

app = FastAPI(
    title="Kannada Pause Remover & Stammer Cleaner API",
    description="REST API for processing hesitant Kannada speech and removing unnatural pauses.",
    version="1.0.0",
)

# Storage directory for processed audio
PROCESSED_DIR = config.paths.data_dir / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class HealthResponse(BaseModel):
    """Health status and stage availability response schema."""

    status: str
    stages: Dict[str, bool]
    version: str


class ProcessResponse(BaseModel):
    """Processing response schema."""

    clean_audio_url: str
    clean_audio_path: str
    original_duration: float
    clean_duration: float
    time_saved: float
    removed_regions: List[Dict[str, Any]]
    words: Optional[List[Dict[str, Any]]] = None
    verbatim_text: Optional[str] = None
    clean_text: Optional[str] = None
    diff: Optional[Any] = None


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report API health and available pipeline stages (plan.md §7 M8).

    Stages implemented in v1: ingest, vad, gap_rules, splicer.
    Stages deferred to v2 (M3-M5): asr, align, classifier.
    """
    return HealthResponse(
        status="healthy",
        stages={
            "ingest": True,
            "vad": True,
            "gap_rules": True,
            "splicer": True,
            "asr": False,
            "align": False,
            "classifier": False,
        },
        version="1.0.0",
    )


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to clean (WAV, MP3, FLAC, etc.)")
) -> ProcessResponse:
    """Upload and process audio through the pipeline (plan.md §7 M8).

    Runs v1 pipeline: Ingest -> VAD -> Gap Rules -> Splicer.
    Returns cleaned audio metadata, URLs, and removed regions.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    file_suffix = Path(file.filename).suffix or ".wav"
    unique_id = uuid.uuid4().hex[:10]
    temp_input = PROCESSED_DIR / f"upload_{unique_id}{file_suffix}"
    clean_output = PROCESSED_DIR / f"clean_{unique_id}.wav"

    try:
        # Save uploaded file
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run pipeline
        result: Result = process(
            audio_path=temp_input,
            config=config,
            output_path=clean_output,
        )

        time_saved = max(0.0, result.original_duration - result.clean_duration)
        clean_audio_filename = clean_output.name
        audio_url = f"/audio/{clean_audio_filename}"

        return ProcessResponse(
            clean_audio_url=audio_url,
            clean_audio_path=str(clean_output),
            original_duration=round(result.original_duration, 3),
            clean_duration=round(result.clean_duration, 3),
            time_saved=round(time_saved, 3),
            removed_regions=result.removed_regions,
            words=result.words,
            verbatim_text=result.verbatim_text,
            clean_text=result.clean_text,
            diff=result.diff,
        )

    except Exception as exc:
        logger.exception("Error processing audio upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(exc)}",
        ) from exc
    finally:
        # Clean up temporary uploaded raw file
        if temp_input.exists():
            try:
                temp_input.unlink()
            except OSError:
                pass


@app.get("/audio/{filename}")
def get_audio_file(filename: str) -> FileResponse:
    """Serve a processed audio file by filename."""
    # Sanitize filename against directory traversal
    safe_name = Path(filename).name
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


# Serve built React frontend if available
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

