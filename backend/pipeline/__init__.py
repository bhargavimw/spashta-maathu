"""Pipeline stages for audio processing, VAD, ASR, alignment, classification, and splicing."""

from backend.pipeline.orchestrator import RemovedRegion, Result, process

__all__ = ["process", "Result", "RemovedRegion"]
