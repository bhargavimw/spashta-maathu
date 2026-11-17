
"""Pipeline stages for audio processing, VAD, ASR, alignment, classification, and splicing."""

__all__ = ["process", "Result", "RemovedRegion"]


def __getattr__(name):
    """Load orchestrator exports only when they are requested."""
    if name in __all__:
        from backend.pipeline import orchestrator
        return getattr(orchestrator, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
