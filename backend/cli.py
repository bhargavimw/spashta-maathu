"""Command-line interface (CLI) for Kannada Pause Remover & Stammer Cleaner.

Provides:
- v1: Pause removal via VAD, zero-crossing snapping, gap-capping, and audio splicing.
- v2: (Future milestone) Full stammer cleaner with ASR, alignment, and LLM classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from backend.config import config, setup_logging
from backend.pipeline.ingest import ingest_audio
from backend.pipeline.splicer import detect_join_clicks, splice_audio
from backend.pipeline.vad import detect_speech_regions


def run_pause_remover(
    input_path: str | Path,
    output_path: str | Path,
    vad_threshold: float | None = None,
    crossfade_ms: float | None = None,
    snap_window_ms: float | None = None,
) -> int:
    """Execute v1 Pause Remover pipeline on an audio file.

    Args:
        input_path: Path to input audio file.
        output_path: Destination path for cleaned audio file.
        vad_threshold: Silero VAD speech detection threshold.
        crossfade_ms: Equal-power crossfade length in milliseconds.
        snap_window_ms: Window for snapping cut points to zero-crossings.

    Returns:
        0 on success, non-zero on failure.
    """
    logger = setup_logging()
    inp = Path(input_path).resolve()
    out = Path(output_path).resolve()

    if not inp.is_file():
        logger.error("Input file does not exist: %s", inp)
        print(f"Error: Input file does not exist: {inp}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Processing audio file: {inp}")

    # Stage 0: Ingest
    audio = ingest_audio(file_path=inp)
    sr = config.audio.sample_rate

    # Stage 1: VAD
    threshold = vad_threshold if vad_threshold is not None else config.models.vad_threshold
    vad_result = detect_speech_regions(audio, sample_rate=sr, threshold=threshold)

    # Stage 5A: Splicer & Pause Policy
    result = splice_audio(
        audio=audio,
        vad_result=vad_result,
        crossfade_ms=crossfade_ms,
        snap_window_ms=snap_window_ms,
    )

    # Automated click check
    clicks = detect_join_clicks(result.audio, result.join_indices)

    # Save output
    sf.write(str(out), result.audio, sr)

    pct_reduced = (
        (result.time_saved_sec / result.original_duration_sec * 100.0)
        if result.original_duration_sec > 0
        else 0.0
    )

    print("\n--- v1 Pause Remover Summary ---")
    print(f"Input file:           {inp.name}")
    print(f"Output file:          {out}")
    print(f"Original duration:    {result.original_duration_sec:.2f} s")
    print(f"Cleaned duration:     {result.output_duration_sec:.2f} s")
    print(f"Time saved:           {result.time_saved_sec:.2f} s ({pct_reduced:.1f}%)")
    print(f"Speech segments:      {len(vad_result.segments)}")
    print(f"Pauses capped:        {result.pauses_capped}")
    print(f"Join clicks detected: {len(clicks)} (clean join validation)")
    print("--------------------------------\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for kannada-pause-remover."""
    parser = argparse.ArgumentParser(
        prog="kannada-cleaner",
        description="Kannada Speech Disfluency & Pause Remover",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: remove-pauses (v1)
    p_remove = subparsers.add_parser(
        "remove-pauses",
        help="v1: Remove long pauses and silence using VAD and deterministic audio splicing",
    )
    p_remove.add_argument(
        "-i", "--input", required=True, type=str, help="Path to input audio file"
    )
    p_remove.add_argument(
        "-o",
        "--output",
        required=False,
        type=str,
        default="clean.wav",
        help="Path to output clean audio file (default: clean.wav)",
    )
    p_remove.add_argument(
        "--vad-threshold",
        type=float,
        default=config.models.vad_threshold,
        help=f"VAD speech detection probability threshold (default: {config.models.vad_threshold})",
    )
    p_remove.add_argument(
        "--crossfade-ms",
        type=float,
        default=config.pause.crossfade_ms,
        help=f"Crossfade duration in ms (default: {config.pause.crossfade_ms})",
    )

    # Allow direct top-level arguments for convenience if no subcommand is passed
    parser.add_argument(
        "-i", "--input", type=str, help="Path to input audio file"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="clean.wav", help="Path to output audio file"
    )

    args = parser.parse_args(argv)

    if args.command == "remove-pauses" or (args.command is None and args.input):
        inp = args.input
        out = args.output
        cf_ms = getattr(args, "crossfade_ms", config.pause.crossfade_ms)
        vad_th = getattr(args, "vad_threshold", config.models.vad_threshold)
        return run_pause_remover(
            input_path=inp,
            output_path=out,
            vad_threshold=vad_th,
            crossfade_ms=cf_ms,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
