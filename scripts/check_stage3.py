"""Check Stage 3 (ASR + forced alignment) on one WAV file.

Usage:
    python scripts/check_stage3.py <wav_path> [script_text]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline.asr import reset_asr_model, transcribe_vad_segments
from backend.pipeline.ingest import ingest_audio
from backend.pipeline.vad import detect_speech_regions


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_stage3.py <wav_path> [script_text]")
        sys.exit(1)

    wav_path = Path(sys.argv[1]).resolve()
    script_text = sys.argv[2] if len(sys.argv) > 2 else None

    if not wav_path.is_file():
        print(f"Not a file: {wav_path}")
        sys.exit(1)

    reset_asr_model()
    audio = ingest_audio(file_path=wav_path, target_sr=16000)
    vad = detect_speech_regions(audio, sample_rate=16000)

    script_texts = [script_text] if script_text else None
    result = transcribe_vad_segments(
        audio,
        vad,
        backend="mms",
        script_texts=script_texts,
    )

    print(f"\nFile: {wav_path.name}")
    print(f"VAD segments: {len(vad.segments)}")
    for seg in result:
        print(f"\nSegment {seg['index']}: {seg['start']:.3f}s - {seg['end']:.3f}s")
        print(f"Transcript: {seg['text']}")
        print(f"{'Word':<15} {'start_s':>9} {'end_s':>9} {'conf':>7}")
        print("-" * 45)
        for w in seg.get("words", []):
            print(f"{w['word']:<15} {w['start_s']:>9.3f} {w['end_s']:>9.3f} {w['confidence']:>7.3f}")


if __name__ == "__main__":
    main()
