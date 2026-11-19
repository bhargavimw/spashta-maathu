"""Check Stage 3 (ASR + forced alignment) on one WAV file.

Writes results to <wav_stem>_stage3.json in the same directory as the WAV.

Usage:
    python scripts/check_stage3.py <wav_path> [script_text]
"""

import json
import sys
from pathlib import Path

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

    # Write JSON output (not committed to repo).
    json_path = wav_path.with_suffix(".stage3.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "file": wav_path.name,
                "vad_segments": len(vad.segments),
                "segments": result,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nJSON written to: {json_path}")

    print(f"\nFile: {wav_path.name}")
    print(f"VAD segments: {len(vad.segments)}")

    result_by_index = {seg["index"]: seg for seg in result}

    for seg_idx, vad_seg in enumerate(vad.segments):
        seg = result_by_index.get(seg_idx)

        print(f"\n=== VAD segment {seg_idx}: {vad_seg.start_sec:.3f}s - {vad_seg.end_sec:.3f}s ===")

        if seg is None or not seg.get("text", "").strip():
            print("  (no speech decoded)")
            continue

        print(f"Transcript: {seg['text']}")
        print(
            f"  frames={seg['num_frames']} tokens={seg['num_tokens']} "
            f"(tokens/frames ratio: {seg['num_tokens']/seg['num_frames']:.2f})"
        )
        print(
            f"  {'Word':<18} {'spike_s':>9} {'spike_e':>9} "
            f"{'start_s':>9} {'end_s':>9} {'conf':>7}  frames  flag"
        )
        print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7}  {'-'*6}  {'-'*4}")
        for w in seg.get("words", []):
            flag = "SUSPECT" if w.get("suspect", False) else ""
            print(
                f"  {w['word']:<18} {w['spike_start_s']:>9.3f} {w['spike_end_s']:>9.3f} "
                f"{w['start_s']:>9.3f} {w['end_s']:>9.3f} {w['confidence']:>7.3f}  "
                f"{w['frames']:>6}  {flag}"
            )

    if not result:
        print("\n(no ASR segments produced)")


if __name__ == "__main__":
    main()
