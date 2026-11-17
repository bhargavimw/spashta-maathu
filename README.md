# Kannada Pause Remover & Stammer Cleaner

Automated pipeline to process hesitant Kannada speech, remove unnatural pauses, stammers, false starts, and fillers, and produce smoothed audio alongside clean transcripts while preserving meaning and natural prosody.

## Core Principles

- **Deterministic Audio Hands:** The LLM only determines word indices to keep/drop and pause overrides; the LLM never generates text or touches raw audio.
- **Sequential GPU Budget:** Stages (VAD, ASR, Forced Alignment, Local LLM) execute sequentially within an 8 GB VRAM budget.
- **Normalized Unicode:** All Kannada text normalized to Unicode NFC.
- **Audio Standards:** 16 kHz mono float32 throughout the pipeline.

## Repository Layout

```
kannada/
├── agents.md             # Standing agent instructions & rules
├── plan.md               # Architecture and roadmap specification
├── PROGRESS.md           # Milestone progress and verification log
├── backend/
│   ├── config.py         # Central configuration and tunable parameters
│   ├── cli.py            # CLI entry point
│   ├── pipeline/
│   │   ├── ingest.py     # Stage 0: Audio ingestion & normalization
│   │   ├── vad.py        # Stage 1: Voice activity detection
│   │   ├── asr.py        # Stage 2: Whole-file ASR transcription
│   │   ├── align.py      # Stage 3: CTC forced alignment
│   │   ├── classifier.py # Stage 4: LLM disfluency classifier & fallback
│   │   └── splicer.py    # Stage 5A: Zero-crossing cut & crossfader
│   ├── synth/
│   │   └── inject.py     # Disfluency synthesizer for evaluation
│   ├── eval/
│   │   └── run_eval.py   # Benchmark evaluation runner
│   ├── tests/            # Test suite
│   └── requirements.txt
├── data/
│   ├── raw/              # Raw audio inputs (not committed)
│   ├── synthetic/        # Synthesized disfluency data (not committed)
│   └── holdout/          # Real hesitant test clips (not committed)
└── requirements.txt
```

## Quickstart

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Run test suite
pytest
```
