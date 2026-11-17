# Kannada Pause Remover & Stammer Cleaner — Project Plan (v2)

## 1. Overview

**Objective:** Take hesitant Kannada speech, remove unnatural pauses, stammers, false starts and fillers, and output a smoothed audio file plus a clean transcript, preserving the speaker's words, meaning and natural prosody.

**Core philosophy:** *Timestamps are the coordinates, deterministic code is the hands, the LLM only chooses which words to keep.* The LLM never writes text and never touches audio.

**Staged delivery (important):**
- **v1 – Pause Remover (no LLM):** VAD + gap rules + splicer. Ships a working end-to-end product and the evaluation harness.
- **v2 – Stammer Cleaner:** word-level alignment + LLM keep/drop decisions on top of the v1 splicer.
- **v3 – API + UI.** Only after v1/v2 evaluate well.

## 2. Pipeline

```
[Input audio]
   │
   ▼
[Stage 0  Ingest]      16 kHz mono, peak normalize
   │
   ▼
[Stage 1  VAD]         Silero: speech regions + measured gaps
   │
   ▼
[Stage 2  ASR]         Whole-file transcription (model chosen by benchmark, see §4)
   │
   ▼
[Stage 3  Alignment]   CTC forced alignment -> word-level start/end times
   │
   ▼
[Stage 4  Classifier]  LLM picks word indices to keep/drop + pause overrides
   │                   (rule-based fallback if LLM fails)
   ├──► [Stage 5A Splicer]  cut at low-energy points, crossfade, rebuild audio
   └──► [Stage 5B Text]     clean text = concat of kept words; verbatim; diff
   │
   ▼
[Output] clean.wav, clean.txt, verbatim.txt, diff.json
```

**Why word-level instead of VAD-segment-level (change from v1 of this plan):**
stammers like "ನ ನ ನಾನು" have ~50 ms gaps that VAD merges into one region, so per-VAD-segment ASR cannot see them, and Whisper-style models tend to normalize disfluencies anyway. Forced alignment gives timestamps for every repeated token. VAD is kept only for measuring real pauses.

## 3. Key Design Decisions

| Decision | Rationale |
|---|---|
| LLM outputs **indices only**; `clean_text` is built in code from kept words | Removes a hallucination path; output text is always a subsequence of the verbatim text |
| Validate LLM output (indices in range, ascending, no duplicates) | Any violation triggers the fallback |
| For repetitions, prefer keeping the **last** (most fluent) instance | Usually the completed, best-prosody one |
| Cut points snapped to nearest low-energy sample / zero crossing within ±15 ms | Avoids clicks and mid-phoneme cuts |
| Pause policy configurable in `config.py` | Tune without code changes |
| Model IDs (LLM, ASR) live in config, never hardcoded | Free-tier model IDs get retired often (Gemini 2.0 Flash is already gone) |
| Normalize all Kannada text to Unicode NFC before comparison | Prevents false CER from equivalent codepoint sequences |

**Default pause policy (tune in M7):**
- within-clause gap > 250 ms -> shorten to 150–250 ms
- clause-boundary gap -> keep up to ~500 ms
- leading/trailing silence -> trim to ~100 ms
- crossfade 10–20 ms, equal-power

## 4. Technology (verify availability/limits at build time)

| Component | Choice | Notes |
|---|---|---|
| Backend | Python 3.10+, FastAPI (v3 only) | CLI first |
| VAD | Silero VAD | CPU is fine |
| ASR | **Benchmark first:** AI4Bharat Indic models (IndicConformer / IndicWhisper) vs Faster-Whisper large-v3 | Choose by CER on 10–20 Kannada clips; do not assume Whisper |
| Alignment | CTC forced aligner supporting Kannada (e.g. torchaudio MMS forced aligner or `ctc-forced-aligner`) | Word-level start/end times; fall back to ASR word timestamps if needed |
| LLM (cloud) | Current Gemini Flash-class model on the free tier (e.g. `gemini-3-flash-preview` or `gemini-3.1-flash-lite`), JSON mode | Model ID from config |
| LLM (local fallback) | Test a Gemma-family model via Ollama; Llama-3-8B is weak in Kannada | Must not be co-loaded with ASR on an 8 GB GPU |
| Audio ops | librosa, soundfile, numpy | |
| Frontend (v3) | React + TypeScript + Wavesurfer.js | |
| Storage (v3) | SQLite + local filesystem | |

**GPU budget (8 GB VRAM):** run ASR, alignment and any local LLM **sequentially**, releasing memory between stages. Do not load them together.

## 5. Data Contracts

**Words (output of Stage 3):**
```json
{"words": [{"i": 0, "text": "ನಾನು", "start": 0.12, "end": 0.55},
           {"i": 1, "text": "ನ",    "start": 0.60, "end": 0.66}],
 "gaps": [{"after": 0, "seconds": 0.05}]}
```

**LLM input:** word list with `i`, `text`, `start`, `end`, `gap_after`.

**LLM output (strict JSON):**
```json
{"keep": [0, 3, 5],
 "pause_overrides": [{"after": 0, "seconds": 0.25}],
 "removed": [{"i": 1, "reason": "false_start"}]}
```
Code derives `clean_text` from `keep`. Reasons enum: `repetition`, `false_start`, `filler`, `disfluent_pause`.

**Rule-based fallback:** drop consecutive duplicate words, drop words from a configurable filler list, cap gaps per the pause policy.

**LLM prompt principles:** Kannada speech-disfluency editor; keep the speaker's real words and meaning; remove repeats, false starts, fillers and disfluency micro-pauses; keep natural pauses at clause boundaries; return only JSON matching the schema; never add words.

## 6. Synthetic Data & Evaluation

**Base corpus:** IISc-MILE Kannada ASR corpus (verify access and licence). Needs word alignments first (reuse Stage 3).

**Injection types:** repetition, false start (first ~150 ms before full word), filler (real extracted/recorded filler audio, not synthetic tones), disfluent pause (200–400 ms inside a phrase).
**Realism rules:** add jitter, small pitch shift and time-stretch to injected audio so repeats are not bit-identical copies. Record ground-truth injection labels (type, word index, times).

**Outputs:** `(hesitant.wav, clean.wav, hesitant_text, clean_text, labels.json)`.

**Hold-out set:** record ~10 *real* hesitant clips (self/friends) and never train/tune on them. Synthetic results alone are not trusted.

**Metrics:**
- CER of clean transcript vs ground truth (NFC-normalized, report per-grapheme-cluster if possible)
- Disfluency recall (% injected disfluencies removed) and false-removal rate (% non-injected words wrongly dropped)
- Duration and pause statistics before/after
- Splice click check (automated: energy spike at join points) + manual listening
- Meaning-preservation spot check on the hold-out set (manual yes/no per clip)

## 7. Milestones (one agent session each; stop and review after every milestone)

| # | Milestone | Acceptance test |
|---|---|---|
| M0 | Repo scaffold, venv, config, logging, test setup | `pytest` runs (even with 1 trivial test); `config.py` loads |
| M1 | `ingest.py` + `vad.py` | On a 10 s clip: output is 16 kHz mono; VAD returns segments and gaps; unit tests pass |
| M2 | `splicer.py` + **v1 pause remover CLI** | Gap-capped output is shorter than input, no clicks at joins (automated check), duration math matches |
| M3 | ASR benchmark + alignment | CER table for candidate ASRs on 10–20 clips; word-level JSON produced; alignment sanity test |
| M4 | `classifier.py` (LLM + validation + fallback) | Mocked-LLM tests for invalid JSON, out-of-range indices and API failure all end in valid output |
| M5 | Wire v2 end-to-end CLI | `clean.wav`, `clean.txt`, `verbatim.txt`, `diff.json` produced; clean text is a subsequence of verbatim |
| M6 | `synth/inject.py` + eval script | 50 clips generated with labels; eval script prints all §6 metrics |
| M7 | Tune pause policy, crossfade, VAD params on the eval + hold-out sets | Documented before/after metrics |
| M8 | FastAPI (`POST /process`, `GET /health`) | Endpoint tests pass |
| M9 | React UI (waveform with removed regions, strikethrough diff, A/B player) | Manual demo on a hold-out clip |

## 8. Repo Structure

```
kannada-pause-remover/
├── AGENTS.md                 # standing instructions for the coding agent
├── plan.md                   # this file (source of truth)
├── PROGRESS.md               # agent updates after each milestone
├── backend/
│   ├── config.py
│   ├── cli.py
│   ├── app.py                # v3
│   ├── pipeline/
│   │   ├── ingest.py
│   │   ├── vad.py
│   │   ├── asr.py
│   │   ├── align.py
│   │   ├── classifier.py
│   │   └── splicer.py
│   ├── synth/inject.py
│   ├── eval/run_eval.py
│   ├── tests/
│   └── requirements.txt
├── frontend/                 # v3
├── data/
│   ├── raw/                  # not committed
│   ├── synthetic/            # not committed
│   └── holdout/              # real hesitant clips, not committed
└── README.md
```
Docker/MinIO are deferred until v3.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ASR normalizes/drops stammers | Word-level alignment; compare against verbatim hold-out |
| Whisper weak on Kannada | Benchmark Indic ASR models in M3 |
| LLM invalid JSON or free-tier model retired | JSON mode, strict validation, config-driven model ID, rule-based fallback |
| LLM drops real content words | Code-built `clean_text`; track false-removal rate; keep-last-repetition rule |
| Clicks / pitch jumps at joins | Cut at low-energy points; 10–20 ms equal-power crossfade; manual listening in M7 |
| GPU OOM (8 GB) | Sequential stages, `int8` where supported, release models between stages |
| Synthetic data too easy | Jitter/pitch/stretch in injector; real hold-out set |
| Dataset access/licence | Verify before M6; fall back to self-recorded clean speech |

## 10. Out of Scope (MVP)
Real-time processing, multi-speaker audio, languages other than Kannada, voice cloning/resynthesis of removed words, auth/multi-user storage.