# PROGRESS.md — Milestone Tracking

## Milestone M0: Repo Scaffold, venv, Config, Logging, Test Setup
- **Status:** DONE
- **Date:** 2026-11-17

### What was done
1. **Repository Structure Scaffolding:**
   - Created directory layout per Section 8 of `plan.md`:
     - `backend/`
     - `backend/pipeline/` (`ingest.py`, `vad.py`, `asr.py`, `align.py`, `classifier.py`, `splicer.py`)
     - `backend/synth/` (`inject.py`)
     - `backend/eval/` (`run_eval.py`)
     - `backend/tests/` (`test_config.py`)
     - `data/raw/`, `data/synthetic/`, `data/holdout/`
   - Added `.gitignore` to prevent committing audio, model weights, secrets, and data.
   - Added project `README.md`.
   - Populated all pipeline and utility modules with descriptive docstrings (no pipeline logic implemented).
   - Omitted Docker and frontend as instructed for M0.

2. **Virtual Environment & Dependencies:**
   - Initialized `.venv` using Python 3.12.3.
   - Created `backend/requirements.txt` and root `requirements.txt` with pinned versions (`pytest`, `numpy`, `scipy`, `soundfile`, `librosa`, `torch`, `faster-whisper`, `google-genai`, etc.).
   - Installed `pytest` and runtime prerequisites in `.venv`.

3. **Configuration Module (`backend/config.py`):**
   - Configured audio standard: 16 kHz mono float32, `snap_window_ms = 15.0`.
   - Configured pause policy defaults:
     - `min_gap_to_shorten_ms = 250.0`
     - `shortened_gap_min_ms = 150.0`, `shortened_gap_max_ms = 250.0`, `shortened_gap_target_ms = 200.0`
     - `clause_boundary_max_gap_ms = 500.0`
     - `edge_silence_trim_ms = 100.0`
     - `crossfade_ms = 15.0`, `crossfade_curve = "equal-power"`
   - Model IDs read from environment variables with fallbacks:
     - `GEMINI_MODEL_ID` (`gemini-2.5-flash`)
     - `LOCAL_LLM_MODEL_ID` (`gemma2:9b`)
     - `ASR_MODEL_ID` (`ai4bharat/indicconformer_stt_kn_hybrid`)
     - `ALIGNER_MODEL_ID` (`facebook/mms-1b-all`)
     - `VAD_MODEL_ID` (`silero_vad`)
   - Hardware constraints: sequential GPU execution, 8.0 GB VRAM budget.
   - Kannada text processing: Unicode NFC normalization, disfluency reasons enum, filler words list.
   - Structured logging setup function `setup_logging()`.

4. **Testing Harness:**
   - Created `backend/tests/test_config.py` with trivial pass test, default verification test, environment override test, and logging test.

### How it was verified
1. **Pytest Run:**
   ```bash
   $ .venv/bin/pytest -v
   ============================= test session starts ==============================
   platform linux -- Python 3.12.3, pytest-9.2.0, pluggy-1.7.0 -- /home/bhargavimw/Desktop/hackathons/kannada/.venv/bin/python3
   cachedir: .pytest_cache
   rootdir: /home/bhargavimw/Desktop/hackathons/kannada
   collecting ... collected 4 items

   backend/tests/test_config.py::test_trivial_pass PASSED                   [ 25%]
   backend/tests/test_config.py::test_config_loads_with_plan_defaults PASSED [ 50%]
   backend/tests/test_config.py::test_config_env_overrides PASSED           [ 75%]
   backend/tests/test_config.py::test_logging_setup PASSED                  [100%]

   ============================== 4 passed in 0.11s ===============================
   ```
2. **Config Loading Test:**
   - Loaded `backend.config` in Python 3.12 to verify dataclass initialization and path resolution.

### Open issues / Next steps
- Ready for Milestone M1: Implement `ingest.py` (16 kHz mono float32 normalization) and `vad.py` (Silero VAD speech/gap segmentation).
