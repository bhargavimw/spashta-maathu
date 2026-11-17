# AGENTS.md — Kannada Pause Remover & Stammer Cleaner

## Source of truth
- `plan.md` defines architecture, data contracts, milestones and acceptance tests. Read it before any task.
- Work on **one milestone at a time**. Do not start the next milestone unless asked.
- After finishing a milestone, update `PROGRESS.md` (done, how it was verified, open issues) and stop.

## Hard rules
- The LLM only returns word indices and pause overrides. `clean_text` is always built in code from kept words. Never let model output introduce new words.
- All model IDs, thresholds, crossfade lengths and pause policy live in `backend/config.py`. Never hardcode them.
- Normalize Kannada text to Unicode NFC before any comparison or metric.
- Audio: 16 kHz mono float32 internally. Never silently resample or clip; assert shapes and sample rates.
- Run ASR, alignment and any local LLM sequentially on the GPU (8 GB VRAM). Release models between stages.
- Never commit audio, model weights, `.env`, or API keys. Read keys from environment variables.
- Do not invent APIs or model names. If unsure a library/model supports Kannada, say so and write a small test to check.

## Code standards
- Python 3.10+, type hints on all signatures, docstrings on public functions.
- Small pure functions for audio operations so they are unit-testable with synthetic numpy signals.
- Every module ships with tests in `backend/tests/`. Use generated synthetic audio (sine bursts + silence) for unit tests; no network in tests (mock the LLM).
- Use a virtual environment. Pin versions in `requirements.txt`.

## Verification
- Before declaring a milestone done: run `pytest`, run the milestone's acceptance test from `plan.md`, and report the actual output.
- If a check fails, fix it or report it. Do not weaken tests to pass.

## System / environment commands
For any system or environment fix, give three steps: (1) a command that checks the current state, (2) the command that makes the change, (3) a command that checks the updated state.

## Scope control
- Do not add Docker, a frontend, or an API before the milestones that call for them.
- If a task needs a design decision not in `plan.md`, stop and ask instead of choosing silently.