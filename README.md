# Spashta Maathu (ಸ್ಪಷ್ಟ ಮಾತು)

**Turning hesitant, stammered Kannada speech into clear, fluent speech, as clean text and clean audio.**

> 🇬🇧 English version first, 🇮🇳 ಕನ್ನಡ ಆವೃತ್ತಿ ಕೆಳಗೆ (jump to [ಕನ್ನಡ](#ಕನ್ನಡ-ಆವೃತ್ತಿ)).

---

## 1. What is this?

Spashta Maathu ("clear speech" in Kannada) is a pipeline that takes a recording of someone speaking Kannada with pauses, fillers, repeated sounds, prolongations, blocks or false starts, and gives back a clean version of the same speech that still sounds like a real person.

We built it because most speech tools are trained on fluent speech, and they either choke on stammered speech or quietly "fix" it in ways you can't control. We wanted something transparent: every disfluency is **found, labelled with a start and end time, and then removed by deterministic rules**, not by a model that rewrites your words.

All speech in this project is either acted or recorded from consenting volunteers.

## 2. What goes in, what comes out

**Input:** a Kannada audio file (16 kHz, mono, WAV).

**Output:**

| # | Output | Description |
|---|--------|-------------|
| 1 | Verbatim transcript | Exactly what was said, disfluencies included |
| 2 | Annotation CSV | One row per disfluency: tag, start time, end time |
| 3 | Clean transcript | Disfluencies removed, meaning preserved |
| 4 | Clean audio | Disfluent spans cut, the rest stitched together smoothly |
| 5 | *(Bonus)* Regenerated speech | Clean sentence spoken by Kannada TTS or a cloned voice |

## 3. Examples

| Type | Messy | Clean |
|------|-------|-------|
| Pause + filler | ನಾನು ... ಉಹ್... ಬೆಂಗಳೂರಿನಲ್ಲಿ...ಮ್ಮ್... ವಾಸಿಸುತ್ತೇನೆ | ನಾನು ಬೆಂಗಳೂರಿನಲ್ಲಿ ವಾಸಿಸುತ್ತೇನೆ. |
| Repeat | ನಾ-ನಾ-ನಾನು ಶಾ-ಶಾಲೆಗೆ ಹೋಗ್ತೀನಿ | ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತೇನೆ. |
| False start | ನನ್ನ... ನನ್ನ ಮನೆ ಮೈಸೂರಿನಲ್ಲಿದೆ | ನನ್ನ ಮನೆ ಮೈಸೂರಿನಲ್ಲಿದೆ. |

The one rule behind all of these: **keep the last complete version of any repeated or restarted unit, and delete the earlier fragments.**

## 4. How the pipeline works

```
Messy audio
  -> Preprocess (16 kHz mono, light denoising only)
  -> VAD (find pauses)
  -> ASR + forced alignment (verbatim transcript, word/char timestamps)
  -> Tagger (PAUSE, FILLER, REP, PROL, BLOCK, FALSESTART)
  -> Annotation CSV (tag, start, end)
  -> Clean text (delete tagged spans)
  -> Cut and stitch audio (crossfade)
  -> (bonus) TTS / voice cloning regeneration
```

**Stage 1: Preprocess.** Resample to 16 kHz mono. Only light, non-generative denoising (DeepFilterNet, RNNoise or noisereduce). We deliberately avoid generative enhancers because they can rewrite the very disfluencies we are trying to measure.

**Stage 2: VAD and pause detection.** Silero VAD finds silence gaps. Each gap is recorded with its duration and labelled light (1 to 3 s), medium (3 to 5 s) or heavy (over 5 s).

**Stage 3: Verbatim transcript and timestamps.** We do *not* use Whisper, because it silently drops disfluencies and is weak on Kannada. Instead we use `facebook/mms-1b-all` (Kannada adapter, character-level CTC) or AI4Bharat IndicConformer, with word and character timestamps from `torchaudio.functional.forced_align`. For acted speech the stammered script is already known, so we force-align against the script rather than relying on free ASR.

**Stage 4: Tagging.** An LLM disfluency classifier labels spans in the aligned word list, with a rule-based fallback for when it is unsure or unavailable. It only labels; it never rewrites text. The rules look for: PAUSE from VAD gaps; FILLER from a lexicon plus short voiced segments with no lexical content; REP from prefix-of-next-word or identical word/n-gram patterns; FALSESTART from a fragment followed by a restart with the same opening (the earlier one is marked); PROL from duration z-scores against the speaker's own median; BLOCK from silence inside a word or an abnormal gap before onset.

**Stage 5: Annotation CSV.** One row per tag (format below).

**Stage 6: Clean text.** Tagged spans are deleted from the verbatim transcript by deterministic rules. No LLM rewriting, because WER is judged against a reference and paraphrasing would hurt. Output is normalised to UTF-8 Unicode Kannada with proper sentence-ending punctuation.

**Stage 7: Cut and stitch audio.** Cuts land on word boundaries snapped to zero crossings, with 10 to 30 ms crossfades at every join. A natural 150 to 300 ms pause is kept or inserted at sentence ends, because stripping all silence makes the result sound robotic.

**Stage 8 (bonus): Regenerate.** Indic Parler-TTS, or IndicF5 for voice cloning (Kannada support and licence should be verified before use), speaks the clean sentence. Cloning needs explicit speaker consent, and synthetic samples are flagged in metadata.

**Annotation CSV format:**

```csv
file,tag,start_s,end_s,text
spk01_partA_1.wav,FILLER,2.41,2.95,ಉಹ್
spk01_partA_1.wav,PAUSE,2.95,4.30,
spk01_partB_3.wav,REP,0.00,0.62,ನಾ-ನಾ-
```

## 5. Tag taxonomy

| Tag | Meaning | Detection signal |
|-----|---------|------------------|
| `PAUSE` | Silence gap in speech | Silero VAD gaps: light 1 to 3 s, medium 3 to 5 s, heavy over 5 s |
| `FILLER` | Hesitation sounds (ಉಹ್, ಮ್ಮ್, ಆಂ) | Lexicon match in transcript plus short voiced-segment acoustic check |
| `REP` | Repeated sound, syllable, word or n-gram | Fragment that is a prefix of the next word, or the same word/n-gram twice in a row |
| `PROL` | Prolonged sound | Phoneme or word duration far above the speaker's norm |
| `BLOCK` | Silent block inside or before a word | Silence inside a word, or a long gap just before word onset |
| `FALSESTART` | Incomplete fragment, then restart | Fragment followed by a restart with the same opening |

## 6. Repository layout

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

A quick tour: `agents.md`, `plan.md` and `PROGRESS.md` hold the working rules, the design, and a running log of what we have built and verified. All tunable parameters (pause thresholds, crossfade length and so on) live in `backend/config.py`, so nothing is buried in the code. Audio and datasets are **not committed** to the repo.

## 7. Setup and usage

You'll need Python 3.10+ and, ideally, a GPU. We developed on an RTX 5050 (8 GB), so we use fp16 inference locally and Kaggle or Colab for anything heavier. Everything here is free and open source.

**Check what you have, install, then verify:**

```bash
# 1. check current state
python --version
pip list | grep -i -E "torch|torchaudio|transformers"

# 2. install
pip install -r requirements.txt

# 3. check updated state
pip list | grep -i -E "torch|torchaudio|transformers"
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
```

**Run the pipeline** through the CLI. Start with the help text to see the exact options:

```bash
python backend/cli.py --help
python backend/cli.py <your-options-here>   # input audio in, clean text + clean audio out
```

**Run the tests and the benchmark:**

```bash
python -m pytest backend/tests
python backend/eval/run_eval.py
```

**Make evaluation data:** `backend/synth/inject.py` adds known disfluencies to clean speech, so we know the ground truth for scoring.

## 8. Dataset

We have **9 speakers**, with **2 sets from each speaker**.

- **Set 1 (pauses and fillers):** one recording of 2 to 3 minutes per speaker, with pauses and fillers, on a topic like daily routine, hometown or an abstract idea.
- **Set 2 (stammers and unclear speech):** 40 or more sentence pairs per speaker, each recorded once unclear/stammered and once clear, with transcripts.

**How the data is organised:** raw recordings go in `data/raw/`, synthetic disfluency data (made with `inject.py`) in `data/synthetic/`, and real hesitant test clips in `data/holdout/`. The holdout clips are kept apart so we can check the system on real speech it was never tuned on. None of these folders are committed to the repo.

**How we made it:** we wrote the stammered script first and had speakers read it, so the verbatim transcript and the clean reference existed before recording. Then we auto-tagged, corrected the tags by hand in Audacity or Label Studio, exported the CSV, and had another team review it.

**Format:** audio is 16 kHz mono WAV; text is UTF-8 Unicode Kannada (no Nudi or Baraha). One metadata CSV per team, under `team_XX/problem_YY/`.

## 9. Evaluation

| Metric | How it is computed |
|--------|--------------------|
| Pauses and fillers removed (%) | (tags before − tags remaining after) / tags before |
| WER of clean text | Word error rate vs the reference clean transcript |
| Tag precision / recall | Predicted tags vs hand-corrected annotation CSV (match on tag type and time overlap) |
| Listening score | 1 to 5 by 3 reviewers; penalises over-cutting, clicks and robotic gaps |

There is a trade-off: removing more disfluencies raises the removal %, but over-cutting hurts WER and the listening score. We favour **high precision** on tags.

**Our results:**

| Metric | Score |
|--------|-------|
| Pauses and fillers removed (%) | _fill in_ |
| WER (clean text) | _fill in_ |
| Tag precision / recall | _fill in_ |
| Listening score (avg of 3) | _fill in_ |

## 10. Ethics, consent and licence

- Signed consent was collected from every speaker and writer (see `consent/`).
- Speech is acted or from consenting volunteers only.
- No phone numbers or addresses appear in any file.
- Content is original: no songs, films or book text.
- Cloned voices need explicit speaker consent, and all synthetic samples are flagged in metadata.
- Planned open release under **CC-BY 4.0**, with contributors credited.

## 11. Known limitations

We would rather be upfront about these:

- ASR is biased towards standard Kannada and does worse on regional accents.
- Very short fragments give noisy alignment.
- CTC character models can drop or merge repeated syllables, so we verify against the known script.
- PROL, BLOCK and FALSESTART are the hardest tags; expect the lowest recall there.
- Cutting too aggressively makes speech sound choppy, so we cut conservatively.

## 12. Team and credits

- **Team:** _Team kannadakaagi ondannu otti, 
- **Tools we relied on:** Silero VAD, MMS (facebook/mms-1b-all), AI4Bharat IndicConformer, torchaudio, DeepFilterNet / RNNoise / noisereduce, Indic Parler-TTS, Audacity, Label Studio.
- Thanks to every volunteer who read our scripts and lent us their voice.

---
---

<a id="ಕನ್ನಡ-ಆವೃತ್ತಿ"></a>

# ಸ್ಪಷ್ಟ ಮಾತು (Spashta Maathu): ಕನ್ನಡ ಆವೃತ್ತಿ

**ತೊದಲುವ, ಹಿಂಜರಿಯುವ ಕನ್ನಡ ಮಾತನ್ನು ಸ್ಪಷ್ಟ, ಸರಾಗ ಮಾತಾಗಿ ಬದಲಾಯಿಸುವ ಯೋಜನೆ: ಸ್ವಚ್ಛ ಪಠ್ಯ ಮತ್ತು ಸ್ವಚ್ಛ ಆಡಿಯೋ.**

## 1. ಇದು ಏನು?

ಸ್ಪಷ್ಟ ಮಾತು ಎಂದರೆ, ವಿರಾಮಗಳು, ತುಂಬುಧ್ವನಿಗಳು (ಉಹ್, ಮ್ಮ್), ಪುನರಾವರ್ತನೆಗಳು, ಧ್ವನಿ ಎಳೆಯುವಿಕೆ, ಅಡಚಣೆಗಳು ಮತ್ತು ತಪ್ಪು ಆರಂಭಗಳಿರುವ ಕನ್ನಡ ಮಾತಿನ ರೆಕಾರ್ಡಿಂಗ್ ಅನ್ನು ತೆಗೆದುಕೊಂಡು, ಅದೇ ಮಾತಿನ ಸ್ವಚ್ಛ ರೂಪವನ್ನು ಹಿಂತಿರುಗಿಸುವ ಪೈಪ್‌ಲೈನ್. ಹೀಗೆ ಸಿಗುವ ಮಾತು ಇನ್ನೂ ನಿಜವಾದ ಮನುಷ್ಯನ ಧ್ವನಿಯಂತೆಯೇ ಕೇಳಿಸುತ್ತದೆ.

ಹೆಚ್ಚಿನ ಮಾತಿನ ಉಪಕರಣಗಳು ಸರಾಗವಾದ ಮಾತಿನ ಮೇಲೆ ತರಬೇತಿ ಪಡೆದಿರುತ್ತವೆ. ತೊದಲುವ ಮಾತಿನಲ್ಲಿ ಅವು ತಪ್ಪುತ್ತವೆ, ಅಥವಾ ನಮಗೆ ತಿಳಿಯದಂತೆ ಮಾತನ್ನೇ ಬದಲಾಯಿಸಿಬಿಡುತ್ತವೆ. ನಮಗೆ ಬೇಕಾಗಿದ್ದದ್ದು ಪಾರದರ್ಶಕ ವ್ಯವಸ್ಥೆ. ಇಲ್ಲಿ ಪ್ರತಿಯೊಂದು ಅಡಚಣೆಯನ್ನೂ **ಪತ್ತೆಹಚ್ಚಿ, ಆರಂಭ ಮತ್ತು ಅಂತ್ಯ ಸಮಯದೊಂದಿಗೆ ಗುರುತಿಸಿ, ನಿಯಮ-ಆಧಾರಿತ ವಿಧಾನದಿಂದ ತೆಗೆದುಹಾಕುತ್ತೇವೆ**. ನಿಮ್ಮ ಮಾತನ್ನು ಮರುಬರೆಯುವ ಯಾವುದೇ ಮಾದರಿ ಇಲ್ಲಿ ಇಲ್ಲ.

ಈ ಯೋಜನೆಯ ಎಲ್ಲಾ ಮಾತು ಅಭಿನಯಿಸಿದ್ದು ಅಥವಾ ಒಪ್ಪಿಗೆ ನೀಡಿದ ಸ್ವಯಂಸೇವಕರಿಂದ ರೆಕಾರ್ಡ್ ಮಾಡಿದ್ದು.

## 2. ಏನು ಕೊಡುತ್ತೇವೆ, ಏನು ಸಿಗುತ್ತದೆ

**ಇನ್‌ಪುಟ್:** ಕನ್ನಡ ಆಡಿಯೋ ಫೈಲ್ (16 kHz, ಮೊನೊ, WAV).

**ಔಟ್‌ಪುಟ್:**

| # | ಔಟ್‌ಪುಟ್ | ವಿವರ |
|---|---------|------|
| 1 | ಯಥಾವತ್ ಲಿಪ್ಯಂತರ (Verbatim transcript) | ಅಡಚಣೆಗಳೂ ಸೇರಿ, ಆಡಿದ್ದು ಆಡಿದಂತೆಯೇ |
| 2 | ಟಿಪ್ಪಣಿ CSV (Annotation CSV) | ಪ್ರತಿ ಅಡಚಣೆಗೂ ಒಂದು ಸಾಲು: ಟ್ಯಾಗ್, ಆರಂಭ ಸಮಯ, ಅಂತ್ಯ ಸಮಯ |
| 3 | ಸ್ವಚ್ಛ ಪಠ್ಯ | ಅಡಚಣೆಗಳನ್ನು ತೆಗೆದಿದೆ, ಅರ್ಥ ಹಾಗೇ ಉಳಿದಿದೆ |
| 4 | ಸ್ವಚ್ಛ ಆಡಿಯೋ | ಅಡಚಣೆಯ ಭಾಗಗಳನ್ನು ಕತ್ತರಿಸಿ, ಉಳಿದದ್ದನ್ನು ಸರಾಗವಾಗಿ ಜೋಡಿಸಿದೆ |
| 5 | *(ಬೋನಸ್)* ಮರುಸೃಷ್ಟಿಸಿದ ಮಾತು | ಸ್ವಚ್ಛ ವಾಕ್ಯವನ್ನು ಕನ್ನಡ TTS ಅಥವಾ ಕ್ಲೋನ್ ಮಾಡಿದ ಧ್ವನಿಯಲ್ಲಿ ಹೇಳಿಸಿದ್ದು |

## 3. ಉದಾಹರಣೆಗಳು

| ವಿಧ | ಅಸ್ಪಷ್ಟ (Messy) | ಸ್ವಚ್ಛ (Clean) |
|-----|----------------|----------------|
| ವಿರಾಮ + ತುಂಬುಧ್ವನಿ | ನಾನು ... ಉಹ್... ಬೆಂಗಳೂರಿನಲ್ಲಿ...ಮ್ಮ್... ವಾಸಿಸುತ್ತೇನೆ | ನಾನು ಬೆಂಗಳೂರಿನಲ್ಲಿ ವಾಸಿಸುತ್ತೇನೆ. |
| ಪುನರಾವರ್ತನೆ | ನಾ-ನಾ-ನಾನು ಶಾ-ಶಾಲೆಗೆ ಹೋಗ್ತೀನಿ | ನಾನು ಶಾಲೆಗೆ ಹೋಗುತ್ತೇನೆ. |
| ತಪ್ಪು ಆರಂಭ | ನನ್ನ... ನನ್ನ ಮನೆ ಮೈಸೂರಿನಲ್ಲಿದೆ | ನನ್ನ ಮನೆ ಮೈಸೂರಿನಲ್ಲಿದೆ. |

ಇವೆಲ್ಲದರ ಹಿಂದಿರುವ ಒಂದೇ ನಿಯಮ: **ಪುನರಾವರ್ತಿತ ಅಥವಾ ಮರುಆರಂಭಿಸಿದ ಭಾಗದ ಕೊನೆಯ ಪೂರ್ಣ ರೂಪವನ್ನು ಉಳಿಸಿ, ಅದಕ್ಕೂ ಮೊದಲಿನ ತುಣುಕುಗಳನ್ನು ತೆಗೆದುಹಾಕಿ.**

## 4. ಪೈಪ್‌ಲೈನ್ ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ

```
ಅಸ್ಪಷ್ಟ ಆಡಿಯೋ
  -> ಪೂರ್ವಸಂಸ್ಕರಣೆ (16 kHz ಮೊನೊ, ಹಗುರ ಶಬ್ದ ನಿವಾರಣೆ ಮಾತ್ರ)
  -> VAD (ವಿರಾಮ ಪತ್ತೆ)
  -> ASR + ಫೋರ್ಸ್ಡ್ ಅಲೈನ್‌ಮೆಂಟ್ (ಯಥಾವತ್ ಲಿಪ್ಯಂತರ, ಪದ/ಅಕ್ಷರ ಸಮಯಮುದ್ರೆ)
  -> ಟ್ಯಾಗರ್ (PAUSE, FILLER, REP, PROL, BLOCK, FALSESTART)
  -> ಟಿಪ್ಪಣಿ CSV (ಟ್ಯಾಗ್, ಆರಂಭ, ಅಂತ್ಯ)
  -> ಸ್ವಚ್ಛ ಪಠ್ಯ (ಟ್ಯಾಗ್ ಮಾಡಿದ ಭಾಗ ಅಳಿಸಿ)
  -> ಆಡಿಯೋ ಕತ್ತರಿಸಿ ಜೋಡಿಸುವುದು (ಕ್ರಾಸ್‌ಫೇಡ್)
  -> (ಬೋನಸ್) TTS / ಧ್ವನಿ ಕ್ಲೋನಿಂಗ್ ಮೂಲಕ ಮರುಸೃಷ್ಟಿ
```

**ಹಂತ 1: ಪೂರ್ವಸಂಸ್ಕರಣೆ.** ಆಡಿಯೋವನ್ನು 16 kHz ಮೊನೊಗೆ ಬದಲಾಯಿಸುತ್ತೇವೆ. ಹಗುರವಾದ, ಜನರೇಟಿವ್ ಅಲ್ಲದ ಶಬ್ದ ನಿವಾರಣೆ ಮಾತ್ರ (DeepFilterNet, RNNoise ಅಥವಾ noisereduce). ಜನರೇಟಿವ್ ಎನ್‌ಹ್ಯಾನ್ಸರ್‌ಗಳನ್ನು ನಾವು ಬೇಕಂತಲೇ ಬಳಸುವುದಿಲ್ಲ, ಏಕೆಂದರೆ ನಾವು ಅಳೆಯಬೇಕಾದ ಅಡಚಣೆಗಳನ್ನೇ ಅವು ಬದಲಾಯಿಸಬಹುದು.

**ಹಂತ 2: VAD ಮತ್ತು ವಿರಾಮ ಪತ್ತೆ.** Silero VAD ಮೌನದ ಅಂತರಗಳನ್ನು ಹುಡುಕುತ್ತದೆ. ಪ್ರತಿ ಅಂತರದ ಅವಧಿಯನ್ನು ದಾಖಲಿಸಿ, ಹಗುರ (1 ರಿಂದ 3 ಸೆ), ಮಧ್ಯಮ (3 ರಿಂದ 5 ಸೆ) ಅಥವಾ ಭಾರಿ (5 ಸೆಕೆಂಡ್‌ಗಿಂತ ಹೆಚ್ಚು) ಎಂದು ಗುರುತಿಸುತ್ತೇವೆ.

**ಹಂತ 3: ಯಥಾವತ್ ಲಿಪ್ಯಂತರ ಮತ್ತು ಸಮಯಮುದ್ರೆ.** ನಾವು Whisper ಬಳಸುವುದಿಲ್ಲ, ಏಕೆಂದರೆ ಅದು ಅಡಚಣೆಗಳನ್ನು ಮೌನವಾಗಿ ತೆಗೆದುಹಾಕುತ್ತದೆ ಮತ್ತು ಕನ್ನಡದಲ್ಲಿ ದುರ್ಬಲ. ಬದಲಿಗೆ `facebook/mms-1b-all` (ಕನ್ನಡ ಅಡಾಪ್ಟರ್, ಅಕ್ಷರ-ಮಟ್ಟದ CTC) ಅಥವಾ AI4Bharat IndicConformer ಬಳಸುತ್ತೇವೆ. ಪದ ಮತ್ತು ಅಕ್ಷರ ಸಮಯಮುದ್ರೆಗಳು `torchaudio.functional.forced_align` ಮೂಲಕ ಬರುತ್ತವೆ. ಅಭಿನಯಿಸಿದ ಮಾತಿಗೆ ತೊದಲುವ ಸ್ಕ್ರಿಪ್ಟ್ ಮೊದಲೇ ತಿಳಿದಿರುವುದರಿಂದ, ಉಚಿತ ASR ಬದಲು ಆ ಸ್ಕ್ರಿಪ್ಟ್ ವಿರುದ್ಧವೇ ಫೋರ್ಸ್ಡ್ ಅಲೈನ್‌ಮೆಂಟ್ ಮಾಡುತ್ತೇವೆ.

**ಹಂತ 4: ಟ್ಯಾಗಿಂಗ್.** ಜೋಡಿಸಿದ ಪದಪಟ್ಟಿಯಲ್ಲಿನ ಭಾಗಗಳಿಗೆ LLM ಅಡಚಣೆ ವರ್ಗೀಕರಣಕಾರ ಲೇಬಲ್ ಹಾಕುತ್ತದೆ; ಅದು ಖಚಿತವಿಲ್ಲದಿದ್ದರೆ ಅಥವಾ ಲಭ್ಯವಿಲ್ಲದಿದ್ದರೆ ನಿಯಮ-ಆಧಾರಿತ ಪರ್ಯಾಯ ವಿಧಾನ ಬಳಸುತ್ತೇವೆ. ಅದು ಲೇಬಲ್ ಮಾತ್ರ ಹಾಕುತ್ತದೆ, ಪಠ್ಯವನ್ನು ಮರುಬರೆಯುವುದಿಲ್ಲ. ನಿಯಮಗಳು ಹುಡುಕುವುದು: PAUSE (VAD ಅಂತರಗಳಿಂದ); FILLER (ಪದಕೋಶ ಮತ್ತು ಅರ್ಥವಿಲ್ಲದ ಚಿಕ್ಕ ಧ್ವನಿ ಭಾಗಗಳಿಂದ); REP (ಮುಂದಿನ ಪದದ ಪೂರ್ವಪ್ರತ್ಯಯವಾದ ತುಣುಕು ಅಥವಾ ಒಂದೇ ಪದ/ಎನ್‌-ಗ್ರಾಮ್ ಸತತ ಎರಡು ಬಾರಿ); FALSESTART (ಅಪೂರ್ಣ ತುಣುಕಿನ ನಂತರ ಅದೇ ಆರಂಭದ ಮರುಆರಂಭ, ಮೊದಲಿನದನ್ನು ಗುರುತಿಸುತ್ತೇವೆ); PROL (ಮಾತನಾಡುವವರ ಸ್ವಂತ ಮಧ್ಯಮ ಅವಧಿಗೆ ಹೋಲಿಸಿದ z-ಸ್ಕೋರ್); BLOCK (ಪದದೊಳಗಿನ ಮೌನ ಅಥವಾ ಆರಂಭದ ಮೊದಲಿನ ಅಸಹಜ ಅಂತರ).

**ಹಂತ 5: ಟಿಪ್ಪಣಿ CSV.** ಪ್ರತಿ ಟ್ಯಾಗ್‌ಗೆ ಒಂದು ಸಾಲು (ಮೇಲಿನ ಇಂಗ್ಲಿಷ್ ವಿಭಾಗದಲ್ಲಿ ಸ್ವರೂಪ ನೋಡಿ).

**ಹಂತ 6: ಸ್ವಚ್ಛ ಪಠ್ಯ.** ಯಥಾವತ್ ಲಿಪ್ಯಂತರದಿಂದ ಟ್ಯಾಗ್ ಮಾಡಿದ ಭಾಗಗಳನ್ನು ನಿರ್ಧಾರಿತ (deterministic) ನಿಯಮಗಳಿಂದ ಅಳಿಸುತ್ತೇವೆ. LLM ಮೂಲಕ ಮರುಬರೆಯುವುದಿಲ್ಲ, ಏಕೆಂದರೆ WER ಅನ್ನು ಉಲ್ಲೇಖ ಪಠ್ಯದ ವಿರುದ್ಧ ಅಳೆಯಲಾಗುತ್ತದೆ ಮತ್ತು ಪದಬದಲಾವಣೆ ಅದನ್ನು ಹಾಳುಮಾಡುತ್ತದೆ. ಔಟ್‌ಪುಟ್ ಅನ್ನು UTF-8 ಯೂನಿಕೋಡ್ ಕನ್ನಡಕ್ಕೆ ಸಾಮಾನ್ಯೀಕರಿಸಿ, ವಾಕ್ಯದ ಕೊನೆಗೆ ಸರಿಯಾದ ವಿರಾಮಚಿಹ್ನೆ ಹಾಕುತ್ತೇವೆ.

**ಹಂತ 7: ಆಡಿಯೋ ಕತ್ತರಿಸಿ ಜೋಡಿಸುವುದು.** ಕತ್ತರಿಸುವುದು ಪದದ ಗಡಿಗಳಲ್ಲಿ, ಝೀರೋ ಕ್ರಾಸಿಂಗ್‌ಗೆ ಹೊಂದಿಸಿ, ಪ್ರತಿ ಜೋಡಣೆಯಲ್ಲಿ 10 ರಿಂದ 30 ms ಕ್ರಾಸ್‌ಫೇಡ್ ಇರುತ್ತದೆ. ವಾಕ್ಯದ ಕೊನೆಯಲ್ಲಿ 150 ರಿಂದ 300 ms ಸಹಜ ವಿರಾಮವನ್ನು ಉಳಿಸುತ್ತೇವೆ ಅಥವಾ ಸೇರಿಸುತ್ತೇವೆ. ಎಲ್ಲಾ ಮೌನವನ್ನು ತೆಗೆದರೆ ಧ್ವನಿ ರೋಬೋಟ್‌ನಂತೆ ಕೇಳಿಸುತ್ತದೆ.

**ಹಂತ 8 (ಬೋನಸ್): ಮರುಸೃಷ್ಟಿ.** Indic Parler-TTS, ಅಥವಾ ಧ್ವನಿ ಕ್ಲೋನಿಂಗ್‌ಗೆ IndicF5 (ಬಳಸುವ ಮೊದಲು ಕನ್ನಡ ಬೆಂಬಲ ಮತ್ತು ಪರವಾನಗಿಯನ್ನು ಪರಿಶೀಲಿಸಿ), ಸ್ವಚ್ಛ ವಾಕ್ಯವನ್ನು ಹೇಳುತ್ತದೆ. ಕ್ಲೋನಿಂಗ್‌ಗೆ ಮಾತನಾಡುವವರ ಸ್ಪಷ್ಟ ಒಪ್ಪಿಗೆ ಬೇಕು, ಮತ್ತು ಕೃತಕ (synthetic) ಮಾದರಿಗಳನ್ನು ಮೆಟಾಡೇಟಾದಲ್ಲಿ ಗುರುತಿಸಲಾಗುತ್ತದೆ.

## 5. ಟ್ಯಾಗ್ ವರ್ಗೀಕರಣ

| ಟ್ಯಾಗ್ | ಅರ್ಥ | ಪತ್ತೆ ಮಾಡುವ ವಿಧಾನ |
|--------|------|-------------------|
| `PAUSE` | ಮಾತಿನಲ್ಲಿ ಮೌನದ ಅಂತರ | Silero VAD ಅಂತರಗಳು: ಹಗುರ 1 ರಿಂದ 3 ಸೆ, ಮಧ್ಯಮ 3 ರಿಂದ 5 ಸೆ, ಭಾರಿ 5 ಸೆ ಮೇಲ್ಪಟ್ಟು |
| `FILLER` | ಹಿಂಜರಿಕೆಯ ಧ್ವನಿಗಳು (ಉಹ್, ಮ್ಮ್, ಆಂ) | ಲಿಪ್ಯಂತರದಲ್ಲಿ ಪದಕೋಶ ಹೊಂದಾಣಿಕೆ + ಚಿಕ್ಕ ಧ್ವನಿ ಭಾಗದ ಅಕೌಸ್ಟಿಕ್ ಪರಿಶೀಲನೆ |
| `REP` | ಪುನರಾವರ್ತಿತ ಧ್ವನಿ, ಅಕ್ಷರ, ಪದ ಅಥವಾ ಎನ್‌-ಗ್ರಾಮ್ | ಮುಂದಿನ ಪದದ ಪೂರ್ವಪ್ರತ್ಯಯವಾದ ತುಣುಕು, ಅಥವಾ ಒಂದೇ ಪದ/ಎನ್‌-ಗ್ರಾಮ್ ಸತತ ಎರಡು ಬಾರಿ |
| `PROL` | ಎಳೆದ ಧ್ವನಿ | ಧ್ವನಿಘಟಕ ಅಥವಾ ಪದದ ಅವಧಿ ಮಾತನಾಡುವವರ ಸಾಮಾನ್ಯಕ್ಕಿಂತ ತುಂಬಾ ಹೆಚ್ಚು |
| `BLOCK` | ಪದದೊಳಗೆ ಅಥವಾ ಮೊದಲು ಮೌನದ ಅಡಚಣೆ | ಪದದೊಳಗಿನ ಮೌನ, ಅಥವಾ ಪದ ಆರಂಭದ ಮುಂಚಿನ ದೀರ್ಘ ಅಂತರ |
| `FALSESTART` | ಅಪೂರ್ಣ ತುಣುಕು, ನಂತರ ಮರುಆರಂಭ | ತುಣುಕಿನ ನಂತರ ಅದೇ ಆರಂಭದ ಮರುಆರಂಭ |

## 6. ಫೋಲ್ಡರ್ ರಚನೆ

```
kannada/
├── agents.md             # ಏಜೆಂಟ್‌ಗೆ ಸ್ಥಿರ ಸೂಚನೆಗಳು ಮತ್ತು ನಿಯಮಗಳು
├── plan.md               # ವಿನ್ಯಾಸ ಮತ್ತು ಮುಂದಿನ ಹಾದಿಯ ವಿವರಣೆ
├── PROGRESS.md           # ಹಂತವಾರು ಪ್ರಗತಿ ಮತ್ತು ಪರಿಶೀಲನೆಯ ದಾಖಲೆ
├── backend/
│   ├── config.py         # ಕೇಂದ್ರ ಸಂರಚನೆ ಮತ್ತು ಬದಲಿಸಬಹುದಾದ ಪ್ಯಾರಾಮೀಟರ್‌ಗಳು
│   ├── cli.py            # CLI ಪ್ರವೇಶ ಬಿಂದು
│   ├── pipeline/
│   │   ├── ingest.py     # ಹಂತ 0: ಆಡಿಯೋ ಸ್ವೀಕಾರ ಮತ್ತು ಸಾಮಾನ್ಯೀಕರಣ
│   │   ├── vad.py        # ಹಂತ 1: ಧ್ವನಿ ಚಟುವಟಿಕೆ ಪತ್ತೆ
│   │   ├── asr.py        # ಹಂತ 2: ಪೂರ್ಣ ಫೈಲ್ ASR ಲಿಪ್ಯಂತರ
│   │   ├── align.py      # ಹಂತ 3: CTC ಫೋರ್ಸ್ಡ್ ಅಲೈನ್‌ಮೆಂಟ್
│   │   ├── classifier.py # ಹಂತ 4: LLM ಅಡಚಣೆ ವರ್ಗೀಕರಣ ಮತ್ತು ಪರ್ಯಾಯ ವಿಧಾನ
│   │   └── splicer.py    # ಹಂತ 5A: ಝೀರೋ-ಕ್ರಾಸಿಂಗ್ ಕತ್ತರಿಸುವಿಕೆ ಮತ್ತು ಕ್ರಾಸ್‌ಫೇಡ್
│   ├── synth/
│   │   └── inject.py     # ಮೌಲ್ಯಮಾಪನಕ್ಕಾಗಿ ಅಡಚಣೆ ಸೇರಿಸುವ ಸಿಂಥಸೈಸರ್
│   ├── eval/
│   │   └── run_eval.py   # ಬೆಂಚ್‌ಮಾರ್ಕ್ ಮೌಲ್ಯಮಾಪನ ರನ್ನರ್
│   ├── tests/            # ಪರೀಕ್ಷಾ ಸೂಟ್
│   └── requirements.txt
├── data/
│   ├── raw/              # ಮೂಲ ಆಡಿಯೋ ಇನ್‌ಪುಟ್ (ರೆಪೊಗೆ ಸೇರಿಸಿಲ್ಲ)
│   ├── synthetic/        # ಕೃತಕವಾಗಿ ರಚಿಸಿದ ಅಡಚಣೆ ಡೇಟಾ (ರೆಪೊಗೆ ಸೇರಿಸಿಲ್ಲ)
│   └── holdout/          # ನಿಜವಾದ ಹಿಂಜರಿಕೆಯ ಪರೀಕ್ಷಾ ಕ್ಲಿಪ್‌ಗಳು (ರೆಪೊಗೆ ಸೇರಿಸಿಲ್ಲ)
└── requirements.txt
```

ಸಂಕ್ಷಿಪ್ತ ಪರಿಚಯ: `agents.md`, `plan.md` ಮತ್ತು `PROGRESS.md` ಕೆಲಸದ ನಿಯಮಗಳು, ವಿನ್ಯಾಸ, ಹಾಗೂ ನಾವು ಏನನ್ನು ನಿರ್ಮಿಸಿ ಪರಿಶೀಲಿಸಿದ್ದೇವೆ ಎಂಬ ದಾಖಲೆಯನ್ನು ಹೊಂದಿವೆ. ಎಲ್ಲಾ ಬದಲಿಸಬಹುದಾದ ಪ್ಯಾರಾಮೀಟರ್‌ಗಳು (ವಿರಾಮದ ಮಿತಿಗಳು, ಕ್ರಾಸ್‌ಫೇಡ್ ಉದ್ದ ಇತ್ಯಾದಿ) `backend/config.py` ನಲ್ಲಿವೆ, ಕೋಡ್‌ನೊಳಗೆ ಅಡಗಿಲ್ಲ. ಆಡಿಯೋ ಮತ್ತು ಡೇಟಾಸೆಟ್‌ಗಳನ್ನು ರೆಪೊಗೆ **ಸೇರಿಸಿಲ್ಲ**.

## 7. ಸೆಟಪ್ ಮತ್ತು ಬಳಕೆ

ನಿಮಗೆ Python 3.10+ ಬೇಕು, ಸಾಧ್ಯವಾದರೆ GPU ಕೂಡ. ನಾವು RTX 5050 (8 GB) ಮೇಲೆ ಅಭಿವೃದ್ಧಿ ಮಾಡಿದ್ದೇವೆ, ಹಾಗಾಗಿ ಸ್ಥಳೀಯವಾಗಿ fp16 ಇನ್‌ಫರೆನ್ಸ್ ಬಳಸುತ್ತೇವೆ, ಭಾರವಾದ ಕೆಲಸಕ್ಕೆ Kaggle ಅಥವಾ Colab. ಇಲ್ಲಿರುವುದೆಲ್ಲವೂ ಉಚಿತ ಮತ್ತು ಮುಕ್ತ-ಮೂಲ (open source).

**ಮೊದಲು ಈಗಿನ ಸ್ಥಿತಿ ನೋಡಿ, ನಂತರ ಇನ್‌ಸ್ಟಾಲ್ ಮಾಡಿ, ಕೊನೆಗೆ ಪರಿಶೀಲಿಸಿ:**

```bash
# 1. ಈಗಿನ ಸ್ಥಿತಿ ಪರಿಶೀಲನೆ
python --version
pip list | grep -i -E "torch|torchaudio|transformers"

# 2. ಇನ್‌ಸ್ಟಾಲ್
pip install -r requirements.txt

# 3. ಬದಲಾದ ಸ್ಥಿತಿ ಪರಿಶೀಲನೆ
pip list | grep -i -E "torch|torchaudio|transformers"
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
```

**ಪೈಪ್‌ಲೈನ್ ಚಲಾಯಿಸಿ** (CLI ಮೂಲಕ). ಮೊದಲು ಸಹಾಯ ಪಠ್ಯ ನೋಡಿ, ನಿಖರ ಆಯ್ಕೆಗಳು ತಿಳಿಯುತ್ತವೆ:

```bash
python backend/cli.py --help
python backend/cli.py <ನಿಮ್ಮ-ಆಯ್ಕೆಗಳು>   # ಆಡಿಯೋ ಒಳಗೆ, ಸ್ವಚ್ಛ ಪಠ್ಯ + ಸ್ವಚ್ಛ ಆಡಿಯೋ ಹೊರಗೆ
```

**ಪರೀಕ್ಷೆ ಮತ್ತು ಬೆಂಚ್‌ಮಾರ್ಕ್ ಚಲಾಯಿಸಿ:**

```bash
python -m pytest backend/tests
python backend/eval/run_eval.py
```

**ಮೌಲ್ಯಮಾಪನ ಡೇಟಾ ರಚನೆ:** `backend/synth/inject.py` ಸ್ವಚ್ಛ ಮಾತಿಗೆ ತಿಳಿದ ಅಡಚಣೆಗಳನ್ನು ಸೇರಿಸುತ್ತದೆ, ಹಾಗಾಗಿ ಅಂಕ ನೀಡಲು ನಮಗೆ ಸರಿಯಾದ ಉತ್ತರ ಮೊದಲೇ ಗೊತ್ತಿರುತ್ತದೆ.

## 8. ಡೇಟಾಸೆಟ್

ನಮ್ಮಲ್ಲಿ **9 ಮಾತನಾಡುವವರು** ಇದ್ದಾರೆ, ಪ್ರತಿಯೊಬ್ಬರಿಂದ **2 ಸೆಟ್‌ಗಳು**.

- **ಸೆಟ್ 1 (ವಿರಾಮ ಮತ್ತು ತುಂಬುಧ್ವನಿ):** ಪ್ರತಿ ಮಾತನಾಡುವವರಿಂದ 2 ರಿಂದ 3 ನಿಮಿಷದ ಒಂದು ರೆಕಾರ್ಡಿಂಗ್, ವಿರಾಮ ಮತ್ತು ತುಂಬುಧ್ವನಿಗಳೊಂದಿಗೆ. ವಿಷಯ: ದಿನಚರಿ, ಹುಟ್ಟೂರು, ಅಥವಾ ಒಂದು ಅಮೂರ್ತ ಕಲ್ಪನೆ.
- **ಸೆಟ್ 2 (ತೊದಲುವ ಮತ್ತು ಅಸ್ಪಷ್ಟ ಮಾತು):** ಪ್ರತಿ ಮಾತನಾಡುವವರಿಗೆ 40 ಅಥವಾ ಹೆಚ್ಚು ವಾಕ್ಯ ಜೋಡಿಗಳು. ಪ್ರತಿ ವಾಕ್ಯವನ್ನು ಒಮ್ಮೆ ಅಸ್ಪಷ್ಟ/ತೊದಲಿ, ಒಮ್ಮೆ ಸ್ಪಷ್ಟವಾಗಿ ರೆಕಾರ್ಡ್ ಮಾಡಿ, ಲಿಪ್ಯಂತರದೊಂದಿಗೆ.

**ಡೇಟಾ ಹೇಗೆ ಜೋಡಿಸಲಾಗಿದೆ:** ಮೂಲ ರೆಕಾರ್ಡಿಂಗ್‌ಗಳು `data/raw/` ನಲ್ಲಿ, `inject.py` ನಿಂದ ರಚಿಸಿದ ಕೃತಕ ಅಡಚಣೆ ಡೇಟಾ `data/synthetic/` ನಲ್ಲಿ, ಮತ್ತು ನಿಜವಾದ ಹಿಂಜರಿಕೆಯ ಪರೀಕ್ಷಾ ಕ್ಲಿಪ್‌ಗಳು `data/holdout/` ನಲ್ಲಿ ಇರುತ್ತವೆ. ಹೋಲ್ಡೌಟ್ ಕ್ಲಿಪ್‌ಗಳನ್ನು ಪ್ರತ್ಯೇಕವಾಗಿಟ್ಟಿದ್ದೇವೆ, ಹಾಗಾಗಿ ವ್ಯವಸ್ಥೆಯನ್ನು ಮೊದಲು ನೋಡದ ನಿಜವಾದ ಮಾತಿನ ಮೇಲೆ ಪರೀಕ್ಷಿಸಬಹುದು. ಈ ಯಾವ ಫೋಲ್ಡರ್‌ಗಳನ್ನೂ ರೆಪೊಗೆ ಸೇರಿಸಿಲ್ಲ.

**ನಾವು ಇದನ್ನು ಹೇಗೆ ಮಾಡಿದೆವು:** ಮೊದಲು ತೊದಲುವ ಸ್ಕ್ರಿಪ್ಟ್ ಬರೆದು ಮಾತನಾಡುವವರಿಂದ ಓದಿಸಿದೆವು, ಹಾಗಾಗಿ ರೆಕಾರ್ಡಿಂಗ್‌ಗೆ ಮೊದಲೇ ಯಥಾವತ್ ಲಿಪ್ಯಂತರ ಮತ್ತು ಸ್ವಚ್ಛ ಉಲ್ಲೇಖ ಸಿದ್ಧವಿದ್ದವು. ನಂತರ ಸ್ವಯಂ-ಟ್ಯಾಗ್ ಮಾಡಿ, Audacity ಅಥವಾ Label Studio ನಲ್ಲಿ ಕೈಯಾರೆ ತಿದ್ದಿ, CSV ರಫ್ತು ಮಾಡಿ, ಮತ್ತೊಂದು ತಂಡದಿಂದ ಪರಿಶೀಲನೆ ಮಾಡಿಸಿದೆವು.

**ಸ್ವರೂಪ:** ಆಡಿಯೋ 16 kHz ಮೊನೊ WAV; ಪಠ್ಯ UTF-8 ಯೂನಿಕೋಡ್ ಕನ್ನಡ (Nudi ಅಥವಾ Baraha ಇಲ್ಲ). ಪ್ರತಿ ತಂಡಕ್ಕೆ ಒಂದು ಮೆಟಾಡೇಟಾ CSV, `team_XX/problem_YY/` ಒಳಗೆ.

## 9. ಮೌಲ್ಯಮಾಪನ

| ಮೆಟ್ರಿಕ್ | ಲೆಕ್ಕಾಚಾರ ಹೇಗೆ |
|----------|----------------|
| ತೆಗೆದ ವಿರಾಮ ಮತ್ತು ತುಂಬುಧ್ವನಿ (%) | (ಮೊದಲಿನ ಟ್ಯಾಗ್‌ಗಳು − ನಂತರ ಉಳಿದ ಟ್ಯಾಗ್‌ಗಳು) / ಮೊದಲಿನ ಟ್ಯಾಗ್‌ಗಳು |
| ಸ್ವಚ್ಛ ಪಠ್ಯದ WER | ಉಲ್ಲೇಖ ಸ್ವಚ್ಛ ಲಿಪ್ಯಂತರದ ವಿರುದ್ಧ ಪದ ದೋಷ ದರ |
| ಟ್ಯಾಗ್ ನಿಖರತೆ / ಮರುಸ್ಮರಣೆ (Precision / Recall) | ಊಹಿಸಿದ ಟ್ಯಾಗ್‌ಗಳು vs ಕೈಯಾರೆ ತಿದ್ದಿದ CSV (ಟ್ಯಾಗ್ ವಿಧ ಮತ್ತು ಸಮಯದ ಅತಿಕ್ರಮಣ ಆಧಾರದ ಮೇಲೆ) |
| ಆಲಿಕೆ ಅಂಕ | 3 ವಿಮರ್ಶಕರಿಂದ 1 ರಿಂದ 5; ಅತಿಯಾದ ಕತ್ತರಿಸುವಿಕೆ, ಕ್ಲಿಕ್ ಶಬ್ದ ಮತ್ತು ರೋಬೋಟ್ ಅಂತರಗಳಿಗೆ ಅಂಕ ಕಡಿತ |

ಇಲ್ಲಿ ಒಂದು ಸಮತೋಲನ ಇದೆ: ಹೆಚ್ಚು ಅಡಚಣೆ ತೆಗೆದರೆ ತೆಗೆದ % ಹೆಚ್ಚುತ್ತದೆ, ಆದರೆ ಅತಿಯಾಗಿ ಕತ್ತರಿಸಿದರೆ WER ಮತ್ತು ಆಲಿಕೆ ಅಂಕ ಕುಸಿಯುತ್ತವೆ. ಹಾಗಾಗಿ ನಾವು ಟ್ಯಾಗ್‌ಗಳಲ್ಲಿ **ಹೆಚ್ಚಿನ ನಿಖರತೆಗೆ (precision)** ಆದ್ಯತೆ ನೀಡುತ್ತೇವೆ.

**ನಮ್ಮ ಫಲಿತಾಂಶಗಳು:**

| ಮೆಟ್ರಿಕ್ | ಅಂಕ |
|----------|-----|
| ತೆಗೆದ ವಿರಾಮ ಮತ್ತು ತುಂಬುಧ್ವನಿ (%) | _ಇಲ್ಲಿ ತುಂಬಿ_ |
| WER (ಸ್ವಚ್ಛ ಪಠ್ಯ) | _ಇಲ್ಲಿ ತುಂಬಿ_ |
| ಟ್ಯಾಗ್ ನಿಖರತೆ / ಮರುಸ್ಮರಣೆ | _ಇಲ್ಲಿ ತುಂಬಿ_ |
| ಆಲಿಕೆ ಅಂಕ (3 ರ ಸರಾಸರಿ) | _ಇಲ್ಲಿ ತುಂಬಿ_ |

## 10. ನೈತಿಕತೆ, ಒಪ್ಪಿಗೆ ಮತ್ತು ಪರವಾನಗಿ

- ಪ್ರತಿಯೊಬ್ಬ ಮಾತನಾಡುವವರು ಮತ್ತು ಬರಹಗಾರರಿಂದ ಸಹಿ ಮಾಡಿದ ಒಪ್ಪಿಗೆ ಪಡೆದಿದ್ದೇವೆ (`consent/` ನೋಡಿ).
- ಮಾತು ಅಭಿನಯಿಸಿದ್ದು ಅಥವಾ ಒಪ್ಪಿಗೆ ನೀಡಿದ ಸ್ವಯಂಸೇವಕರದ್ದು ಮಾತ್ರ.
- ಯಾವುದೇ ಫೈಲ್‌ನಲ್ಲಿ ಫೋನ್ ಸಂಖ್ಯೆ ಅಥವಾ ವಿಳಾಸ ಇಲ್ಲ.
- ವಿಷಯ ಮೂಲ (original) ಆಗಿದೆ: ಹಾಡುಗಳು, ಚಲನಚಿತ್ರಗಳು ಅಥವಾ ಪುಸ್ತಕಗಳ ಪಠ್ಯ ಇಲ್ಲ.
- ಕ್ಲೋನ್ ಮಾಡಿದ ಧ್ವನಿಗಳಿಗೆ ಸ್ಪಷ್ಟ ಒಪ್ಪಿಗೆ ಬೇಕು, ಮತ್ತು ಎಲ್ಲಾ ಕೃತಕ ಮಾದರಿಗಳನ್ನು ಮೆಟಾಡೇಟಾದಲ್ಲಿ ಗುರುತಿಸಲಾಗಿದೆ.
- ಕೊಡುಗೆದಾರರಿಗೆ ಕ್ರೆಡಿಟ್ ನೀಡಿ, **CC-BY 4.0** ಅಡಿಯಲ್ಲಿ ಮುಕ್ತವಾಗಿ ಬಿಡುಗಡೆ ಮಾಡುವ ಯೋಜನೆ ಇದೆ.

## 11. ತಿಳಿದಿರುವ ಮಿತಿಗಳು

ಇವುಗಳ ಬಗ್ಗೆ ನಾವು ಮುಕ್ತವಾಗಿ ಹೇಳುತ್ತೇವೆ:

- ASR ಪ್ರಮಾಣಿತ ಕನ್ನಡದ ಕಡೆಗೆ ವಾಲುತ್ತದೆ, ಪ್ರಾದೇಶಿಕ ಉಚ್ಚಾರಣೆಗಳಲ್ಲಿ ಕಡಿಮೆ ನಿಖರ.
- ತುಂಬಾ ಚಿಕ್ಕ ತುಣುಕುಗಳಲ್ಲಿ ಅಲೈನ್‌ಮೆಂಟ್ ಅಸ್ಥಿರವಾಗಿರುತ್ತದೆ.
- CTC ಅಕ್ಷರ ಮಾದರಿಗಳು ಪುನರಾವರ್ತಿತ ಅಕ್ಷರಗಳನ್ನು ಬಿಡಬಹುದು ಅಥವಾ ವಿಲೀನಗೊಳಿಸಬಹುದು, ಆದ್ದರಿಂದ ತಿಳಿದ ಸ್ಕ್ರಿಪ್ಟ್ ವಿರುದ್ಧ ಪರಿಶೀಲಿಸುತ್ತೇವೆ.
- PROL, BLOCK ಮತ್ತು FALSESTART ಪತ್ತೆ ಮಾಡಲು ಅತ್ಯಂತ ಕಷ್ಟ; ಅಲ್ಲಿ ಮರುಸ್ಮರಣೆ (recall) ಕಡಿಮೆ ಇರುತ್ತದೆ.
- ಅತಿಯಾಗಿ ಕತ್ತರಿಸಿದರೆ ಮಾತು ತುಂಡುತುಂಡಾಗಿ ಕೇಳುತ್ತದೆ, ಹಾಗಾಗಿ ನಾವು ಎಚ್ಚರಿಕೆಯಿಂದ ಕತ್ತರಿಸುತ್ತೇವೆ.

## 12. ತಂಡ ಮತ್ತು ಕೃತಜ್ಞತೆಗಳು

- **ತಂಡ:** _Team ಕನ್ನಡಕ್ಕಾಗಿ ಒಂದನ್ನು ಒತ್ತಿ
- **ಬಳಸಿದ ಉಪಕರಣಗಳು:** Silero VAD, MMS (facebook/mms-1b-all), AI4Bharat IndicConformer, torchaudio, DeepFilterNet / RNNoise / noisereduce, Indic Parler-TTS, Audacity, Label Studio.
- ನಮ್ಮ ಸ್ಕ್ರಿಪ್ಟ್‌ಗಳನ್ನು ಓದಿ ತಮ್ಮ ಧ್ವನಿ ನೀಡಿದ ಎಲ್ಲಾ ಸ್ವಯಂಸೇವಕರಿಗೆ ಧನ್ಯವಾದಗಳು.
