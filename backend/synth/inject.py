"""Synthetic Disfluency Injector.

Generates realistic hesitant Kannada audio from clean speech by injecting:
- Word repetitions (preferring modified repetitions with jitter/pitch/stretch)
- False starts (truncated word audio, ~150 ms)
- Fillers (natural recorded Kannada filler tokens)
- Disfluent pauses (200-400 ms mid-clause pauses)

Produces paired datasets: (hesitant.wav, clean.wav, hesitant_text, clean_text, labels.json).
"""
