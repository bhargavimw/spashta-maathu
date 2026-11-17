"""Evaluation runner for Kannada Pause Remover and Stammer Cleaner.

Calculates benchmark metrics:
- Character Error Rate (CER) on Unicode NFC normalized clean transcripts
- Disfluency recall (% injected disfluencies detected and removed)
- False-removal rate (% non-injected words wrongly dropped)
- Duration and pause compression statistics
- Splice click check (automated energy spike detection at join points)
"""
