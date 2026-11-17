"""Stage 3: CTC Forced Alignment.

Aligns verbatim Kannada transcript tokens with input audio at word level
using CTC forced alignment (e.g., torchaudio MMS Kannada aligner),
returning precise word start/end times and inter-word gaps.
"""
