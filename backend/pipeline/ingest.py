"""Stage 0: Audio Ingest and Normalization.

Ingests user audio, resamples to 16 kHz mono float32, and performs peak normalization.
Asserts audio shape and sample rate to avoid silent downsampling or clipping.
"""
