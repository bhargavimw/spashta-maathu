"""Stage 4a: Annotation CSV writer.

Produces PAUSE tags from Stage 1 VAD gaps per SPEC.md §3 tag taxonomy:

| Tag | Duration | Label |
|-----|----------|-------|
| (none) | < 1 s | — |
| PAUSE | 1–3 s | light |
| PAUSE | 3–5 s | medium |
| PAUSE | > 5 s | heavy |

CSV schema: file,tag,start_s,end_s,text
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from backend.pipeline.vad import AudioGap, VADResult

PAUSE_TAG = "PAUSE"

# Duration thresholds in seconds (SPEC.md §3).
LIGHT_MIN = 1.0
MEDIUM_MIN = 3.0
HEAVY_MIN = 5.0


@dataclass(frozen=True)
class AnnotationRow:
    """Single row in the annotation CSV."""

    file: str
    tag: str
    start_s: float
    end_s: float
    text: str


def _pause_severity(duration_sec: float) -> Optional[str]:
    """Return severity label for a gap duration, or None if below 1 s threshold."""
    if duration_sec < LIGHT_MIN:
        return None
    if duration_sec < MEDIUM_MIN:
        return "light"
    if duration_sec < HEAVY_MIN:
        return "medium"
    return "heavy"


def tag_pauses(vad_result: VADResult, audio_filename: str) -> List[AnnotationRow]:
    """Tag VAD gaps as PAUSE with severity based on duration.

    Gaps under 1 s are not tagged (returned as empty).
    """
    rows: List[AnnotationRow] = []
    for gap in vad_result.gaps:
        severity = _pause_severity(gap.duration_sec)
        if severity is None:
            continue
        rows.append(
            AnnotationRow(
                file=audio_filename,
                tag=PAUSE_TAG,
                start_s=round(gap.start_sec, 6),
                end_s=round(gap.end_sec, 6),
                text=severity,
            )
        )
    return rows


def write_annotation_csv(
    rows: List[AnnotationRow],
    output_path: Path,
) -> None:
    """Write annotation rows to CSV with header file,tag,start_s,end_s,text."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "tag", "start_s", "end_s", "text"])
        for row in rows:
            writer.writerow(
                [row.file, row.tag, f"{row.start_s:.6f}", f"{row.end_s:.6f}", row.text]
            )


def tag_and_write_pauses(
    vad_result: VADResult,
    audio_filename: str,
    output_path: Path,
) -> List[AnnotationRow]:
    """Convenience: tag VAD gaps and write annotation CSV in one call."""
    rows = tag_pauses(vad_result, audio_filename)
    write_annotation_csv(rows, output_path)
    return rows
