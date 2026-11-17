
import json
import os
import warnings

from dotenv import load_dotenv
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Tolerance for ASR/VAD timestamp jitter (seconds).
TIMING_TOLERANCE = 0.05

NonNegInt = Annotated[int, Field(ge=0)]
NonNegSeconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def _require_strictly_ascending(values: list[int], label: str) -> None:
    for prev, curr in zip(values, values[1:]):
        if curr <= prev:
            raise ValueError(
                f"{label} must be unique and strictly ascending; "
                f"got {prev} followed by {curr}"
            )


class SpeechSegment(BaseModel):
    """One immutable speech segment from Module 2."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    index: NonNegInt
    text: str
    start: NonNegSeconds
    end: NonNegSeconds
    gap_after: NonNegSeconds

    @model_validator(mode="after")
    def _check_times_and_text(self) -> "SpeechSegment":
        if not self.text.strip():
            raise ValueError(f"segment {self.index}: text must not be empty")
        if self.start > self.end:
            raise ValueError(
                f"segment {self.index}: start ({self.start}) is greater than end ({self.end})"
            )
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class ClassifierInput(BaseModel):
    """Ordered list of segments passed to the classifier."""

    model_config = ConfigDict(frozen=True)

    segments: list[SpeechSegment]

    @classmethod
    def from_list(cls, raw: list[dict[str, Any]]) -> "ClassifierInput":
        return cls(segments=raw)

    @model_validator(mode="after")
    def _check_segments(self) -> "ClassifierInput":
        _require_strictly_ascending(
            [s.index for s in self.segments], "segment indices"
        )

        for prev, curr in zip(self.segments, self.segments[1:]):
            if curr.start < prev.end - TIMING_TOLERANCE:
                raise ValueError(
                    f"segment {curr.index} starts ({curr.start}) before "
                    f"segment {prev.index} ends ({prev.end})"
                )

            actual_gap = curr.start - prev.end
            if abs(prev.gap_after - actual_gap) > TIMING_TOLERANCE:
                raise ValueError(
                    f"segment {prev.index}: gap_after ({prev.gap_after}) does not match "
                    f"actual gap to segment {curr.index} ({actual_gap:.3f})"
                )

        return self

    def by_index(self) -> dict[int, SpeechSegment]:
        return {s.index: s for s in self.segments}


class RemovalReason(str, Enum):
    STAMMER = "stammer"
    FALSE_START = "false_start"
    FILLER = "filler"
    REPETITION = "repetition"
    UNNECESSARY_PAUSE = "unnecessary_pause"
    OTHER = "other"


class RemovedSegment(BaseModel):
    """A removed segment: index and reason only."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    index: NonNegInt
    reason: RemovalReason


class _ClassificationCore(BaseModel):
    """Shared fields and validation for classifier output."""

    kept_segments: list[NonNegInt]
    pause_after_kept: list[NonNegSeconds]
    removed: list[RemovedSegment]

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            unknown = set(data) - set(cls.model_fields)
            if unknown:
                raise ValueError(
                    f"unexpected field(s) not allowed: {sorted(unknown)}"
                )
        return data

    @model_validator(mode="after")
    def _check_structure(self) -> "_ClassificationCore":
        removed_indices = [r.index for r in self.removed]

        _require_strictly_ascending(self.kept_segments, "kept_segments")
        _require_strictly_ascending(removed_indices, "removed indices")

        overlap = set(self.kept_segments) & set(removed_indices)
        if overlap:
            raise ValueError(
                f"indices cannot be both kept and removed: {sorted(overlap)}"
            )

        expected = max(len(self.kept_segments) - 1, 0)
        if len(self.pause_after_kept) != expected:
            raise ValueError(
                f"pause_after_kept needs exactly {expected} entries; "
                f"got {len(self.pause_after_kept)}"
            )

        return self

    def validate_against_input(self, source: ClassifierInput) -> None:
        segments = source.by_index()
        classified = set(self.kept_segments) | {r.index for r in self.removed}

        unknown = classified - segments.keys()
        if unknown:
            raise ValueError(f"indices not present in input: {sorted(unknown)}")

        missing = segments.keys() - classified
        if missing:
            raise ValueError(
                f"segments neither kept nor removed: {sorted(missing)}"
            )

        if segments and not self.kept_segments:
            raise ValueError("at least one segment must be kept")

        for i, pause in enumerate(self.pause_after_kept):
            a = segments[self.kept_segments[i]]
            b = segments[self.kept_segments[i + 1]]
            available = b.start - a.end

            if pause > available + TIMING_TOLERANCE:
                raise ValueError(
                    f"pause_after_kept[{i}] ({pause}s) exceeds the original "
                    f"gap between segments {a.index} and {b.index}"
                )


class LLMClassification(_ClassificationCore):
    """Exactly what the LLM may return. No text field."""


class ClassifierOutput(_ClassificationCore):
    """Final Module 3 output consumed by Module 4."""

    clean_text: str

    @classmethod
    def from_llm(
        cls, llm: LLMClassification, source: ClassifierInput
    ) -> "ClassifierOutput":
        llm.validate_against_input(source)

        return cls(
            **llm.model_dump(),
            clean_text=assemble_clean_text(llm.kept_segments, source),
        )

    def validate_against_input(self, source: ClassifierInput) -> None:
        super().validate_against_input(source)

        expected = assemble_clean_text(self.kept_segments, source)
        if self.clean_text != expected:
            raise ValueError(
                "clean_text must contain the original text of kept segments; "
                f"expected {expected!r}, got {self.clean_text!r}"
            )


def assemble_clean_text(
    kept_segments: list[int], source: ClassifierInput
) -> str:
    """Build clean text from original retained segments."""

    segments = source.by_index()
    return " ".join(segments[i].text.strip() for i in kept_segments)


def _build_prompt(source: ClassifierInput) -> str:
    """Create the Kannada disfluency classification prompt."""

    segments = [
        {
            "index": s.index,
            "text": s.text,
            "start": s.start,
            "end": s.end,
            "gap_after": s.gap_after,
        }
        for s in source.segments
    ]

    return f"""
You are a Kannada speech-disfluency classifier.

Identify which speech segments should be retained and which
should be removed to produce smooth, natural speech.

RULES:
1. Remove stammers, incomplete false starts, filler words,
   and accidental repeated words or syllables.
2. Preserve the speaker's original meaning.
3. Preserve meaningful repetitions used for emphasis.
   Example: "ಇಲ್ಲ, ಇಲ್ಲ, ಅದು ಸರಿಯಲ್ಲ."
4. Preserve natural pauses between clauses.
5. Do not invent, rewrite, correct, or paraphrase words.
6. Every input segment must appear exactly once:
   either in kept_segments or removed.
7. Use the original segment indices.
8. pause_after_kept must contain one pause between each
   pair of consecutive retained segments.
9. Pauses must be non-negative and fit within the
   available original timeline gap.

ALLOWED REMOVAL REASONS:
stammer, false_start, filler, repetition, unnecessary_pause, other

INPUT SEGMENTS:
{json.dumps(segments, ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
    "kept_segments": [0, 2, 3],
    "pause_after_kept": [0.25, 0.40],
    "removed": [
        {{"index": 1, "reason": "false_start"}}
    ]
}}

Do not include clean_text. The application generates it.
"""


def _rule_based_fallback(source: ClassifierInput) -> ClassifierOutput:
    """Remove the second segment in an adjacent exact repetition."""

    kept = []
    removed = []

    for segment in source.segments:
        if (
            kept
            and segment.text.strip().casefold()
            == kept[-1].text.strip().casefold()
        ):
            removed.append(
                RemovedSegment(
                    index=segment.index,
                    reason=RemovalReason.REPETITION,
                )
            )
        else:
            kept.append(segment)

    pauses = [
        max(0.0, min(0.25, second.start - first.end))
        for first, second in zip(kept, kept[1:])
    ]

    result = LLMClassification(
        kept_segments=[s.index for s in kept],
        pause_after_kept=pauses,
        removed=removed,
    )

    return ClassifierOutput.from_llm(result, source)


def classify_segments(
    raw_segments: list[dict],
    api_key: str | None = None,
    model: str = "gemini-3.8-flash",
) -> ClassifierOutput:
    """
    Main Module 3 entry point.

    Accepts Module 2 segments and returns validated Module 3 output.
    Uses rule-based fallback if Gemini fails.
    """

    source = ClassifierInput.from_list(raw_segments)

    if not source.segments:
        return ClassifierOutput(
            kept_segments=[],
            pause_after_kept=[],
            removed=[],
            clean_text="",
        )

    load_dotenv()
    api_key = api_key or os.getenv("GEMINI_API_KEY")

    if not api_key:
        warnings.warn(
            "GEMINI_API_KEY not found. Using rule-based fallback."
        )
        return _rule_based_fallback(source)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=15000),
        )

        response = client.models.generate_content(
            model=model,
            contents=_build_prompt(source),
            config={
                "response_mime_type": "application/json",
            },
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        llm_data = json.loads(response.text)
        llm_result = LLMClassification.model_validate(llm_data)

        print("CLASSIFICATION SOURCE: GEMINI")
        return ClassifierOutput.from_llm(llm_result, source)

    except Exception as exc:
        warnings.warn(
            f"Gemini classification failed: {exc}. "
            "Using rule-based fallback."
        )
        print("CLASSIFICATION SOURCE: FALLBACK")
        return _rule_based_fallback(source)
