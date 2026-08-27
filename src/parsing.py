"""
Parsing and validation of model output.

Local models are less reliable at instruction-following than hosted ones, so
this module assumes the output is messy and recovers what it can. Anything that
cannot be parsed into a schema-valid label is returned as a failure rather than
being silently coerced to a default, because a silent default becomes label
noise that is indistinguishable from a genuine model judgement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any

from .prompts import DISCLOSURE_TYPES, DIRECTNESS


# Matches the outermost {...} block. Non-greedy would stop at the first closing
# brace, which breaks on nested objects, so this is deliberately greedy.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Label:
    is_disclosure: bool
    disclosure_type: str
    directness: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParseFailure:
    reason: str
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.reason, "raw": self.raw[:500]}


def extract_json(text: str) -> dict | None:
    """Pull the first plausible JSON object out of a model response."""
    if not text:
        return None

    # Strip markdown fences, which some models add despite instructions.
    text = re.sub(r"```(?:json)?", "", text)

    # Chain-of-thought responses put reasoning before a LABEL: marker.
    if "LABEL:" in text:
        text = text.rsplit("LABEL:", 1)[1]

    match = _JSON_BLOCK.search(text)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Trailing commas are the most common malformation. One repair attempt,
        # then give up rather than escalating to increasingly loose heuristics.
        repaired = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def validate(obj: dict) -> Label | ParseFailure:
    """Check a parsed object against the schema. No silent defaults."""
    required = {"is_disclosure", "disclosure_type", "directness", "confidence"}
    missing = required - obj.keys()
    if missing:
        return ParseFailure(f"missing keys: {sorted(missing)}", json.dumps(obj))

    is_disclosure = _coerce_bool(obj["is_disclosure"])
    if is_disclosure is None:
        return ParseFailure(
            f"is_disclosure not boolean: {obj['is_disclosure']!r}", json.dumps(obj)
        )

    dtype = str(obj["disclosure_type"]).strip().lower()
    if dtype not in DISCLOSURE_TYPES:
        return ParseFailure(f"bad disclosure_type: {dtype!r}", json.dumps(obj))

    directness = str(obj["directness"]).strip().lower()
    if directness not in DIRECTNESS:
        return ParseFailure(f"bad directness: {directness!r}", json.dumps(obj))

    try:
        confidence = float(obj["confidence"])
    except (TypeError, ValueError):
        return ParseFailure(f"confidence not numeric: {obj['confidence']!r}",
                            json.dumps(obj))
    if not 0.0 <= confidence <= 1.0:
        return ParseFailure(f"confidence out of range: {confidence}", json.dumps(obj))

    # Cross-field consistency. These combinations are contradictory, and a model
    # that produces them has not understood the codebook for that post.
    if is_disclosure and dtype == "none":
        return ParseFailure("is_disclosure=true with type 'none'", json.dumps(obj))
    if not is_disclosure and dtype != "none":
        return ParseFailure(f"is_disclosure=false with type {dtype!r}", json.dumps(obj))
    if not is_disclosure and directness != "not_applicable":
        return ParseFailure(
            f"is_disclosure=false with directness {directness!r}", json.dumps(obj)
        )
    if is_disclosure and directness == "not_applicable":
        return ParseFailure("is_disclosure=true with directness 'not_applicable'",
                            json.dumps(obj))

    return Label(is_disclosure, dtype, directness, confidence)


def parse_response(text: str) -> Label | ParseFailure:
    """Full pipeline: raw model output to validated label or explicit failure."""
    obj = extract_json(text)
    if obj is None:
        return ParseFailure("no parsable JSON in response", text or "")
    return validate(obj)