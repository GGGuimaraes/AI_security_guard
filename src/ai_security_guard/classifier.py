"""Validates the AI response and decides whether to notify.

`ai_client.py` returns raw text; here it becomes a trustworthy `Analysis`. Two rules
govern this stage:

- **Nothing here raises.** An unreadable response or an AI outage becomes `medium` with
  the reason "analysis unavailable" — an AI failure must never silence an event.
- **Never downgrade a risk the model did state.** If `risk` is valid, bad secondary
  fields (no reason, absurd `confidence`) are corrected, not discarded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .config import RiskLevel

log = logging.getLogger(__name__)

#: Reason used whenever the classification did not come from the AI.
REASON_UNAVAILABLE = "analysis unavailable"
REASON_MISSING = "no reason given by the AI"
MAX_REASON = 200
MAX_DESCRIPTION = 1000

_DECODER = json.JSONDecoder()


@dataclass(frozen=True, slots=True)
class Analysis:
    """Validated result of an event — what goes to the alert and to SQLite."""

    risk: RiskLevel
    reason: str
    description: str = ""
    confidence: float = 0.0
    people: int = 0
    available: bool = True  # False = fallback, the AI did not classify

    def should_notify(self, min_risk: RiskLevel) -> bool:
        return self.risk >= min_risk


def classify(content: str) -> Analysis:
    """Turns the AI text into an `Analysis`. Never raises: anything unreadable becomes `medium`."""
    data = _extract_json(content)
    if data is None:
        return unavailable("response has no valid JSON")
    risk = _risk(data.get("risk"))
    if risk is None:
        return unavailable(f"invalid risk: {str(data.get('risk'))[:40]!r}")
    return Analysis(
        risk=risk,
        reason=_text(data.get("reason"), MAX_REASON) or REASON_MISSING,
        description=_text(data.get("description"), MAX_DESCRIPTION),
        confidence=_confidence(data.get("confidence")),
        people=_count(data.get("people")),
    )


def unavailable(reason: object = "") -> Analysis:
    """Safety fallback: `medium` risk with a warning, so the event is not missed."""
    detail = _text(str(reason) if reason else "", MAX_DESCRIPTION)
    log.warning("classification unavailable: %s", detail or "no detail")
    return Analysis(
        risk=RiskLevel.MEDIUM,
        reason=REASON_UNAVAILABLE,
        description=detail,
        available=False,
    )


def _extract_json(content: str) -> dict | None:
    """First JSON object in the text — tolerates ```json, a greeting before, a comment after."""
    if not isinstance(content, str):
        return None
    for start in _starts(content):
        try:
            data, _ = _DECODER.raw_decode(content, start)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _starts(content: str):
    index = content.find("{")
    while index != -1:
        yield index
        index = content.find("{", index + 1)


def _risk(value: object) -> RiskLevel | None:
    if not isinstance(value, str):
        return None
    try:
        return RiskLevel.parse(value)
    except ValueError:
        return None


def _text(value: object, limit: int) -> str:
    """Clean, truncated text — a runaway model can't flood the caption or the database."""
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _count(value: object) -> int:
    try:
        return max(0, int(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
