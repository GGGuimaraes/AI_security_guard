"""Classification — only strings in and out: no network, no AI, no camera."""

import json

import pytest

from ia_watcher.classifier import (
    MAX_DESCRIPTION,
    MAX_REASON,
    REASON_UNAVAILABLE,
    Analysis,
    classify,
    unavailable,
)
from ia_watcher.config import RiskLevel

VALID = {
    "risk": "high",
    "reason": "Person tampering with the door lock.",
    "description": "Someone handles the door for several frames and looks around.",
    "confidence": 0.9,
    "people": 1,
}


def _content(**overrides) -> str:
    return json.dumps({**VALID, **overrides}, ensure_ascii=False)


# --------------------------------------------------------------- valid responses


def test_valid_response_becomes_a_full_analysis():
    a = classify(_content())
    assert a == Analysis(
        risk=RiskLevel.HIGH,
        reason="Person tampering with the door lock.",
        description="Someone handles the door for several frames and looks around.",
        confidence=0.9,
        people=1,
        available=True,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("medium", RiskLevel.MEDIUM),
        ("MEDIUM", RiskLevel.MEDIUM),
        ("  High ", RiskLevel.HIGH),
        ("LOW", RiskLevel.LOW),
    ],
)
def test_risk_case_and_spacing_are_normalized(raw, expected):
    assert classify(_content(risk=raw)).risk is expected


def test_json_wrapped_in_markdown_and_text():
    content = f"Sure! Here is the analysis:\n```json\n{_content()}\n```\nHope this helps."
    a = classify(content)
    assert a.risk is RiskLevel.HIGH
    assert a.available is True


def test_json_followed_by_loose_text():
    a = classify(_content() + "\n\nNote: night-time image.")
    assert a.risk is RiskLevel.HIGH
    assert a.people == 1


# ------------------------------------------------- lenient secondary fields


def test_valid_risk_without_reason_is_not_downgraded():
    """Downgrading a 'high' for lack of a reason would be worse than a generic reason."""
    a = classify(json.dumps({"risk": "high"}))
    assert a.risk is RiskLevel.HIGH
    assert a.available is True
    assert a.reason  # never empty
    assert a.confidence == 0.0
    assert a.people == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1.7, 1.0), (-0.5, 0.0), ("0.4", 0.4), ("very high", 0.0), (None, 0.0)],
)
def test_out_of_range_confidence_is_clamped(raw, expected):
    assert classify(_content(confidence=raw)).confidence == expected


@pytest.mark.parametrize("raw", ["two", None, -3, 2.9])
def test_invalid_people_does_not_break_the_analysis(raw):
    a = classify(_content(people=raw))
    assert a.risk is RiskLevel.HIGH
    assert a.people >= 0


def test_huge_text_is_truncated():
    a = classify(_content(reason="a" * 5000, description="b" * 5000))
    assert len(a.reason) <= MAX_REASON
    assert len(a.description) <= MAX_DESCRIPTION
    assert a.risk is RiskLevel.HIGH


# ------------------------------------------------------------- invalid responses


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        "I could not analyze the images",
        "{risk: high}",  # not JSON
        '{"risk": "high"',  # truncated JSON
        '["high"]',  # not an object
        '"high"',
        "{}",  # no risk
        _content(risk="critical"),  # outside the enum
        _content(risk="URGENT"),
        _content(risk=3),
        _content(risk=None),
    ],
)
def test_invalid_response_falls_back_to_medium(content):
    a = classify(content)
    assert a.risk is RiskLevel.MEDIUM
    assert a.reason == REASON_UNAVAILABLE
    assert a.available is False


# ------------------------------------------------------- AI unavailable (fallback)


def test_unavailable_gives_medium_with_detail_in_description():
    a = unavailable("HTTP 500 upstream error")
    assert a.risk is RiskLevel.MEDIUM
    assert a.reason == REASON_UNAVAILABLE
    assert a.available is False
    assert "HTTP 500" in a.description


def test_unavailable_accepts_the_exception_directly():
    from ia_watcher.ai_client import AIUnavailable

    a = unavailable(AIUnavailable("3 attempts failed"))
    assert a.risk is RiskLevel.MEDIUM
    assert "attempts" in a.description


def test_unavailable_without_detail():
    a = unavailable()
    assert a.risk is RiskLevel.MEDIUM
    assert a.available is False


# ----------------------------------------------------------------- notify decision


@pytest.mark.parametrize(
    ("risk", "min_risk", "expected"),
    [
        (RiskLevel.LOW, RiskLevel.LOW, True),
        (RiskLevel.LOW, RiskLevel.MEDIUM, False),
        (RiskLevel.MEDIUM, RiskLevel.MEDIUM, True),
        (RiskLevel.MEDIUM, RiskLevel.HIGH, False),
        (RiskLevel.HIGH, RiskLevel.HIGH, True),
        (RiskLevel.HIGH, RiskLevel.LOW, True),
    ],
)
def test_should_notify_compares_with_min_risk(risk, min_risk, expected):
    a = classify(_content(risk=risk.value))
    assert a.should_notify(min_risk) is expected


def test_fallback_notifies_with_the_default_setting():
    """An AI failure must not silence an event: medium >= the default min_risk."""
    assert unavailable("timeout").should_notify(RiskLevel.MEDIUM) is True


def test_analysis_is_immutable():
    a = classify(_content())
    with pytest.raises(AttributeError):
        a.risk = RiskLevel.LOW
