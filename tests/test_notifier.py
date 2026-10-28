"""Notification — no network: alert assembly, formatting and the simulated channel."""

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ai_security_guard.classifier import REASON_UNAVAILABLE, Analysis, unavailable
from ai_security_guard.config import RiskLevel, load_settings
from ai_security_guard.event import Event, EventImage
from ai_security_guard.notifier import (
    EMOJI,
    MAX_TEXT,
    Alert,
    BaseNotifier,
    SimulatedNotifier,
    build_alert,
    build_notifier,
    format_text,
)

# 2026-09-20 03:14:15 in America/Sao_Paulo (UTC-3)
TS = datetime.fromisoformat("2026-09-20T03:14:15-03:00").timestamp()
TZ = "America/Sao_Paulo"

ANALYSIS = Analysis(
    risk=RiskLevel.HIGH,
    reason="Person tampering with the lock.",
    description="Someone handles the door for several frames.",
    confidence=0.9,
    people=1,
)


def _event(n: int = 3) -> Event:
    images = tuple(EventImage(data=f"jpeg{i}".encode(), ts=TS + i) for i in range(n))
    return Event(started_at=TS, images=images, area_ratio=0.05)


def _alert(analysis: Analysis = ANALYSIS, n: int = 3, max_images: int = 4) -> Alert:
    return build_alert(_event(n), analysis, timezone=TZ, max_images=max_images)


# ------------------------------------------------------------------ alert assembly


def test_alert_uses_the_local_capture_time():
    alert = _alert()
    assert alert.happened_at.hour == 3 and alert.happened_at.minute == 14
    assert alert.risk is RiskLevel.HIGH
    assert alert.available is True


def test_alert_honors_max_images_and_keeps_the_order():
    alert = _alert(n=6, max_images=2)
    assert alert.images == (b"jpeg0", b"jpeg1")


def test_alert_without_images_is_still_valid():
    alert = _alert(n=3, max_images=0)
    assert alert.images == ()
    assert alert.text


# ------------------------------------------------------------------ text


def test_text_has_emoji_risk_reason_and_time():
    text = _alert().text
    assert text.startswith(EMOJI[RiskLevel.HIGH])
    assert "HIGH" in text
    assert "Person tampering with the lock." in text
    assert "2026-09-20 03:14" in text


@pytest.mark.parametrize("risk", list(RiskLevel))
def test_each_risk_has_its_own_emoji(risk):
    text = format_text(Analysis(risk=risk, reason="x"), datetime.now())
    assert text.startswith(EMOJI[risk])


def test_ai_failure_shows_in_the_text():
    alert = _alert(unavailable("timeout"))
    assert alert.available is False
    assert REASON_UNAVAILABLE in alert.text


def test_text_is_truncated_at_the_channel_limit():
    long = Analysis(risk=RiskLevel.MEDIUM, reason="m", description="x" * 2000)
    text = format_text(long, datetime.now())
    assert len(text) <= MAX_TEXT


# ------------------------------------------------------------------ simulated channel


def test_simulated_only_logs_and_writes_nothing(tmp_path: Path, monkeypatch, caplog):
    """The frames are already in data/events (storage); the channel doesn't duplicate them."""
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO, logger="ai_security_guard.notifier"):
        assert SimulatedNotifier().send(_alert()) is True
    assert caplog.text.count(EMOJI[RiskLevel.HIGH]) == 1
    assert "Person tampering with the lock." in caplog.text
    assert "3 image(s)" in caplog.text
    assert not list(tmp_path.iterdir())


def test_status_notice_is_logged_without_images(caplog):
    when = datetime.fromtimestamp(TS, ZoneInfo(TZ))
    notice = Alert.status("✅ ai_security_guard is running", when)
    assert notice.risk is None and notice.images == ()
    with caplog.at_level(logging.INFO, logger="ai_security_guard.notifier"):
        assert SimulatedNotifier().send(notice) is True
    assert "✅ ai_security_guard is running" in caplog.text


# ------------------------------------------------------------------ channel contract


class _Broken(BaseNotifier):
    """A channel that fails the way the network fails — with the secret in the message."""

    name = "broken"

    def _deliver(self, alert: Alert) -> bool:
        raise RuntimeError("failed to POST https://api.example/bot123:SECRET-TOKEN/send")


def test_channel_failure_does_not_propagate_or_leak_the_secret(caplog):
    notifier = _Broken(redact=["123:SECRET-TOKEN"])
    with caplog.at_level(logging.ERROR, logger="ai_security_guard.notifier"):
        assert notifier.send(_alert()) is False
    assert "SECRET-TOKEN" not in caplog.text
    assert "broken" in caplog.text


# ------------------------------------------------------------------ factory


def test_factory_returns_the_configured_channel(example_config, fake_env):
    settings = load_settings(example_config, env=fake_env)
    notifier = build_notifier(settings)
    assert isinstance(notifier, SimulatedNotifier)
    assert notifier.name == "simulated"


def test_unknown_channel_fails_listing_the_available_ones(example_config, fake_env):
    settings = load_settings(example_config, env=fake_env)
    settings.notify.channel = "carrier-pigeon"
    with pytest.raises(ValueError, match="simulated"):
        build_notifier(settings)
