"""Notification: turns a classified event into an alert and delivers it through a channel.

The module separates **what is said** from **where it is sent**:

- `build_alert()` builds an `Alert` (ready text + JPEGs) from the `Event` and the `Analysis`.
  Nothing here knows what Telegram is.
- `Notifier` is a channel's contract: `send(alert) -> bool`, which **never raises** —
  anything inheriting from `BaseNotifier` gets that guarantee for free.

Channels are registered in `CHANNELS`; `notify.channel` picks one. Today there is only
`simulated`, which only logs the alert — the frames already live in `data/events` (storage).
A new channel is a `BaseNotifier` subclass plus one line in `CHANNELS`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from .classifier import Analysis
from .config import RiskLevel, Settings
from .event import Event

log = logging.getLogger(__name__)

EMOJI = {RiskLevel.LOW: "🟢", RiskLevel.MEDIUM: "🟡", RiskLevel.HIGH: "🔴"}

#: Text length limit. It is Telegram's caption limit (1024); it works as a common ceiling
#: for every channel, since the alert is the same everywhere.
MAX_TEXT = 1024


@dataclass(frozen=True, slots=True)
class Alert:
    """Alert ready to send — text and images, nothing channel-specific."""

    risk: RiskLevel | None  # None = operational notice (heartbeat, camera offline)
    text: str
    images: tuple[bytes, ...]
    happened_at: datetime  # local capture time (notify.timezone)
    available: bool = True  # False = the AI did not classify

    @classmethod
    def status(cls, text: str, when: datetime) -> Alert:
        """Notice about the service itself, text only — goes out through the alert channel."""
        return cls(risk=None, text=_truncate(text, MAX_TEXT), images=(), happened_at=when)


def build_alert(
    event: Event,
    analysis: Analysis,
    *,
    timezone: str = "UTC",
    max_images: int = 4,
) -> Alert:
    """Builds the event's alert: the first `max_images` images, in chronological order."""
    when = datetime.fromtimestamp(event.started_at, ZoneInfo(timezone))
    return Alert(
        risk=analysis.risk,
        text=format_text(analysis, when),
        images=tuple(image.data for image in event.images[:max_images]),
        happened_at=when,
        available=analysis.available,
    )


def format_text(analysis: Analysis, when: datetime) -> str:
    """Alert text: emoji + risk + time, reason, description and confidence."""
    lines = [
        f"{EMOJI[analysis.risk]} Risk {analysis.risk.value.upper()} — {when:%Y-%m-%d %H:%M}",
        analysis.reason,
    ]
    if analysis.description:
        lines.append(analysis.description)
    if analysis.available:
        footer = f"confidence {analysis.confidence:.0%}"
        if analysis.people:
            footer += f" · {analysis.people} person(s)"
        lines.append(footer)
    return _truncate("\n".join(lines), MAX_TEXT)


class Notifier(Protocol):
    """A notification channel's contract."""

    name: str

    def send(self, alert: Alert) -> bool:
        """Delivers the alert. Returns whether it worked; **does not raise** on failure."""
        ...


class BaseNotifier(ABC):
    """Base for channels: turns any failure in `_deliver` into `False` + a clean log line.

    `redact` lists secrets (bot token, API key) that must vanish from the error message
    before it is logged — HTTP libraries tend to include the URL, and Telegram's URL
    carries the token.
    """

    name = "base"

    def __init__(self, *, redact: Iterable[str] = ()) -> None:
        self._redact = tuple(s for s in redact if s)

    def send(self, alert: Alert) -> bool:
        try:
            ok = self._deliver(alert)
        except Exception as err:  # noqa: BLE001 - a channel failure never takes down the pipeline
            log.error("channel %s failed: %s: %s", self.name, type(err).__name__, self._clean(err))
            return False
        if not ok:
            log.error("channel %s did not deliver the alert", self.name)
        return ok

    @abstractmethod
    def _deliver(self, alert: Alert) -> bool:
        """The actual delivery. May raise: `send()` handles it."""

    def _clean(self, err: Exception) -> str:
        text = str(err)
        for secret in self._redact:
            text = text.replace(secret, "***")
        return text[:MAX_TEXT]


class SimulatedNotifier(BaseNotifier):
    """Fake channel: logs the alert and writes nothing — runs the whole pipeline with no
    bot and no network. The event's frames are already in `data/events` (storage)."""

    name = "simulated"

    def _deliver(self, alert: Alert) -> bool:
        log.info("[simulated] alert with %d image(s):\n%s", len(alert.images), alert.text)
        return True


def _simulated(_settings: Settings) -> SimulatedNotifier:
    return SimulatedNotifier()


#: Available channels, keyed by `notify.channel`.
CHANNELS = {SimulatedNotifier.name: _simulated}


def build_notifier(settings: Settings) -> Notifier:
    """Instantiates the channel chosen in `notify.channel`."""
    channel = settings.notify.channel
    if channel not in CHANNELS:
        options = ", ".join(sorted(CHANNELS))
        raise ValueError(f"unknown channel '{channel}'; available: {options}")
    return CHANNELS[channel](settings)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
