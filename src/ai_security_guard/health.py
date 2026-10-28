"""24/7 service health: reports through the channel that it is alive, or the camera is gone.

An alert that never arrives looks exactly like a quiet night. So the service speaks up
on its own from time to time:

- **startup**: "ai_security_guard is running" as soon as it comes up — confirms the deploy and
  exposes restarts;
- **daily**: "ai_security_guard is running" once a day at `health.daily_at` (local time);
- **camera offline**: no frames for more than `health.camera_offline_minutes`, plus another
  notice when it comes back.

Notices are `Alert.status` and go out through the same `Notifier` as the risk alerts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from datetime import time as dtime
from typing import Self
from zoneinfo import ZoneInfo

from .config import Settings
from .notifier import Alert, Notifier

log = logging.getLogger(__name__)


class HealthMonitor:
    """Called on every pipeline loop iteration; `check()` is cheap when all is well."""

    def __init__(
        self,
        notifier: Notifier,
        *,
        timezone: str = "UTC",
        daily_at: dtime | None = dtime(8, 0),
        offline_after_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.notifier = notifier
        self.tz = ZoneInfo(timezone)
        self.daily_at = daily_at
        self.offline_after_seconds = offline_after_seconds
        self._clock = clock
        self._started_at: float | None = None
        self._next_daily: float | None = None
        self._offline_since: float | None = None  # last frame before it went silent

    @classmethod
    def from_settings(cls, settings: Settings, notifier: Notifier) -> Self:
        return cls(
            notifier,
            timezone=settings.notify.timezone,
            daily_at=settings.health.daily_at,
            offline_after_seconds=settings.health.camera_offline_minutes * 60,
        )

    def start(self, summary: object = "") -> None:
        """Marks the startup and sends the first "ai_security_guard is running"."""
        now = self._clock()
        self._started_at = now
        self._next_daily = self._following_daily(now)
        self._send(f"✅ ai_security_guard is running — {self._fmt(now)}", summary)

    def check(self, last_frame_at: float | None, summary: object = "") -> None:
        """Sends the daily heartbeat and the camera offline/back notices when it's time."""
        if self._started_at is None:
            self.start(summary)
        now = self._clock()
        self._check_camera(now, last_frame_at)
        if self._next_daily is not None and now >= self._next_daily:
            self._next_daily = self._following_daily(now)
            self._send(f"✅ ai_security_guard is running — {self._fmt(now)}", summary)

    def _check_camera(self, now: float, last_frame_at: float | None) -> None:
        reference = last_frame_at if last_frame_at is not None else self._started_at
        silent = now - reference
        if self._offline_since is None:
            if silent > self.offline_after_seconds:
                self._offline_since = reference
                self._send(
                    f"⚠️ camera offline — no images for {_minutes(silent)} min "
                    f"(since {self._fmt(reference)})"
                )
        elif silent <= self.offline_after_seconds:
            gap = reference - self._offline_since
            self._offline_since = None
            self._send(f"✅ camera back — {_minutes(gap)} min without images")

    def _following_daily(self, now: float) -> float | None:
        """Next local `daily_at` strictly after `now` (days spent stopped count as one)."""
        if self.daily_at is None:
            return None
        today = datetime.fromtimestamp(now, self.tz).date()
        moment = datetime.combine(today, self.daily_at, self.tz)
        if moment.timestamp() <= now:
            moment = datetime.combine(today + timedelta(days=1), self.daily_at, self.tz)
        return moment.timestamp()

    def _send(self, text: str, summary: object = "") -> None:
        extra = str(summary)
        if extra:
            text = f"{text}\n{extra}"
        log.info("health notice: %s", text.replace("\n", " | "))
        self.notifier.send(Alert.status(text, datetime.fromtimestamp(self._clock(), self.tz)))

    def _fmt(self, ts: float) -> str:
        return f"{datetime.fromtimestamp(ts, self.tz):%Y-%m-%d %H:%M}"


def _minutes(seconds: float) -> int:
    return round(seconds / 60)
