"""Event persistence: one row in SQLite and the JPEGs on disk.

The database is the history used to review false positives, measure AI cost and tune
the criteria. It keeps **every** analyzed event, including those that did not become an
alert because they were below `notify.min_risk`.

Layout::

    data/ia_watcher.db                                  one row per event
    data/events/2026-09-20/031415-12/1.jpg ...     the frames the AI saw

This is the only place where images of the home are written. `cleanup()` deletes
whatever is older than `storage.keep_hours`. The cutoff is by the hour, and each folder's
timestamp comes from its own path.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo

from .ai_client import AIResponse
from .classifier import Analysis
from .config import Settings
from .event import Event

log = logging.getLogger(__name__)

DAY_FORMAT = "%Y-%m-%d"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        REAL NOT NULL,   -- epoch seconds of the first frame
    day               TEXT NOT NULL,   -- YYYY-MM-DD in notify.timezone
    dir               TEXT,            -- folder with the JPEGs
    images            INTEGER NOT NULL,
    area_ratio        REAL,
    risk              TEXT,            -- NULL until the AI answers
    reason            TEXT,
    description       TEXT,
    confidence        REAL,
    people            INTEGER,
    available         INTEGER,         -- 0 = fallback, the AI did not classify
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cached_tokens     INTEGER,
    cost              REAL,            -- US$ as estimated by the provider
    latency_seconds   REAL,
    notified          INTEGER          -- NULL = not attempted | 0 = failed | 1 = sent
);
CREATE INDEX IF NOT EXISTS events_day ON events (day);
"""


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """Where the event ended up: the database row and the frames folder."""

    id: int
    dir: Path


class Storage:
    """SQLite + JPEGs. Writes come from the worker and cleanup from the main loop, so the
    connection is shared between threads under a lock (sqlite dislikes concurrency).
    """

    def __init__(
        self,
        *,
        db_path: Path | str,
        events_dir: Path | str,
        keep_hours: int = 12,
        timezone: str = "UTC",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db_path = Path(db_path)
        self.events_dir = Path(events_dir)
        self.keep_hours = keep_hours
        self.tz = ZoneInfo(timezone)
        self._clock = clock
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        self._con.executescript(_SCHEMA)
        self._con.commit()

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> Self:
        s = settings.storage
        options = dict(
            db_path=s.db_path,
            events_dir=s.events_dir,
            keep_hours=s.keep_hours,
            timezone=settings.notify.timezone,
        )
        options.update(kwargs)
        return cls(**options)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._con.close()

    # ---------------------------------------------------------------- writing

    def save_event(self, event: Event) -> StoredEvent:
        """Creates the event row and writes the frames. The analysis comes later."""
        when = datetime.fromtimestamp(event.started_at, self.tz)
        day = when.strftime(DAY_FORMAT)
        with self._lock:
            cursor = self._con.execute(
                "INSERT INTO events (started_at, day, images, area_ratio) VALUES (?, ?, ?, ?)",
                (event.started_at, day, len(event.images), event.area_ratio),
            )
            event_id = int(cursor.lastrowid)
            # the id in the name keeps the folder unique without relying on the clock
            target = self.events_dir / day / f"{when:%H%M%S}-{event_id}"
            target.mkdir(parents=True, exist_ok=True)
            for index, image in enumerate(event.images, start=1):
                (target / f"{index}.jpg").write_bytes(image.data)
            self._con.execute("UPDATE events SET dir = ? WHERE id = ?", (str(target), event_id))
            self._con.commit()
        log.debug("event %d written to %s", event_id, target)
        return StoredEvent(id=event_id, dir=target)

    def save_analysis(
        self, event_id: int, analysis: Analysis, response: AIResponse | None = None
    ) -> None:
        """Completes the row with what the AI said and what it cost (None = the AI failed)."""
        usage = response.usage if response else None
        self._write(
            """UPDATE events SET risk = ?, reason = ?, description = ?, confidence = ?,
                   people = ?, available = ?, model = ?, prompt_tokens = ?,
                   completion_tokens = ?, cached_tokens = ?, cost = ?, latency_seconds = ?
               WHERE id = ?""",
            (
                analysis.risk.value,
                analysis.reason,
                analysis.description,
                analysis.confidence,
                analysis.people,
                int(analysis.available),
                response.model if response else None,
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
                usage.cached_tokens if usage else None,
                usage.cost if usage else None,
                response.latency_seconds if response else None,
                event_id,
            ),
        )

    def mark_notified(self, event_id: int, ok: bool) -> None:
        """Records whether the alert went out — delivery failures stay visible in the history."""
        self._write("UPDATE events SET notified = ? WHERE id = ?", (int(ok), event_id))

    def _write(self, sql: str, params: tuple) -> None:
        with self._lock:
            self._con.execute(sql, params)
            self._con.commit()

    # ---------------------------------------------------------------- cleanup

    def cleanup(self) -> int:
        """Deletes whatever is older than `keep_hours`. Returns the rows removed."""
        cutoff = self._clock() - self.keep_hours * 3600
        with self._lock:
            cursor = self._con.execute("DELETE FROM events WHERE started_at < ?", (cutoff,))
            self._con.commit()
            removed = cursor.rowcount
        _purge_older(self.events_dir, cutoff, self.tz)
        if removed:
            log.info("cleanup: removed %d event(s) older than %dh", removed, self.keep_hours)
        return removed


def _purge_older(root: Path, cutoff: float, tz: ZoneInfo) -> None:
    """Removes event folders older than the cutoff, and any day folder left empty.

    The timestamp comes from the path itself (`YYYY-MM-DD/HHMMSS-...`), so cleanup also
    reaches folders with no database row (an orphan event). A folder whose name doesn't
    follow the pattern stays: no age, no delete.
    """
    if not root.is_dir():
        return
    for day_dir in root.iterdir():
        if not day_dir.is_dir() or not _is_day(day_dir.name):
            continue
        for event_dir in day_dir.iterdir():
            stamp = _stamp(day_dir.name, event_dir.name, tz)
            if event_dir.is_dir() and stamp is not None and stamp < cutoff:
                shutil.rmtree(event_dir, ignore_errors=True)
                log.debug("cleanup: removed folder %s", event_dir)
        with contextlib.suppress(OSError):
            day_dir.rmdir()  # only goes if empty; otherwise it still holds events


def _stamp(day: str, name: str, tz: ZoneInfo) -> float | None:
    """Timestamp that `YYYY-MM-DD/HHMMSS-...` stands for; None if it doesn't match."""
    try:
        when = datetime.strptime(f"{day} {name[:6]}", f"{DAY_FORMAT} %H%M%S")
    except ValueError:
        return None
    return when.replace(tzinfo=tz).timestamp()


def _is_day(name: str) -> bool:
    try:
        datetime.strptime(name, DAY_FORMAT)
    except ValueError:
        return False
    return True
