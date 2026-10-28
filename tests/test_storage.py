"""Persistence — SQLite and JPEGs on disk. No network, no camera, everything in tmp_path."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from ai_security_guard.ai_client import AIResponse, Usage
from ai_security_guard.classifier import Analysis, unavailable
from ai_security_guard.config import RiskLevel, load_settings
from ai_security_guard.event import Event, EventImage
from ai_security_guard.storage import Storage

TZ = "America/Sao_Paulo"
# 2026-09-20 03:14:15 in São Paulo (UTC-3)
TS = datetime.fromisoformat("2026-09-20T03:14:15-03:00").timestamp()
NOW = datetime.fromisoformat("2026-09-20T12:00:00-03:00").timestamp()
HOUR = 3600.0

ANALYSIS = Analysis(
    risk=RiskLevel.HIGH,
    reason="Person tampering with the lock.",
    description="Someone handles the door for several frames.",
    confidence=0.9,
    people=1,
)
RESPONSE = AIResponse(
    content='{"risk": "high"}',
    model="google/gemini-2.5-flash",
    usage=Usage(prompt_tokens=1500, completion_tokens=60, cached_tokens=1200, cost=0.00042),
    latency_seconds=2.5,
)


# --------------------------------------------------------------------------- helpers


def _event(ts: float = TS, n: int = 3, area_ratio: float = 0.05) -> Event:
    images = tuple(EventImage(data=f"jpeg{i}".encode(), ts=ts + i) for i in range(n))
    return Event(started_at=ts, images=images, area_ratio=area_ratio)


def _storage(tmp_path: Path, *, now: float = NOW, **kwargs) -> Storage:
    options = dict(
        db_path=tmp_path / "ai_security_guard.db",
        events_dir=tmp_path / "events",
        keep_hours=12,
        timezone=TZ,
        clock=lambda: now,
    )
    options.update(kwargs)
    return Storage(**options)


def _rows(db_path: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        return con.execute("SELECT * FROM events ORDER BY id").fetchall()


# --------------------------------------------------------------------------- writing


def test_save_event_writes_jpegs_to_the_day_folder(tmp_path: Path):
    with _storage(tmp_path) as storage:
        stored = storage.save_event(_event())

    assert stored.dir.parent == tmp_path / "events" / "2026-09-20"
    assert sorted(p.name for p in stored.dir.glob("*.jpg")) == ["1.jpg", "2.jpg", "3.jpg"]
    assert (stored.dir / "1.jpg").read_bytes() == b"jpeg0"


def test_save_event_creates_a_row_with_what_the_detector_saw(tmp_path: Path):
    with _storage(tmp_path) as storage:
        stored = storage.save_event(_event())

    (row,) = _rows(tmp_path / "ai_security_guard.db")
    assert row["id"] == stored.id
    assert row["started_at"] == pytest.approx(TS)
    assert row["day"] == "2026-09-20"
    assert row["images"] == 3
    assert row["area_ratio"] == pytest.approx(0.05)
    assert row["dir"].endswith(stored.dir.name)
    assert row["risk"] is None  # the analysis hasn't happened yet
    assert row["notified"] is None


def test_day_folder_uses_the_configured_timezone(tmp_path: Path):
    """23:30 in São Paulo is already the next day in UTC — the folder follows notify.timezone."""
    late = datetime.fromisoformat("2026-09-20T23:30:00-03:00").timestamp()

    with _storage(tmp_path) as sp:
        assert sp.save_event(_event(ts=late)).dir.parent.name == "2026-09-20"
    with _storage(tmp_path, timezone="UTC", db_path=tmp_path / "utc.db") as utc:
        assert utc.save_event(_event(ts=late)).dir.parent.name == "2026-09-21"


def test_events_in_the_same_second_do_not_overwrite_each_other(tmp_path: Path):
    with _storage(tmp_path) as storage:
        first = storage.save_event(_event())
        second = storage.save_event(_event())

    assert first.dir != second.dir
    assert len(list((tmp_path / "events" / "2026-09-20").iterdir())) == 2


# --------------------------------------------------------------------------- analysis


def test_save_analysis_records_risk_tokens_cost_and_latency(tmp_path: Path):
    with _storage(tmp_path) as storage:
        stored = storage.save_event(_event())
        storage.save_analysis(stored.id, ANALYSIS, RESPONSE)

    (row,) = _rows(tmp_path / "ai_security_guard.db")
    assert row["risk"] == "high"
    assert row["reason"] == "Person tampering with the lock."
    assert row["confidence"] == pytest.approx(0.9)
    assert row["people"] == 1
    assert row["available"] == 1
    assert row["model"] == "google/gemini-2.5-flash"
    assert row["prompt_tokens"] == 1500
    assert row["cached_tokens"] == 1200
    assert row["cost"] == pytest.approx(0.00042)
    assert row["latency_seconds"] == pytest.approx(2.5)


def test_unavailable_analysis_is_recorded_without_tokens(tmp_path: Path):
    """The AI failed, but the event must not vanish from the database."""
    with _storage(tmp_path) as storage:
        stored = storage.save_event(_event())
        storage.save_analysis(stored.id, unavailable("timeout"), None)

    (row,) = _rows(tmp_path / "ai_security_guard.db")
    assert row["risk"] == "medium"
    assert row["available"] == 0
    assert row["prompt_tokens"] is None
    assert row["cost"] is None


@pytest.mark.parametrize("ok", [True, False])
def test_mark_notified_records_the_delivery_result(tmp_path: Path, ok: bool):
    with _storage(tmp_path) as storage:
        stored = storage.save_event(_event())
        storage.mark_notified(stored.id, ok)

    (row,) = _rows(tmp_path / "ai_security_guard.db")
    assert row["notified"] == int(ok)


def test_reopening_the_database_keeps_the_events(tmp_path: Path):
    with _storage(tmp_path) as storage:
        storage.save_event(_event())
    with _storage(tmp_path) as storage:  # the schema is idempotent
        storage.save_event(_event())

    assert len(_rows(tmp_path / "ai_security_guard.db")) == 2


# --------------------------------------------------------------------------- cleanup


def test_cleanup_removes_what_is_past_the_window_and_keeps_the_rest(tmp_path: Path):
    with _storage(tmp_path, keep_hours=12) as storage:
        old = storage.save_event(_event(ts=NOW - 20 * HOUR))
        recent = storage.save_event(_event(ts=NOW - 2 * HOUR))

        removed = storage.cleanup()

    assert removed == 1
    assert not old.dir.exists()
    assert recent.dir.exists()
    assert [row["id"] for row in _rows(tmp_path / "ai_security_guard.db")] == [recent.id]


def test_cleanup_cuts_by_the_hour_within_the_same_day(tmp_path: Path):
    """12h doesn't fit day granularity: today's folder holds both cases."""
    with _storage(tmp_path, keep_hours=6) as storage:  # cutoff at 06:00
        morning = storage.save_event(_event(ts=NOW - 10 * HOUR))  # 02:00
        recent = storage.save_event(_event(ts=NOW - 2 * HOUR))  # 10:00
        assert morning.dir.parent == recent.dir.parent  # same day

        assert storage.cleanup() == 1

    assert not morning.dir.exists()
    assert recent.dir.exists()
    assert recent.dir.parent.exists()  # the day folder survives: it still has an event


def test_cleanup_removes_the_day_folder_when_it_empties(tmp_path: Path):
    with _storage(tmp_path, keep_hours=12) as storage:
        old = storage.save_event(_event(ts=NOW - 30 * HOUR))
        storage.cleanup()

    assert not old.dir.parent.exists()


def test_cleanup_ignores_folders_that_are_not_a_day(tmp_path: Path):
    stray = tmp_path / "events" / "not-a-date"
    stray.mkdir(parents=True)

    with _storage(tmp_path) as storage:
        assert storage.cleanup() == 0

    assert stray.exists()


def test_cleanup_ignores_event_folders_with_an_unexpected_name(tmp_path: Path):
    stray = tmp_path / "events" / "2026-09-01" / "notes"
    stray.mkdir(parents=True)

    with _storage(tmp_path, keep_hours=1) as storage:
        storage.cleanup()

    assert stray.exists()  # unknown age: when in doubt, don't delete


def test_cleanup_with_nothing_stored_does_not_fail(tmp_path: Path):
    with _storage(tmp_path) as storage:
        assert storage.cleanup() == 0


# --------------------------------------------------------------------------- factory


def test_from_settings_uses_the_configured_paths(example_config, fake_env, tmp_path: Path):
    settings = load_settings(example_config, env=fake_env)
    settings.storage.db_path = tmp_path / "ai_security_guard.db"
    settings.storage.events_dir = tmp_path / "events"

    with Storage.from_settings(settings) as storage:
        assert storage.keep_hours == settings.storage.keep_hours
        assert storage.events_dir == tmp_path / "events"
        assert storage.save_event(_event()).dir.exists()
