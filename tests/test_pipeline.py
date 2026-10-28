"""Orchestration — the full pipeline with a fake source, fake AI and fake channel.

No network, no camera and no real sleeping: the `EventBuilder` gets a fake clock and
sleep, and the source tells the pipeline when it runs out, so `run()` ends on its own.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from ai_security_guard.ai_client import AIResponse, AIUnavailable, Usage
from ai_security_guard.capture import Frame
from ai_security_guard.classifier import REASON_UNAVAILABLE
from ai_security_guard.config import RiskLevel, load_settings
from ai_security_guard.event import Event, EventBuilder, EventImage
from ai_security_guard.motion import MIN_WARMUP, MotionDetector
from ai_security_guard.notifier import Alert, SimulatedNotifier
from ai_security_guard.pipeline import Pipeline
from ai_security_guard.storage import Storage

TZ = "America/Sao_Paulo"
TS = datetime.fromisoformat("2026-09-20T03:14:15-03:00").timestamp()
BG_SHAPE = (240, 320, 3)

RESPONSE_HIGH = (
    '{"risk": "high", "reason": "Person handling the door.",'
    ' "description": "Someone at the entrance.", "confidence": 0.9, "people": 1}'
)
RESPONSE_LOW = (
    '{"risk": "low", "reason": "Cat walking by.",'
    ' "description": "Small animal.", "confidence": 0.8, "people": 0}'
)


# --------------------------------------------------------------------------- scene


def _background(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(40, 80, size=BG_SHAPE, dtype=np.uint8)


def _with_square(base: np.ndarray, x: float, value: int = 230) -> np.ndarray:
    """Pastes a bright object on the background, at horizontal position `x` (normalized)."""
    h, w = base.shape[:2]
    out = base.copy()
    out[int(0.35 * h) : int(0.6 * h), int(x * w) : int((x + 0.2) * w)] = value
    return out


def _walking(base: np.ndarray, frames: int) -> list[np.ndarray]:
    """An object crossing the scene. Standing still it would become background within a
    few frames — as really happens with MOG2, which is adaptive."""
    return [_with_square(base, 0.2 + 0.05 * i) for i in range(frames)]


def _scene(*, still: int = 8, moving: int = 4) -> list[np.ndarray]:
    """Background frames (warm-up), then someone walking."""
    base = _background()
    return [base] * still + _walking(base, moving)


# --------------------------------------------------------------------------- fakes


class FakeClock:
    """Manual clock: `sleep` advances time instead of waiting."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeSource:
    """Delivers a list of images and then runs dry, telling whoever wants to know."""

    name = "fake://camera"

    def __init__(self, images: list[np.ndarray]) -> None:
        self.images = images
        self.on_exhausted = None
        self.seq = 0
        self.started = False
        self.stopped = False

    def start(self) -> "FakeSource":
        self.started = True
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True

    def wait_for_frame(self, timeout: float = 5.0, after: int = 0) -> Frame | None:
        if self.seq >= len(self.images):
            if self.on_exhausted is not None:
                self.on_exhausted()
            return None
        image = self.images[self.seq]
        self.seq += 1
        return Frame(image=image, ts=TS + self.seq * 0.1, seq=self.seq)


class FakeAI:
    """Fake AI: always returns the same content, or raises the configured error."""

    def __init__(self, content: str = RESPONSE_HIGH, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[Event] = []
        self.closed = False

    def analyze(self, event: Event) -> AIResponse:
        self.calls.append(event)
        if self.error is not None:
            raise self.error
        return AIResponse(
            content=self.content,
            model="fake/model",
            usage=Usage(prompt_tokens=1500, cached_tokens=1200, cost=0.0004),
            latency_seconds=0.3,
        )

    def close(self) -> None:
        self.closed = True


class FakeNotifier:
    name = "fake"

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return self.ok


class BrokenStorage:
    """Full disk / locked database: nothing storage does may silence an alert."""

    keep_hours = 12

    def save_event(self, event: Event):
        raise OSError("disk full")

    def save_analysis(self, *args) -> None:
        raise sqlite3.OperationalError("database is locked")

    def mark_notified(self, *args) -> None:
        raise sqlite3.OperationalError("database is locked")

    def cleanup(self) -> int:
        raise OSError("disk full")

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- assembly


def _event(n: int = 2) -> Event:
    images = tuple(EventImage(data=f"jpeg{i}".encode(), ts=TS + i) for i in range(n))
    return Event(started_at=TS, images=images, area_ratio=0.05)


def _pipeline(
    tmp_path: Path,
    *,
    images: list[np.ndarray] | None = None,
    ai: object | None = None,
    notifier: object | None = None,
    storage: object | None = None,
    **kwargs,
) -> Pipeline:
    clock = FakeClock()
    source = FakeSource(images if images is not None else _scene())
    builder = EventBuilder(
        source,
        frames_per_event=2,
        frame_interval_seconds=1.0,
        cooldown_seconds=60.0,
        clock=clock,
        sleep=clock.sleep,
    )
    options = dict(
        source=source,
        detector=MotionDetector(warmup_frames=MIN_WARMUP, min_area_ratio=0.01),
        builder=builder,
        ai=ai if ai is not None else FakeAI(),
        notifier=notifier if notifier is not None else FakeNotifier(),
        storage=storage
        if storage is not None
        else Storage(
            db_path=tmp_path / "ai_security_guard.db",
            events_dir=tmp_path / "events",
            timezone=TZ,
        ),
        min_risk=RiskLevel.MEDIUM,
        trigger_frames=2,
        timezone=TZ,
    )
    options.update(kwargs)
    return Pipeline(**options)


def _rows(tmp_path: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(tmp_path / "ai_security_guard.db") as con:
        con.row_factory = sqlite3.Row
        return con.execute("SELECT * FROM events ORDER BY id").fetchall()


def _feed(pipeline: Pipeline, source: FakeSource) -> list[Event]:
    """Runs only the capture/motion stage until the source runs dry."""
    events = []
    while (frame := source.wait_for_frame()) is not None:
        if (event := pipeline.on_frame(frame)) is not None:
            events.append(event)
    return events


# --------------------------------------------------------------------------- motion


def test_sustained_motion_opens_one_event(tmp_path: Path):
    pipeline = _pipeline(tmp_path)
    events = _feed(pipeline, pipeline.source)

    assert len(events) == 1
    assert len(events[0].images) == 2
    assert events[0].area_ratio > 0.01


def test_a_single_motion_frame_does_not_open_an_event(tmp_path: Path):
    """Noise crosses the threshold for one frame; two in a row, it doesn't."""
    base = _background()
    scene = [base] * 8 + [_with_square(base, 0.3)] + [base] * 3
    pipeline = _pipeline(tmp_path, images=scene)

    assert _feed(pipeline, pipeline.source) == []


def test_cooldown_prevents_a_second_event(tmp_path: Path):
    pipeline = _pipeline(tmp_path, images=_scene(still=8, moving=12))

    assert len(_feed(pipeline, pipeline.source)) == 1


def test_debounce_off_opens_an_event_on_the_first_frame(tmp_path: Path):
    pipeline = _pipeline(tmp_path, images=_scene(moving=3), trigger_frames=1)

    assert len(_feed(pipeline, pipeline.source)) == 1


# --------------------------------------------------------------------------- analysis


def test_analysis_notifies_and_records_in_the_database(tmp_path: Path):
    notifier = FakeNotifier()
    pipeline = _pipeline(tmp_path, notifier=notifier)

    analysis = pipeline.analyze(_event())

    assert analysis.risk is RiskLevel.HIGH
    (alert,) = notifier.alerts
    assert alert.risk is RiskLevel.HIGH
    assert "Person handling the door." in alert.text

    (row,) = _rows(tmp_path)
    assert row["risk"] == "high"
    assert row["cached_tokens"] == 1200
    assert row["notified"] == 1
    assert Path(row["dir"]).exists()


def test_risk_below_the_minimum_is_recorded_but_not_notified(tmp_path: Path):
    notifier = FakeNotifier()
    pipeline = _pipeline(tmp_path, ai=FakeAI(RESPONSE_LOW), notifier=notifier)

    pipeline.analyze(_event())

    assert notifier.alerts == []
    (row,) = _rows(tmp_path)
    assert row["risk"] == "low"
    assert row["notified"] is None  # not even attempted: below notify.min_risk


def test_ai_outage_notifies_as_medium(tmp_path: Path):
    """An AI failure does not silence the event."""
    notifier = FakeNotifier()
    pipeline = _pipeline(tmp_path, ai=FakeAI(error=AIUnavailable("HTTP 503")), notifier=notifier)

    analysis = pipeline.analyze(_event())

    assert analysis.risk is RiskLevel.MEDIUM
    assert analysis.available is False
    (alert,) = notifier.alerts
    assert REASON_UNAVAILABLE in alert.text
    (row,) = _rows(tmp_path)
    assert row["available"] == 0
    assert row["notified"] == 1


def test_unexpected_ai_error_also_notifies_as_medium(tmp_path: Path, caplog):
    """Any client exception hits the same fallback, not just AIUnavailable."""
    pipeline = _pipeline(tmp_path, ai=FakeAI(error=RuntimeError("bug")))
    notifier = pipeline.notifier

    with caplog.at_level(logging.ERROR, logger="ai_security_guard.pipeline"):
        analysis = pipeline.analyze(_event())

    assert analysis.risk is RiskLevel.MEDIUM
    assert len(notifier.alerts) == 1


def test_channel_failure_does_not_crash_and_is_recorded(tmp_path: Path):
    pipeline = _pipeline(tmp_path, notifier=FakeNotifier(ok=False))

    pipeline.analyze(_event())

    (row,) = _rows(tmp_path)
    assert row["notified"] == 0


def test_write_failure_does_not_block_the_alert(tmp_path: Path, caplog):
    """The same goes for the disk: a broken database must not silence the notification."""
    notifier = FakeNotifier()
    pipeline = _pipeline(tmp_path, storage=BrokenStorage(), notifier=notifier)

    with caplog.at_level(logging.ERROR, logger="ai_security_guard.pipeline"):
        pipeline.analyze(_event())

    assert len(notifier.alerts) == 1


# --------------------------------------------------------------------------- run/stop


def test_run_goes_from_the_source_to_the_alert_saving_only_in_events(tmp_path: Path, caplog):
    pipeline = _pipeline(tmp_path, notifier=SimulatedNotifier())
    pipeline.source.on_exhausted = pipeline.stop

    with caplog.at_level(logging.INFO, logger="ai_security_guard.notifier"):
        pipeline.run()

    assert "Person handling the door." in caplog.text
    (row,) = _rows(tmp_path)
    assert row["risk"] == "high"
    assert row["notified"] == 1
    assert sorted(p.name for p in Path(row["dir"]).glob("*.jpg")) == ["1.jpg", "2.jpg"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ai_security_guard.db", "events"]
    assert pipeline.stats.events == 1
    assert pipeline.stats.alerts == 1


def test_run_shuts_down_the_source_and_the_worker(tmp_path: Path):
    pipeline = _pipeline(tmp_path)
    pipeline.source.on_exhausted = pipeline.stop

    pipeline.run()

    assert pipeline.source.started is True
    assert pipeline.source.stopped is True
    assert pipeline.ai.closed is True


def test_run_cleans_up_old_history_at_startup(tmp_path: Path):
    old = tmp_path / "events" / "2020-01-01"
    old.mkdir(parents=True)
    pipeline = _pipeline(tmp_path, images=[])
    pipeline.source.on_exhausted = pipeline.stop

    pipeline.run()

    assert not old.exists()


def test_worker_error_does_not_crash_the_pipeline(tmp_path: Path, caplog):
    pipeline = _pipeline(tmp_path, storage=BrokenStorage())
    pipeline.source.on_exhausted = pipeline.stop

    with caplog.at_level(logging.ERROR, logger="ai_security_guard.pipeline"):
        pipeline.run()

    assert pipeline.stats.events == 1
    assert len(pipeline.notifier.alerts) == 1


def test_lost_source_warns_only_once(tmp_path: Path, caplog):
    pipeline = _pipeline(tmp_path)

    with caplog.at_level(logging.WARNING, logger="ai_security_guard.pipeline"):
        pipeline.on_source_lost()
        pipeline.on_source_lost()  # camera still offline: no second warning

    assert sum("no frames" in r.getMessage() for r in caplog.records) == 1


# --------------------------------------------------------------------------- factory


def test_from_settings_builds_the_pipeline_from_the_config(
    example_config, fake_env, tmp_path: Path
):
    settings = load_settings(example_config, env=fake_env)
    settings.storage.db_path = tmp_path / "ai_security_guard.db"
    settings.storage.events_dir = tmp_path / "events"

    pipeline = Pipeline.from_settings(settings)
    try:
        assert pipeline.min_risk is settings.notify.min_risk
        assert pipeline.trigger_frames == settings.motion.trigger_frames
        assert isinstance(pipeline.notifier, SimulatedNotifier)
        assert pipeline.builder.frames_per_event == settings.event.frames_per_event
    finally:
        pipeline.close()


def test_from_settings_does_not_leak_the_camera_password(example_config, fake_env, tmp_path: Path):
    settings = load_settings(example_config, env=fake_env)
    settings.storage.db_path = tmp_path / "ai_security_guard.db"
    settings.storage.events_dir = tmp_path / "events"

    pipeline = Pipeline.from_settings(settings)
    try:
        assert fake_env["TAPO_PASSWORD"] not in repr(pipeline)
        assert fake_env["TAPO_PASSWORD"] not in pipeline.source.name
    finally:
        pipeline.close()


# --------------------------------------------------------------------------- health


class FakeHealth:
    def __init__(self) -> None:
        self.started = False
        self.checks: list[float | None] = []

    def start(self) -> None:
        self.started = True

    def check(self, last_frame_at: float | None, summary: object = "") -> None:
        self.checks.append(last_frame_at)


def test_run_feeds_the_health_monitor_the_last_frame(tmp_path: Path):
    health = FakeHealth()
    pipeline = _pipeline(tmp_path, health=health)
    pipeline.source.on_exhausted = pipeline.stop
    pipeline.run()

    assert health.started
    assert health.checks, "the loop must consult the health monitor"
    assert health.checks[0] == TS + 0.1  # timestamp of the first frame
    # the source ran dry: the monitor keeps seeing the last frame, not a made-up "now"
    assert health.checks[-1] == health.checks[-2] > health.checks[0]


def test_health_monitor_failure_does_not_stop_the_watch(tmp_path: Path):
    class Broken(FakeHealth):
        def check(self, last_frame_at, summary=""):
            raise RuntimeError("bug")

    notifier = FakeNotifier()
    pipeline = _pipeline(tmp_path, health=Broken(), notifier=notifier)
    pipeline.source.on_exhausted = pipeline.stop
    pipeline.run()
    assert len(notifier.alerts) == 1  # the scene's event still became an alert
