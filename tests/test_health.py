"""Service health — daily heartbeat and camera offline, with a fake clock and a fake channel."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from ia_watcher.health import HealthMonitor
from ia_watcher.notifier import Alert

TZ = "America/Sao_Paulo"


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).replace(tzinfo=ZoneInfo(TZ)).timestamp()


class Clock:
    def __init__(self, iso: str) -> None:
        self.now = _ts(iso)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeNotifier:
    name = "fake"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True

    @property
    def texts(self) -> list[str]:
        return [a.text for a in self.alerts]


def _monitor(clock: Clock, notifier: FakeNotifier, **kwargs) -> HealthMonitor:
    options = dict(timezone=TZ, daily_at=time(8, 0), offline_after_seconds=600, clock=clock)
    options.update(kwargs)
    return HealthMonitor(notifier, **options)


def test_start_announces_that_ia_watcher_is_running():
    clock, channel = Clock("2026-09-22T16:00:00"), FakeNotifier()
    _monitor(clock, channel).start()
    assert len(channel.alerts) == 1
    alert = channel.alerts[0]
    assert "ia_watcher is running" in alert.text
    assert "2026-09-22 16:00" in alert.text
    assert alert.risk is None and alert.images == ()


def test_daily_heartbeat_once_a_day_at_the_set_time():
    clock, channel = Clock("2026-09-22T16:00:00"), FakeNotifier()
    health = _monitor(clock, channel)
    health.start()

    clock.now = _ts("2026-09-23T07:59:00")
    health.check(last_frame_at=clock.now, summary="3 events")
    assert len(channel.alerts) == 1  # not 08:00 yet

    clock.now = _ts("2026-09-23T08:00:30")
    health.check(last_frame_at=clock.now, summary="3 events")
    assert len(channel.alerts) == 2
    assert "ia_watcher is running" in channel.texts[1] and "3 events" in channel.texts[1]

    clock.advance(3600)
    health.check(last_frame_at=clock.now)
    assert len(channel.alerts) == 2  # not repeated on the same day

    clock.now = _ts("2026-09-24T08:01:00")
    health.check(last_frame_at=clock.now)
    assert len(channel.alerts) == 3


def test_service_stopped_for_several_days_sends_a_single_heartbeat():
    clock, channel = Clock("2026-09-22T16:00:00"), FakeNotifier()
    health = _monitor(clock, channel)
    health.start()
    clock.now = _ts("2026-09-26T09:00:00")
    health.check(last_frame_at=clock.now)
    health.check(last_frame_at=clock.now)
    assert len(channel.alerts) == 2


def test_daily_heartbeat_disabled():
    clock, channel = Clock("2026-09-22T16:00:00"), FakeNotifier()
    health = _monitor(clock, channel, daily_at=None)
    health.start()
    clock.now = _ts("2026-09-23T09:00:00")
    health.check(last_frame_at=clock.now)
    assert len(channel.alerts) == 1  # only the startup one


def test_camera_offline_warns_once_and_reports_the_return():
    clock, channel = Clock("2026-09-22T16:00:00"), FakeNotifier()
    health = _monitor(clock, channel)
    health.start()
    last = clock.now

    clock.advance(599)
    health.check(last_frame_at=last)
    assert len(channel.alerts) == 1  # 9m59s: still tolerated

    clock.advance(2)
    health.check(last_frame_at=last)
    assert len(channel.alerts) == 2
    assert "camera offline" in channel.texts[1]
    assert "10 min" in channel.texts[1]

    clock.advance(300)
    health.check(last_frame_at=last)
    assert len(channel.alerts) == 2  # not repeated on every check

    health.check(last_frame_at=clock.now)  # frames are back
    assert len(channel.alerts) == 3
    assert "camera back" in channel.texts[2]
    assert "15 min" in channel.texts[2]


def test_camera_that_never_connected_counts_from_startup():
    clock, channel = Clock("2026-09-22T16:00:00"), FakeNotifier()
    health = _monitor(clock, channel)
    health.start()
    clock.advance(601)
    health.check(last_frame_at=None)
    assert "camera offline" in channel.texts[-1]


def test_check_before_start_starts_on_its_own():
    clock, channel = Clock("2026-09-22T16:00:00"), FakeNotifier()
    health = _monitor(clock, channel)
    health.check(last_frame_at=clock.now)
    assert len(channel.alerts) == 1 and "ia_watcher is running" in channel.texts[0]
