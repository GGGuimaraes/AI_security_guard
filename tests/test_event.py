"""Event assembly — no camera and no network: fake source, fake clock and fake sleep."""

import cv2
import numpy as np
import pytest

from ia_watcher.capture import Frame
from ia_watcher.config import load_settings
from ia_watcher.event import Event, EventBuilder, encode_jpeg

# --------------------------------------------------------------------------- helpers


class FakeClock:
    """Manual clock; `sleep` advances time instead of actually waiting."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _img(width: int = 640, height: int = 360, value: int = 120) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _noise(width: int = 640, height: int = 360) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


class FakeSource:
    """Delivers a new frame on every call, stamped with the fake clock.

    With `limit`, stops delivering after N frames (simulates the camera dropping midway).
    """

    def __init__(
        self, clock: FakeClock, *, image: np.ndarray | None = None, limit: int | None = None
    ) -> None:
        self.clock = clock
        self.image = _img() if image is None else image
        self.limit = limit
        self.seq = 0
        self.calls: list[int] = []

    def wait_for_frame(self, timeout: float = 5.0, after: int = 0) -> Frame | None:
        self.calls.append(after)
        if self.limit is not None and self.seq >= self.limit:
            return None
        self.seq += 1
        return Frame(image=self.image, ts=self.clock(), seq=self.seq)


def _builder(source: FakeSource, clock: FakeClock, **kwargs) -> EventBuilder:
    options = dict(
        frames_per_event=4,
        frame_interval_seconds=1.0,
        cooldown_seconds=60.0,
        max_image_width=1024,
        jpeg_quality=80,
    )
    options.update(kwargs)
    return EventBuilder(source, clock=clock, sleep=clock.sleep, **options)


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None, "invalid JPEG"
    return image


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# --------------------------------------------------------------------------- collection


def test_collects_the_configured_number_of_frames(clock):
    event = _builder(FakeSource(clock), clock, frames_per_event=3).collect()

    assert isinstance(event, Event)
    assert len(event.images) == 3


def test_frames_are_decodable_jpeg(clock):
    event = _builder(FakeSource(clock), clock, frames_per_event=2).collect()

    for image in event.images:
        assert image.data[:2] == b"\xff\xd8"  # JPEG marker
        assert _decode(image.data).shape[:2] == (360, 640)


def test_frames_are_spaced_by_the_interval(clock):
    event = _builder(
        FakeSource(clock), clock, frames_per_event=4, frame_interval_seconds=0.5
    ).collect()

    assert clock.slept == [0.5, 0.5, 0.5]  # waits between frames, not before the first
    assert [i.ts for i in event.images] == [1000.0, 1000.5, 1001.0, 1001.5]


def test_always_asks_for_a_new_frame(clock):
    source = FakeSource(clock)
    event = _builder(source, clock, frames_per_event=3).collect()

    assert source.calls == [0, 1, 2]  # each wait requires seq > the previous frame's
    assert len(event.images) == 3


def test_keeps_the_motion_area(clock):
    event = _builder(FakeSource(clock), clock, frames_per_event=1).collect(0.12)

    assert event.area_ratio == pytest.approx(0.12)
    assert event.started_at == 1000.0


# --------------------------------------------------------------------------- cooldown


def test_cooldown_blocks_the_next_event(clock):
    builder = _builder(FakeSource(clock), clock, frames_per_event=2, cooldown_seconds=60)

    assert builder.collect() is not None
    assert builder.ready() is False
    assert builder.collect() is None


def test_cooldown_counts_from_the_event_start(clock):
    builder = _builder(FakeSource(clock), clock, frames_per_event=2, cooldown_seconds=10)

    builder.collect()  # starts at 1000, spends 1s collecting
    clock.now = 1009.5
    assert builder.ready() is False

    clock.now = 1010.0
    assert builder.ready() is True
    assert builder.collect() is not None


def test_zero_cooldown_allows_back_to_back_events(clock):
    builder = _builder(FakeSource(clock), clock, frames_per_event=1, cooldown_seconds=0)

    assert builder.collect() is not None
    assert builder.collect() is not None


def test_cooldown_does_not_advance_without_frames(clock):
    """Silent camera: don't spend the cooldown on an empty event."""
    builder = _builder(FakeSource(clock, limit=0), clock, cooldown_seconds=60)

    assert builder.collect() is None
    assert builder.ready() is True


# --------------------------------------------------------------------------- image


def test_downscales_to_the_max_width(clock):
    source = FakeSource(clock, image=_img(1920, 1080))

    event = _builder(source, clock, frames_per_event=1, max_image_width=640).collect()

    assert _decode(event.images[0].data).shape[:2] == (360, 640)  # keeps the aspect ratio


def test_does_not_upscale_a_smaller_image(clock):
    source = FakeSource(clock, image=_img(320, 180))

    event = _builder(source, clock, frames_per_event=1, max_image_width=1024).collect()

    assert _decode(event.images[0].data).shape[:2] == (180, 320)


def test_jpeg_quality_changes_the_size():
    image = _noise()

    assert len(encode_jpeg(image, max_width=640, quality=20)) < len(
        encode_jpeg(image, max_width=640, quality=95)
    )


# --------------------------------------------------------------------------- failures


def test_source_dropping_midway_returns_what_was_collected(clock):
    source = FakeSource(clock, limit=2)

    event = _builder(source, clock, frames_per_event=4).collect()

    assert len(event.images) == 2  # a partial event beats no event


def test_source_without_frames_produces_no_event(clock):
    assert _builder(FakeSource(clock, limit=0), clock).collect() is None


# --------------------------------------------------------------------------- config


def test_from_settings_uses_the_configuration(example_config, fake_env, clock):
    settings = load_settings(example_config, env=fake_env)
    source = FakeSource(clock)

    builder = EventBuilder.from_settings(settings, source, clock=clock, sleep=clock.sleep)

    assert builder.frames_per_event == settings.event.frames_per_event
    assert builder.frame_interval_seconds == settings.event.frame_interval_seconds
    assert builder.cooldown_seconds == settings.event.cooldown_seconds
    assert builder.max_image_width == settings.event.max_image_width
    assert builder.jpeg_quality == settings.event.jpeg_quality
    assert len(builder.collect().images) == settings.event.frames_per_event
