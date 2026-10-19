"""Frame capture — no camera and no network: a fake source and a synthetic .mp4."""

import logging
import time

import cv2
import numpy as np
import pytest

from ia_watcher.capture import Frame, FrameSource
from ia_watcher.config import load_settings

# --------------------------------------------------------------------------- helpers


def _img(value: int, size: int = 4) -> np.ndarray:
    return np.full((size, size, 3), value % 256, dtype=np.uint8)


class FakeCapture:
    """Fake source: delivers `frames` reads and then drops (`read` -> False).

    With `endless=True` it never drops — a final state so the thread doesn't spin in a
    tight reconnect loop during the test.
    """

    def __init__(
        self, frames: int = 1, *, opened: bool = True, endless: bool = False, error: bool = False
    ) -> None:
        self.opened = opened
        self.endless = endless
        self.error = error
        self.remaining = frames
        self.reads = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoCapture's signature
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.reads += 1
        if self.error:
            raise RuntimeError("corrupted stream")
        if self.remaining > 0:
            self.remaining -= 1
            return True, _img(self.reads)
        if self.endless:
            time.sleep(0.005)
            return True, _img(self.reads)
        return False, None

    def release(self) -> None:
        self.released = True


class FakeOpener:
    """Returns the captures in order; once exhausted, always one made by `default`."""

    def __init__(
        self, *captures: FakeCapture, default=lambda: FakeCapture(0, endless=True)
    ) -> None:
        self.queue = list(captures)
        self.default = default
        self.urls: list[str] = []
        self.made: list[FakeCapture] = []

    def __call__(self, url: str) -> FakeCapture:
        self.urls.append(url)
        cap = self.queue.pop(0) if self.queue else self.default()
        self.made.append(cap)
        return cap


class RecordingSleep:
    """Replaces sleep: records the requested backoff and returns right away."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, delay: float) -> None:
        self.calls.append(delay)
        time.sleep(0.001)


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------------------------- tests


def test_no_frame_before_start():
    src = FrameSource("rtsp://cam/stream1", opener=FakeOpener(), sleep=RecordingSleep())
    assert src.wait_for_frame(timeout=0.05) is None


def test_reads_frames_and_keeps_only_the_newest():
    opener = FakeOpener(FakeCapture(5, endless=True))
    clock = iter([100.0, 101.0, 102.0, 103.0, 104.0])
    with FrameSource(
        "rtsp://cam/stream1",
        opener=opener,
        sleep=RecordingSleep(),
        clock=lambda: next(clock, 999.0),
    ) as src:
        frame = src.wait_for_frame(timeout=2.0, after=4)

    assert isinstance(frame, Frame)
    assert frame.seq >= 5  # no queued frames: only the newest
    assert frame.ts >= 104.0
    assert frame.image.shape == (4, 4, 3)


def test_wait_for_frame_returns_new_frame_only():
    opener = FakeOpener(FakeCapture(3, endless=True))
    with FrameSource("rtsp://cam/stream1", opener=opener, sleep=RecordingSleep()) as src:
        first = src.wait_for_frame(timeout=2.0)
        assert first is not None
        later = src.wait_for_frame(timeout=2.0, after=first.seq)
        assert later is not None and later.seq > first.seq


def test_wait_for_frame_times_out_when_source_is_down():
    opener = FakeOpener(default=lambda: FakeCapture(0, opened=False))
    with FrameSource("rtsp://cam/stream1", opener=opener, sleep=RecordingSleep()) as src:
        assert src.wait_for_frame(timeout=0.2) is None


def test_reconnects_after_source_drop():
    first = FakeCapture(2)
    opener = FakeOpener(first, FakeCapture(0, endless=True))
    sleep = RecordingSleep()
    with FrameSource(
        "rtsp://cam/stream1", reconnect_seconds=5.0, opener=opener, sleep=sleep
    ) as src:
        assert wait_until(lambda: len(opener.urls) >= 2)
        assert src.wait_for_frame(timeout=2.0, after=2) is not None  # voltou a entregar frames

    assert first.released is True
    assert sleep.calls[0] == 5.0


def test_read_error_is_treated_as_drop():
    broken = FakeCapture(0, error=True)
    opener = FakeOpener(broken, FakeCapture(0, endless=True))
    with FrameSource("rtsp://cam/stream1", opener=opener, sleep=RecordingSleep()) as src:
        assert src.wait_for_frame(timeout=2.0) is not None
    assert broken.released is True


def test_backoff_doubles_and_resets_after_success():
    opener = FakeOpener(
        FakeCapture(0, opened=False),
        FakeCapture(0, opened=False),
        FakeCapture(0, opened=False),
        FakeCapture(1),  # abre, entrega 1 frame e cai -> backoff volta ao inicial
        FakeCapture(0, endless=True),
    )
    sleep = RecordingSleep()
    with FrameSource(
        "rtsp://cam/stream1", reconnect_seconds=5.0, opener=opener, sleep=sleep
    ) as src:
        assert wait_until(lambda: len(sleep.calls) >= 4)
        assert src.wait_for_frame(timeout=2.0) is not None

    assert sleep.calls[:4] == [5.0, 10.0, 20.0, 5.0]


def test_backoff_is_capped():
    opener = FakeOpener(default=lambda: FakeCapture(0, opened=False))
    sleep = RecordingSleep()
    with FrameSource(
        "rtsp://cam/stream1",
        reconnect_seconds=5.0,
        max_backoff_seconds=10.0,
        opener=opener,
        sleep=sleep,
    ):
        assert wait_until(lambda: len(sleep.calls) >= 4)
    assert sleep.calls[:4] == [5.0, 10.0, 10.0, 10.0]


def test_stop_interrupts_a_long_backoff():
    opener = FakeOpener(default=lambda: FakeCapture(0, opened=False))
    src = FrameSource("rtsp://cam/stream1", reconnect_seconds=30.0, opener=opener).start()
    assert wait_until(lambda: len(opener.urls) >= 1)
    started = time.monotonic()
    src.stop()
    assert time.monotonic() - started < 2.0  # does not wait out the 30s backoff


def test_stop_is_idempotent_and_releases_capture():
    cap = FakeCapture(0, endless=True)
    src = FrameSource("rtsp://cam/stream1", opener=FakeOpener(cap), sleep=RecordingSleep()).start()
    assert src.wait_for_frame(timeout=2.0) is not None
    src.stop()
    src.stop()
    assert cap.released is True


def test_logs_never_contain_the_password(caplog):
    url = "rtsp://ia_watcher:s3cr3tPass@192.168.0.50:554/stream1"
    opener = FakeOpener(FakeCapture(1), FakeCapture(0, endless=True))
    with (
        caplog.at_level(logging.DEBUG, logger="ia_watcher.capture"),
        FrameSource(url, opener=opener, sleep=RecordingSleep()),
    ):
        assert wait_until(lambda: len(opener.urls) >= 2)
    text = caplog.text
    assert "s3cr3tPass" not in text
    assert "rtsp://ia_watcher:***@192.168.0.50:554/stream1" in text


def test_from_settings_uses_masked_name(example_config, fake_env):
    settings = load_settings(example_config, env=fake_env)
    src = FrameSource.from_settings(settings, opener=FakeOpener())
    assert src.name == settings.rtsp_url_safe
    assert fake_env["TAPO_PASSWORD"] not in repr(src)
    assert src.url == settings.rtsp_url


def test_reads_real_video_file(tmp_path):
    """Reads a real synthetic .mp4 (same code path as RTSP)."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    if not writer.isOpened():
        pytest.skip("OpenCV build without the mp4v encoder")
    for i in range(20):
        writer.write(np.full((120, 160, 3), 10 * i % 256, dtype=np.uint8))
    writer.release()

    with FrameSource(str(path), reconnect_seconds=30.0, clock=lambda: 1_700_000_000.0) as src:
        frame = src.wait_for_frame(timeout=5.0)

    assert frame is not None
    assert frame.image.shape == (120, 160, 3)
    assert frame.ts == 1_700_000_000.0
    assert frame.seq >= 1
