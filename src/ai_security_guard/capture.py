"""Frame capture from a video source (the Tapo camera's RTSP stream, or a file for tests).

Reading runs in its own thread and keeps **only the most recent frame**: the consumer
(motion detection) can be slower than the camera without piling up buffer or latency.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Self

import cv2
import numpy as np

from .config import Settings, mask_url

log = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Frame:
    """A BGR frame (OpenCV's format) with the capture time."""

    image: np.ndarray
    ts: float  # epoch seconds
    seq: int  # increasing counter since start() — identifies new frames


class Capture(Protocol):
    """The subset of `cv2.VideoCapture` used here (allows a fake source in tests)."""

    def isOpened(self) -> bool: ...  # noqa: N802 - OpenCV's signature

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


Opener = Callable[[str], Capture]


def open_capture(url: str) -> Capture:
    """Opens the source through FFmpeg: RTSP over TCP, minimal buffer, FFmpeg logs silenced.

    The Tapo drops packets over UDP ("torn" images); silencing FFmpeg keeps it from
    printing the URL with the password to stderr.
    """
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # AV_LOG_QUIET
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class FrameSource:
    """Reads a video source in the background, reconnecting with exponential backoff."""

    def __init__(
        self,
        url: str,
        *,
        reconnect_seconds: float = 5.0,
        max_backoff_seconds: float = MAX_BACKOFF_SECONDS,
        opener: Opener = open_capture,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.url = url
        self.name = mask_url(url)  # what may appear in logs
        self._reconnect_seconds = reconnect_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._opener = opener
        self._clock = clock
        self._stop = threading.Event()
        self._sleep = sleep if sleep is not None else self._stop.wait
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._seq = 0
        self._thread: threading.Thread | None = None

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> Self:
        return cls(
            settings.rtsp_url,
            reconnect_seconds=settings.camera.reconnect_seconds,
            **kwargs,
        )

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> Self:
        if self._thread is not None:
            raise RuntimeError("FrameSource already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-source", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stops the thread. Idempotent, and interrupts a backoff in progress."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                log.warning("capture thread did not stop within %.0fs (%s)", timeout, self.name)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # ---------------------------------------------------------------- reading

    def wait_for_frame(self, timeout: float = 5.0, after: int = 0) -> Frame | None:
        """Waits for a frame with `seq > after`. None on timeout or if the source stopped."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                frame = self._frame
                if frame is not None and frame.seq > after:
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._cond.wait(remaining)

    # ---------------------------------------------------------------- thread

    def _run(self) -> None:
        backoff = self._reconnect_seconds
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                log.warning("source unavailable (%s); retrying in %.0fs", self.name, backoff)
            else:
                log.info("source connected: %s", self.name)
                backoff = self._reconnect_seconds
                try:
                    self._pump(cap)
                finally:
                    _release(cap)
                if self._stop.is_set():
                    break
                log.warning("source dropped (%s); reconnecting in %.0fs", self.name, backoff)
            self._sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff_seconds)

    def _open(self) -> Capture | None:
        try:
            cap = self._opener(self.url)
        except Exception:
            log.warning("error opening the source (%s)", self.name, exc_info=True)
            return None
        if not cap.isOpened():
            _release(cap)
            return None
        return cap

    def _pump(self, cap: Capture) -> None:
        """Reads nonstop until the source drops — never leaves stale frames in FFmpeg's queue."""
        while not self._stop.is_set():
            try:
                ok, image = cap.read()
            except Exception:
                log.warning("error reading the source (%s)", self.name, exc_info=True)
                return
            if not ok or image is None:
                return
            self._publish(image)

    def _publish(self, image: np.ndarray) -> None:
        with self._cond:
            self._seq += 1
            self._frame = Frame(image=image, ts=self._clock(), seq=self._seq)
            self._cond.notify_all()


def _release(cap: Capture) -> None:
    try:
        cap.release()
    except Exception:  # pragma: no cover - depends on the backend
        log.debug("failed to release the capture", exc_info=True)
