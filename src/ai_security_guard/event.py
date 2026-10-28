"""Event assembly: the batch of frames that will be sent to the AI.

When `motion.py` reports motion, this stage collects `frames_per_event` frames spaced
out in time (so the AI sees the scene unfold: arriving, handling something, leaving),
downscales each one to `max_image_width` and encodes it as JPEG. The cooldown keeps a
single person walking across the room from turning into ten AI calls in a row.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Self

import cv2
import numpy as np

from .capture import Frame
from .config import Settings

log = logging.getLogger(__name__)


class FrameProvider(Protocol):
    """What an event needs from a frame source (`FrameSource` already provides it)."""

    def wait_for_frame(self, timeout: float = 5.0, after: int = 0) -> Frame | None: ...


@dataclass(frozen=True, slots=True)
class EventImage:
    """A frame ready to send: JPEG, with the time it was captured."""

    data: bytes
    ts: float


@dataclass(frozen=True, slots=True)
class Event:
    """Batch of images from one motion, plus the fraction of the frame that changed."""

    started_at: float  # epoch seconds of the first frame
    images: tuple[EventImage, ...]
    area_ratio: float = 0.0


def encode_jpeg(image: np.ndarray, *, max_width: int, quality: int) -> bytes:
    """Downscales (never upscales) to `max_width` and encodes as JPEG."""
    h, w = image.shape[:2]
    if w > max_width:
        height = max(1, round(h * max_width / w))
        image = cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover - only fails with an invalid image
        raise ValueError("could not encode the frame as JPEG")
    return buffer.tobytes()


class EventBuilder:
    """Collects the batch of frames for an event, honoring the cooldown.

    `collect()` blocks for about `frames_per_event * frame_interval_seconds`.
    Not thread-safe: one instance per source.
    """

    def __init__(
        self,
        source: FrameProvider,
        *,
        frames_per_event: int = 4,
        frame_interval_seconds: float = 1.0,
        cooldown_seconds: float = 60.0,
        max_image_width: int = 1024,
        jpeg_quality: int = 80,
        frame_timeout_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.source = source
        self.frames_per_event = frames_per_event
        self.frame_interval_seconds = frame_interval_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_image_width = max_image_width
        self.jpeg_quality = jpeg_quality
        self.frame_timeout_seconds = frame_timeout_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_event_at: float | None = None

    @classmethod
    def from_settings(cls, settings: Settings, source: FrameProvider, **kwargs) -> Self:
        e = settings.event
        return cls(
            source,
            frames_per_event=e.frames_per_event,
            frame_interval_seconds=e.frame_interval_seconds,
            cooldown_seconds=e.cooldown_seconds,
            max_image_width=e.max_image_width,
            jpeg_quality=e.jpeg_quality,
            **kwargs,
        )

    def ready(self) -> bool:
        """True once the cooldown has passed since the start of the last collected event."""
        if self._last_event_at is None:
            return True
        return self._clock() - self._last_event_at >= self.cooldown_seconds

    def collect(self, area_ratio: float = 0.0) -> Event | None:
        """Builds the event. None while in cooldown or if no frame arrived."""
        if not self.ready():
            return None
        started = self._clock()
        images = self._grab_images()
        if not images:
            log.warning("motion detected but no frame available; event dropped")
            return None  # no cooldown: the source failed, nothing was analyzed

        self._last_event_at = started
        if len(images) < self.frames_per_event:
            log.warning("partial event: %d of %d frames", len(images), self.frames_per_event)
        return Event(started_at=images[0].ts, images=tuple(images), area_ratio=area_ratio)

    def _grab_images(self) -> list[EventImage]:
        images: list[EventImage] = []
        after = 0
        for index in range(self.frames_per_event):
            if index:
                self._sleep(self.frame_interval_seconds)
            frame = self.source.wait_for_frame(timeout=self._timeout(), after=after)
            if frame is None:
                break  # source dropped: send what we have instead of losing the event
            after = frame.seq
            jpeg = encode_jpeg(
                frame.image, max_width=self.max_image_width, quality=self.jpeg_quality
            )
            images.append(EventImage(data=jpeg, ts=frame.ts))
        return images

    def _timeout(self) -> float:
        """Never wait less than the interval between frames."""
        return max(self.frame_timeout_seconds, self.frame_interval_seconds)
