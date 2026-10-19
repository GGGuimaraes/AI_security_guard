"""Local motion detection (OpenCV), the stage that decides whether the AI is worth calling.

Works on a downscaled copy of the frame (`detect_width`): the cost drops a lot, and all
that matters here is "did a large area change?", not the details.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np

from .config import Settings

Zone = tuple[float, float, float, float]  # x, y, w, h normalized 0-1

_SHADOW = 127  # value MOG2 uses for shadows; only 255 counts as motion
MIN_WARMUP = 5  # before this, MOG2's model is immature and marks new objects as shadow
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass(frozen=True, slots=True)
class MotionResult:
    """Result for one frame. `bbox` is in pixels of the original frame (x, y, w, h)."""

    moving: bool
    area_ratio: float
    bbox: tuple[int, int, int, int] | None = None
    warming_up: bool = False


class MotionDetector:
    """Background subtraction (MOG2) with ignored zones and an initial warm-up.

    During warm-up (`warmup_frames`, at least `MIN_WARMUP`) the background is learned
    and nothing is reported as motion. Not thread-safe: one instance per source.
    """

    def __init__(
        self,
        *,
        detect_width: int = 320,
        min_area_ratio: float = 0.01,
        warmup_frames: int = 30,
        ignore_zones: Iterable[Sequence[float]] = (),
    ) -> None:
        self.detect_width = detect_width
        self.min_area_ratio = min_area_ratio
        self.warmup_frames = max(warmup_frames, MIN_WARMUP)
        self.ignore_zones = [_validate_zone(z) for z in ignore_zones]
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self._seen = 0
        self._ignore_mask: np.ndarray | None = None  # cached per downscaled size

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        m = settings.motion
        return cls(
            detect_width=m.detect_width,
            min_area_ratio=m.min_area_ratio,
            warmup_frames=m.warmup_frames,
            ignore_zones=m.ignore_zones,
        )

    def update(self, image: np.ndarray) -> MotionResult:
        """Processes one frame (BGR or grayscale) and tells whether there is relevant motion."""
        small = self._shrink(image)
        mask = self._foreground(small)
        area_ratio = float(np.count_nonzero(mask)) / mask.size
        self._seen += 1

        if self._seen <= self.warmup_frames:
            return MotionResult(moving=False, area_ratio=area_ratio, warming_up=True)
        if area_ratio < self.min_area_ratio:
            return MotionResult(moving=False, area_ratio=area_ratio)

        scale = image.shape[1] / small.shape[1]
        return MotionResult(moving=True, area_ratio=area_ratio, bbox=_bbox(mask, scale))

    # ---------------------------------------------------------------- internals

    def _shrink(self, image: np.ndarray) -> np.ndarray:
        """Downscales and converts to grayscale. Never upscales: that would only waste CPU."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        if w > self.detect_width:
            height = max(1, round(h * self.detect_width / w))
            image = cv2.resize(image, (self.detect_width, height), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(image, (5, 5), 0)  # removes sensor noise

    def _foreground(self, small: np.ndarray) -> np.ndarray:
        mask = self._subtractor.apply(small)
        mask[mask <= _SHADOW] = 0  # shadows are not motion
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)  # kills specks
        mask = cv2.dilate(mask, _KERNEL)  # joins pieces of the same object
        ignore = self._mask_for(small.shape[:2])
        return mask if ignore is None else cv2.bitwise_and(mask, ignore)

    def _mask_for(self, shape: tuple[int, int]) -> np.ndarray | None:
        """Mask 255 = analyze, 0 = ignore. Recomputed only when the size changes."""
        if not self.ignore_zones:
            return None
        if self._ignore_mask is None or self._ignore_mask.shape != shape:
            h, w = shape
            mask = np.full(shape, 255, dtype=np.uint8)
            for x, y, zw, zh in self.ignore_zones:
                x0, y0 = int(x * w), int(y * h)
                mask[y0 : y0 + round(zh * h), x0 : x0 + round(zw * w)] = 0
            self._ignore_mask = mask
        return self._ignore_mask


def _bbox(mask: np.ndarray, scale: float) -> tuple[int, int, int, int]:
    """Rectangle covering all the motion, converted back to the original frame."""
    x, y, w, h = cv2.boundingRect(mask)
    return (round(x * scale), round(y * scale), round(w * scale), round(h * scale))


def _validate_zone(zone: Sequence[float]) -> Zone:
    if len(zone) != 4 or not all(0 <= v <= 1 for v in zone):
        raise ValueError(f"invalid zone {list(zone)}: use [x, y, w, h] normalized between 0 and 1")
    x, y, w, h = zone
    return (float(x), float(y), float(w), float(h))
