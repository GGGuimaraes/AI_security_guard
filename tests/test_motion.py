"""Motion detection — synthetic frames only, no camera and no network."""

import numpy as np
import pytest

from ai_security_guard.config import load_settings
from ai_security_guard.motion import MIN_WARMUP, MotionDetector, MotionResult

BG_SHAPE = (240, 320, 3)  # height, width, channels


# --------------------------------------------------------------------------- helpers


def background(shape: tuple[int, int, int] = BG_SHAPE, seed: int = 0) -> np.ndarray:
    """Static textured background (a real scene is never uniform)."""
    rng = np.random.default_rng(seed)
    return rng.integers(40, 80, size=shape, dtype=np.uint8)


def with_square(
    base: np.ndarray, x: float, y: float, size: float = 0.25, value: int = 230
) -> np.ndarray:
    """Pastes a bright square on the background. Coordinates normalized 0-1."""
    h, w = base.shape[:2]
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = x0 + int(size * w), y0 + int(size * h)
    out = base.copy()
    out[y0:y1, x0:x1] = value
    return out


def warm_up(detector: MotionDetector, base: np.ndarray, frames: int) -> None:
    """Feeds the detector the still background until warm-up ends."""
    for _ in range(frames):
        detector.update(base)


# --------------------------------------------------------------------------- tests


def test_static_background_has_no_motion():
    base = background()
    det = MotionDetector(warmup_frames=10, min_area_ratio=0.01)
    warm_up(det, base, 10)

    result = det.update(base)

    assert isinstance(result, MotionResult)
    assert result.moving is False
    assert result.warming_up is False
    assert result.bbox is None
    assert result.area_ratio < 0.01


def test_square_entering_triggers_motion():
    base = background()
    det = MotionDetector(warmup_frames=10, min_area_ratio=0.01)
    warm_up(det, base, 10)

    result = det.update(with_square(base, x=0.4, y=0.4))

    assert result.moving is True
    assert result.area_ratio > 0.01
    assert result.bbox is not None


def test_bbox_is_in_original_frame_pixels():
    """The detector works downscaled, but the bbox comes back at the input frame's scale."""
    base = background((480, 640, 3))
    det = MotionDetector(detect_width=160, warmup_frames=10)
    warm_up(det, base, 10)

    result = det.update(with_square(base, x=0.5, y=0.25, size=0.2))

    assert result.bbox is not None
    x, y, w, h = result.bbox
    assert 0 <= x <= 640 and 0 <= y <= 480
    assert x + w <= 640 and y + h <= 480
    assert abs(x - 320) < 40 and abs(y - 120) < 40  # square's corner
    assert abs(w - 128) < 40 and abs(h - 96) < 40  # 20% of 640x480


def test_motion_only_inside_ignore_zone_is_discarded():
    base = background()
    det = MotionDetector(
        warmup_frames=10,
        min_area_ratio=0.01,
        ignore_zones=[(0.0, 0.0, 0.5, 0.5)],  # top-left corner (e.g. date overlay)
    )
    warm_up(det, base, 10)

    result = det.update(with_square(base, x=0.1, y=0.1, size=0.2))

    assert result.moving is False
    assert result.bbox is None
    assert result.area_ratio == 0.0


def test_motion_outside_ignore_zone_still_detected():
    base = background()
    det = MotionDetector(warmup_frames=10, min_area_ratio=0.01, ignore_zones=[(0.0, 0.0, 0.5, 0.5)])
    warm_up(det, base, 10)

    result = det.update(with_square(base, x=0.6, y=0.6, size=0.25))

    assert result.moving is True
    assert result.bbox is not None
    x, _y, _w, _h = result.bbox
    assert x >= BG_SHAPE[1] * 0.4  # the bbox stayed outside the ignored zone


def test_no_motion_during_warmup():
    base = background()
    det = MotionDetector(warmup_frames=8, min_area_ratio=0.01)

    results = [det.update(with_square(base, x=0.3, y=0.3)) for _ in range(8)]

    assert all(r.moving is False for r in results)
    assert all(r.warming_up is True for r in results)


def test_small_change_below_min_area_ratio_is_ignored():
    base = background()
    det = MotionDetector(warmup_frames=10, min_area_ratio=0.2)
    warm_up(det, base, 10)

    result = det.update(with_square(base, x=0.4, y=0.4, size=0.05))

    assert result.moving is False
    assert 0.0 < result.area_ratio < 0.2


def test_object_that_leaves_stops_the_motion():
    base = background()
    det = MotionDetector(warmup_frames=10, min_area_ratio=0.01)
    warm_up(det, base, 10)

    assert det.update(with_square(base, x=0.4, y=0.4)).moving is True
    det.update(base)  # object leaves; the next frame is background only
    assert det.update(base).moving is False


def test_accepts_grayscale_frames():
    base = background()[:, :, 0]
    det = MotionDetector(warmup_frames=5, min_area_ratio=0.01)
    warm_up(det, base, 5)

    assert det.update(base).moving is False
    assert det.update(with_square(base, x=0.4, y=0.4)).moving is True


def test_frame_narrower_than_detect_width_is_not_upscaled():
    base = background((60, 80, 3))
    det = MotionDetector(detect_width=320, warmup_frames=5, min_area_ratio=0.01)
    warm_up(det, base, 5)

    result = det.update(with_square(base, x=0.4, y=0.4))

    assert result.moving is True
    assert result.bbox is not None
    x, y, w, h = result.bbox
    assert x + w <= 80 and y + h <= 60


def test_from_settings_reads_the_motion_section(example_config, fake_env):
    settings = load_settings(example_config, env=fake_env)
    det = MotionDetector.from_settings(settings)

    assert det.detect_width == settings.motion.detect_width
    assert det.min_area_ratio == settings.motion.min_area_ratio
    assert det.warmup_frames == settings.motion.warmup_frames


def test_invalid_ignore_zone_is_rejected():
    with pytest.raises(ValueError, match="zone"):
        MotionDetector(ignore_zones=[(0.0, 0.0, 1.5, 0.5)])


def test_warmup_is_never_shorter_than_the_mog2_floor():
    """Below MIN_WARMUP, MOG2's model still mistakes objects for shadows."""
    base = background()
    det = MotionDetector(warmup_frames=0, min_area_ratio=0.01)
    warm_up(det, base, MIN_WARMUP)

    assert det.update(with_square(base, x=0.4, y=0.4)).moving is True
