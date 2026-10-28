"""Real detection smoke test: watches the Tapo and reports motion for N seconds.

Used to tune `motion.*` on the real scene. At the end it prints a summary with the
distribution of the changed area — that's what tells where `min_area_ratio` should sit.

Needs config/config.yaml and a filled-in .env (TAPO_USERNAME, TAPO_PASSWORD).
Usage: uv run python scripts/smoke_motion.py --seconds 120
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ai_security_guard.capture import FrameSource
from ai_security_guard.config import CAMERA_SECRETS, MissingSecrets, load_settings
from ai_security_guard.motion import MotionDetector, MotionResult

STATUS_EVERY = 10.0  # seconds between status lines when nothing happens
BURST_GAP = 2.0  # silence that separates two bursts of motion


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
        settings.secrets.require(*CAMERA_SECRETS)
    except (MissingSecrets, FileNotFoundError) as err:
        print(err, file=sys.stderr)
        return 2

    detector = MotionDetector.from_settings(settings)
    out_dir = Path(args.out)
    print(f"connecting to {settings.rtsp_url_safe}")
    print(f"{detector}")
    print(f"warming up {detector.warmup_frames} frames, then {args.seconds:.0f}s of monitoring")
    print("walk in front of the camera during the test\n")

    stats = _Stats()
    deadline = time.monotonic() + args.seconds
    next_status = time.monotonic() + STATUS_EVERY
    last_saved = 0.0
    seq = 0

    with FrameSource.from_settings(settings) as source:
        if source.wait_for_frame(timeout=args.timeout) is None:
            print(f"no frame within {args.timeout:.0f}s — run smoke_camera.py", file=sys.stderr)
            return 1

        started = time.monotonic()
        while time.monotonic() < deadline:
            frame = source.wait_for_frame(timeout=args.timeout, after=seq)
            if frame is None:
                print("the source stopped delivering frames", file=sys.stderr)
                return 1
            seq = frame.seq
            result = detector.update(frame.image)
            stats.add(result, frame.ts)

            if result.moving:
                print(_line(frame.ts, result))
                if not args.no_save and time.monotonic() - last_saved >= args.save_every:
                    _save(out_dir, frame.image, frame.ts, result)
                    last_saved = time.monotonic()
                next_status = time.monotonic() + STATUS_EVERY
            elif not result.warming_up and time.monotonic() >= next_status:
                print(f"  {_clock(frame.ts)} no motion (current area {result.area_ratio:.2%})")
                next_status = time.monotonic() + STATUS_EVERY

        stats.report(time.monotonic() - started, detector)
    if not args.no_save:
        print(f"frames with bbox in {out_dir}/")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seconds", type=float, default=120.0, help="monitoring duration")
    parser.add_argument("--timeout", type=float, default=15.0, help="seconds to the first frame")
    parser.add_argument("--out", default="data/smoke/motion")
    parser.add_argument(
        "--save-every", type=float, default=1.0, help="seconds between saved frames"
    )
    parser.add_argument("--no-save", action="store_true", help="don't save detection frames")
    return parser.parse_args()


class _Stats:
    """Gathers what matters for choosing the `motion.*` thresholds."""

    def __init__(self) -> None:
        self.frames = 0
        self.quiet: list[float] = []
        self.moving: list[float] = []
        self.bursts = 0
        self._last_move_ts = 0.0

    def add(self, result: MotionResult, ts: float) -> None:
        self.frames += 1
        if result.warming_up:
            return
        if result.moving:
            self.moving.append(result.area_ratio)
            if ts - self._last_move_ts > BURST_GAP:
                self.bursts += 1
            self._last_move_ts = ts
        else:
            self.quiet.append(result.area_ratio)

    def report(self, elapsed: float, detector: MotionDetector) -> None:
        fps = self.frames / elapsed if elapsed else 0.0
        print(f"\n{'-' * 60}\ncalibration summary")
        print(f"  {self.frames} frames in {elapsed:.0f}s ({fps:.1f} fps)")
        print(f"  motion bursts: {self.bursts} ({len(self.moving)} frames)")
        print(f"  current threshold (min_area_ratio): {detector.min_area_ratio:.2%}")

        if self.quiet:
            q = np.percentile(self.quiet, [50, 90, 99])
            print(
                f"  area with a still scene: p50={q[0]:.2%} p90={q[1]:.2%} "
                f"p99={q[2]:.2%} max={max(self.quiet):.2%}"
            )
        if self.moving:
            print(
                f"  area with motion: min={min(self.moving):.2%} "
                f"median={np.median(self.moving):.2%} max={max(self.moving):.2%}"
            )
            floor = np.percentile(self.quiet, 99) if self.quiet else 0.0
            print(f"\nsuggestion: min_area_ratio between {floor:.2%} and {min(self.moving):.2%}")
        else:
            print("\nno motion recorded — nobody walked by, or the threshold is too high")


def _clock(ts: float) -> str:
    return f"{datetime.fromtimestamp(ts):%H:%M:%S}"


def _line(ts: float, result: MotionResult) -> str:
    return f"{_clock(ts)} MOTION area={result.area_ratio:.2%} bbox={result.bbox}"


def _save(out_dir: Path, image: np.ndarray, ts: float, result: MotionResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marked = image.copy()
    if result.bbox:
        x, y, w, h = result.bbox
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 3)
    path = out_dir / f"motion-{datetime.fromtimestamp(ts):%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(str(path), marked, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    raise SystemExit(main())
