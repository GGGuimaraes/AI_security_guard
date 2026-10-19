"""Real camera smoke test: saves one frame from the Tapo to data/smoke/.

Needs config/config.yaml and a filled-in .env (TAPO_USERNAME, TAPO_PASSWORD).
Usage: uv run python scripts/smoke_camera.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2

from ia_watcher.capture import FrameSource
from ia_watcher.config import CAMERA_SECRETS, MissingSecrets, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--timeout", type=float, default=15.0, help="seconds to the first frame")
    parser.add_argument("--out", default="data/smoke")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
        settings.secrets.require(*CAMERA_SECRETS)
    except (MissingSecrets, FileNotFoundError) as err:
        print(err, file=sys.stderr)
        return 2
    print(f"connecting to {settings.rtsp_url_safe}")

    with FrameSource.from_settings(settings) as source:
        frame = source.wait_for_frame(timeout=args.timeout)

    if frame is None:
        print(
            "no frame within "
            f"{args.timeout:.0f}s — check the camera's IP, the camera account in the Tapo app "
            "(Advanced Settings > Camera Account) and whether port 554 answers",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame-{datetime.fromtimestamp(frame.ts):%Y%m%d-%H%M%S}.jpg"
    if not cv2.imwrite(str(path), frame.image):
        print(f"failed to save {path}", file=sys.stderr)
        return 1

    height, width = frame.image.shape[:2]
    print(f"ok: {width}x{height} frame saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
