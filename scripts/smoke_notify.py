"""Notification smoke test: builds an alert from data/smoke/ JPEGs and sends it via the channel.

With `notify.channel: simulated` (the default) nothing leaves the machine: the alert is
only logged. Switching the channel in config/config.yaml,
this same script exercises the real channel — that's the point of a decoupled notifier.

Usage: uv run python scripts/smoke_notify.py --risk high
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ai_security_guard.classifier import Analysis, unavailable
from ai_security_guard.config import RiskLevel, load_settings
from ai_security_guard.event import Event, EventImage
from ai_security_guard.notifier import build_alert, build_notifier


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
        notifier = build_notifier(settings)
    except (ValueError, FileNotFoundError) as err:
        print(err, file=sys.stderr)
        return 2

    event = _fake_event(Path(args.images), settings.notify.max_images)
    analysis = (
        unavailable("smoke test: simulating an AI outage")
        if args.unavailable
        else Analysis(
            risk=RiskLevel.parse(args.risk),
            reason="Notification channel smoke test.",
            description="Test alert built by scripts/smoke_notify.py.",
            confidence=0.42,
            people=1,
        )
    )
    alert = build_alert(
        event,
        analysis,
        timezone=settings.notify.timezone,
        max_images=settings.notify.max_images,
    )
    print(f"channel: {notifier.name} · images: {len(alert.images)}\n---\n{alert.text}\n---")
    ok = notifier.send(alert)
    print("sent" if ok else "FAILED (see the log above)")
    return 0 if ok else 1


def _fake_event(images_dir: Path, limit: int) -> Event:
    paths = sorted(p for p in images_dir.glob("**/*.jpg"))[:limit]
    now = time.time()
    if not paths:
        print(
            f"warning: no .jpg in {images_dir}/ — the alert goes out as text only", file=sys.stderr
        )
    return Event(
        started_at=now,
        images=tuple(EventImage(data=p.read_bytes(), ts=now + i) for i, p in enumerate(paths)),
        area_ratio=0.05,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--images", default="data/smoke")
    parser.add_argument("--risk", default="high", choices=[r.value for r in RiskLevel])
    parser.add_argument(
        "--unavailable", action="store_true", help="simulates an AI failure (medium fallback)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
