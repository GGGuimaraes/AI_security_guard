"""Service entry point: `uv run python -m ai_security_guard`.

Loads the configuration, requires the camera and AI secrets (the channel asks for its
own in its factory) and hands the process over to the `Pipeline`. SIGINT and SIGTERM —
Ctrl+C and `docker compose stop` — request a clean shutdown.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from types import FrameType

from .config import AI_SECRETS, CAMERA_SECRETS, MissingSecrets, load_settings
from .pipeline import Pipeline

log = logging.getLogger("ai_security_guard")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError) as err:
        print(f"invalid configuration: {err}", file=sys.stderr)
        return 2

    _setup_logging(settings.logging.level)
    try:
        settings.secrets.require(*CAMERA_SECRETS, *AI_SECRETS)
        pipeline = Pipeline.from_settings(settings)
    except (MissingSecrets, ValueError, OSError) as err:
        log.error("could not start: %s", err)
        return 2

    _install_signals(pipeline)
    try:
        pipeline.run()
    except KeyboardInterrupt:  # Ctrl+C between installing the handler and the loop
        log.info("interrupted by the user")
    return 0


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _install_signals(pipeline: Pipeline) -> None:
    def handler(signum: int, _frame: FrameType | None) -> None:
        log.info("signal %s received; shutting down", signal.Signals(signum).name)
        pipeline.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ai_security_guard", description=__doc__)
    parser.add_argument("--config", default="config/config.yaml", help="configuration file")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
