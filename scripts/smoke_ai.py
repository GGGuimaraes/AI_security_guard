"""Real AI smoke test: classifies JPEGs from data/smoke/ through OpenRouter.

Makes the same call twice (same system prefix) to show how many tokens came from the
cache on the second one — that's how to check whether prompt caching is kicking in.

Needs config/config.yaml, a .env with OPENROUTER_API_KEY and images in data/smoke/
(run first: uv run python scripts/smoke_camera.py or smoke_motion.py).
Usage: uv run python scripts/smoke_ai.py --frames 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ia_watcher.ai_client import AIClient, AIResponse, AIUnavailable
from ia_watcher.config import AI_SECRETS, MissingSecrets, load_settings
from ia_watcher.event import Event, EventImage


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args.config)
        settings.secrets.require(*AI_SECRETS)
    except (MissingSecrets, FileNotFoundError) as err:
        print(err, file=sys.stderr)
        return 2

    paths = _pick_images(Path(args.images), args.frames)
    if not paths:
        print(
            f"no .jpg in {args.images}/ — run scripts/smoke_camera.py first",
            file=sys.stderr,
        )
        return 1

    event = _event_from(paths)
    print(f"model: {settings.ai.model} (cache_control={settings.ai.enable_cache_control})")
    print(f"criteria: {settings.ai.criteria_file} ({len(settings.criteria_text())} characters)")
    for path in paths:
        print(f"  image: {path} ({path.stat().st_size / 1024:.0f} KB)")
    print()

    with AIClient.from_settings(settings) as ai:
        for number in (1, 2):
            try:
                _show(number, ai.analyze(event))
            except AIUnavailable as err:
                print(f"call {number}: AI unavailable — {err}", file=sys.stderr)
                return 1
    print("\n2nd call with cached tokens > 0 = prompt caching is working")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--images", default="data/smoke", help="folder with the JPEGs")
    parser.add_argument("--frames", type=int, default=4, help="how many images to send")
    return parser.parse_args()


def _pick_images(root: Path, limit: int) -> list[Path]:
    """The most recent JPEGs in the folder (in chronological capture order)."""
    found = sorted(root.rglob("*.jpg"), key=lambda p: p.stat().st_mtime)
    return found[-limit:]


def _event_from(paths: list[Path]) -> Event:
    images = tuple(EventImage(data=p.read_bytes(), ts=p.stat().st_mtime) for p in paths)
    return Event(started_at=images[0].ts, images=images)


def _show(number: int, result: AIResponse) -> None:
    u = result.usage
    print(f"--- call {number} ({result.latency_seconds:.1f}s) ---")
    print(f"  tokens: {u.prompt_tokens} in / {u.completion_tokens} out")
    print(f"  cached: {u.cached_tokens}  cost: US$ {u.cost:.5f}")
    try:
        data = json.loads(result.content)
    except json.JSONDecodeError:
        print(
            f"  response is not plain JSON (the classifier tolerates it):\n  {result.content[:300]}"
        )
        return
    print(f"  risk: {data.get('risk')}  confidence: {data.get('confidence')}")
    print(f"  reason: {data.get('reason')}")
    print(f"  description: {data.get('description')}")


if __name__ == "__main__":
    raise SystemExit(main())
