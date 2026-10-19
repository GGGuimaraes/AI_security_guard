"""Vision AI client (OpenRouter, OpenAI-compatible API).

Sends an event's batch of frames and returns the raw response text — validating the
JSON is `classifier.py`'s job. Messages are built in two parts:

1. `system`: the contents of the risk criteria file, always **first and byte-for-byte
   identical**, so the provider can reuse the cached prefix.
2. `user`: short variable context (local time) + the images in chronological order.

Any failure — network, HTTP, empty response — becomes `AIUnavailable`, never silence:
the pipeline handles it by alerting as `medium`.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self
from zoneinfo import ZoneInfo

import httpx

from .config import AI_SECRETS, Settings
from .event import Event

log = logging.getLogger(__name__)

BACKOFF_SECONDS = 1.0
MAX_ERROR_CHARS = 200

#: Required response format — mirrors the "Output format" section of the criteria.
RISK_JSON_SCHEMA = {
    "name": "risk_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            "reason": {"type": "string"},
            "description": {"type": "string"},
            "confidence": {"type": "number"},
            "people": {"type": "integer"},
        },
        "required": ["risk", "reason", "description", "confidence", "people"],
        "additionalProperties": False,
    },
}


class AIUnavailable(RuntimeError):
    """The AI did not return anything usable. The message never carries a secret."""


@dataclass(frozen=True, slots=True)
class Usage:
    """What the call consumed — `cached_tokens` shows whether the cache kicked in."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0

    @classmethod
    def from_payload(cls, data: Any) -> Usage:
        if not isinstance(data, dict):
            return cls()
        details = data.get("prompt_tokens_details") or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            cached_tokens=int(details.get("cached_tokens") or 0),
            cost=float(data.get("cost") or 0.0),
        )


@dataclass(frozen=True, slots=True)
class AIResponse:
    """Raw AI response, not validated yet."""

    content: str
    model: str
    usage: Usage
    latency_seconds: float


class AIClient:
    """Calls the OpenRouter vision model for an `Event`. Blocks during the call."""

    def __init__(
        self,
        api_key: str,
        criteria: str,
        *,
        model: str = "google/gemini-2.5-flash",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        enable_cache_control: bool = True,
        timezone: str = "America/Sao_Paulo",
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.max_retries = max_retries
        self.temperature = temperature
        self.tz = ZoneInfo(timezone)
        self._api_key = api_key
        self._system = _system_message(criteria, cache_control=enable_cache_control)
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> Self:
        settings.secrets.require(*AI_SECRETS)
        a = settings.ai
        return cls(
            settings.secrets.openrouter_api_key.get_secret_value(),
            settings.criteria_text(),
            model=a.model,
            base_url=a.base_url,
            timeout_seconds=a.timeout_seconds,
            max_retries=a.max_retries,
            temperature=a.temperature,
            enable_cache_control=a.enable_cache_control,
            timezone=settings.notify.timezone,
            **kwargs,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def analyze(self, event: Event) -> AIResponse:
        """Classifies the event. Raises `AIUnavailable` on any failure."""
        if not event.images:
            raise AIUnavailable("event has no images")
        started = self._clock()
        data = self._post(self._payload(event))
        response = AIResponse(
            content=_content(data),
            model=str(data.get("model") or self.model),
            usage=Usage.from_payload(data.get("usage")),
            latency_seconds=self._clock() - started,
        )
        log.info(
            "AI answered in %.1fs (%d tokens, %d cached, US$ %.5f)",
            response.latency_seconds,
            response.usage.prompt_tokens,
            response.usage.cached_tokens,
            response.usage.cost,
        )
        return response

    def _payload(self, event: Event) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [self._system, _user_message(event, self.tz)],
            "response_format": {"type": "json_schema", "json_schema": RISK_JSON_SCHEMA},
            "usage": {"include": True},
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        problem = "no attempt"
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
                log.warning("AI: %s — retrying in %.0fs", problem, delay)
                self._sleep(delay)
            try:
                response = self._client.post(self.url, json=payload, headers=headers)
            except httpx.HTTPError as err:
                problem = f"network failure ({type(err).__name__})"
                continue
            status = response.status_code
            if status == 429 or status >= 500:
                problem = _problem(response)
                continue
            if status >= 400:  # key, credits, model: retrying won't fix it
                raise AIUnavailable(_problem(response))
            try:
                return response.json()
            except ValueError:
                raise AIUnavailable("response is not JSON") from None
        raise AIUnavailable(f"{self.max_retries + 1} attempts failed: {problem}")


def _system_message(criteria: str, *, cache_control: bool) -> dict[str, Any]:
    """Static criteria block. Nothing in it may change between calls."""
    block: dict[str, Any] = {"type": "text", "text": criteria}
    if cache_control:
        block["cache_control"] = {"type": "ephemeral"}
    return {"role": "system", "content": [block]}


def _user_message(event: Event, tz: ZoneInfo) -> dict[str, Any]:
    """Variable context (local time) + images in the order they were captured."""
    content: list[dict[str, Any]] = [{"type": "text", "text": _context_text(event, tz)}]
    content += [
        {"type": "image_url", "image_url": {"url": _data_url(image.data)}} for image in event.images
    ]
    return {"role": "user", "content": content}


def _context_text(event: Event, tz: ZoneInfo) -> str:
    local = datetime.fromtimestamp(event.started_at, tz)
    span = event.images[-1].ts - event.images[0].ts
    return (
        f"Local capture time: {local:%Y-%m-%d %H:%M:%S} ({local.tzname()}).\n"
        f"{len(event.images)} images in chronological order, covering {span:.1f} s."
    )


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _content(data: Any) -> str:
    """Response text. Some providers return blocks instead of a string."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AIUnavailable("empty response") from None
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    if not isinstance(content, str) or not content.strip():
        raise AIUnavailable("empty response")
    return content


def _problem(response: httpx.Response) -> str:
    return f"HTTP {response.status_code} {_error_detail(response)}".strip()


def _error_detail(response: httpx.Response) -> str:
    """Only the provider's error message, truncated — never the whole body or headers."""
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return ""
    return str(message)[:MAX_ERROR_CHARS]
