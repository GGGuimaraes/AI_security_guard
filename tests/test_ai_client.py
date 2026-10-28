"""AI client — no real network: every HTTP call goes through respx."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from ai_security_guard.ai_client import AIClient, AIUnavailable, Usage
from ai_security_guard.config import load_settings
from ai_security_guard.event import Event, EventImage

URL = "https://openrouter.ai/api/v1/chat/completions"
TZ = ZoneInfo("America/Sao_Paulo")
NIGHT = datetime(2026, 9, 20, 3, 14, 0, tzinfo=TZ).timestamp()

CONTENT = json.dumps(
    {
        "risk": "high",
        "reason": "Person tampering with the lock.",
        "description": "Someone handles the door for several frames.",
        "confidence": 0.9,
        "people": 1,
    },
    ensure_ascii=False,
)

# --------------------------------------------------------------------------- helpers


def _event(n: int = 3, started: float = NIGHT, interval: float = 1.0) -> Event:
    images = tuple(
        EventImage(data=f"jpeg-{i}".encode(), ts=started + i * interval) for i in range(n)
    )
    return Event(started_at=started, images=images, area_ratio=0.05)


def _reply(content: str = CONTENT, **usage) -> dict:
    return {
        "model": "google/gemini-2.5-flash",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": 1500,
            "completion_tokens": 60,
            "total_tokens": 1560,
            "cost": 0.00042,
            "prompt_tokens_details": {"cached_tokens": 1200},
            **usage,
        },
    }


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def settings(fake_env, example_config):
    return load_settings(example_config, env=fake_env)


@pytest.fixture
def client(settings):
    sleep = FakeSleep()
    with AIClient.from_settings(settings, sleep=sleep) as ai:
        ai.sleep_calls = sleep.calls  # shortcut for the retry tests
        yield ai


def _sent(route) -> dict:
    return json.loads(route.calls.last.request.content)


# --------------------------------------------------------------------------- payload


@respx.mock
def test_system_comes_first_with_the_criteria_file(client, settings):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    client.analyze(_event())

    messages = _sent(route)["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"][0]["text"] == settings.criteria_text()


@respx.mock
def test_system_is_identical_across_calls_and_has_nothing_dynamic(client):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    client.analyze(_event(started=NIGHT))
    first = _sent(route)["messages"][0]
    client.analyze(_event(started=NIGHT + 86400))
    second = _sent(route)["messages"][0]

    assert first == second  # byte-for-byte identical prefix: what the cache requires
    text = json.dumps(first, ensure_ascii=False)
    assert "03:14" not in text and "2026" not in text


@respx.mock
def test_cache_control_on_the_system_block(client):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    client.analyze(_event())

    block = _sent(route)["messages"][0]["content"][0]
    assert block["cache_control"] == {"type": "ephemeral"}


@respx.mock
def test_cache_control_disabled(settings):
    settings.ai.enable_cache_control = False
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    with AIClient.from_settings(settings) as ai:
        ai.analyze(_event())

    assert "cache_control" not in _sent(route)["messages"][0]["content"][0]


@respx.mock
def test_user_message_has_local_time_and_images_in_order(client):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    client.analyze(_event(n=3))

    blocks = _sent(route)["messages"][1]["content"]
    assert blocks[0]["type"] == "text"
    assert "2026-09-20 03:14" in blocks[0]["text"]
    urls = [b["image_url"]["url"] for b in blocks[1:]]
    assert len(urls) == 3
    assert all(u.startswith("data:image/jpeg;base64,") for u in urls)
    assert [_decode(u) for u in urls] == [b"jpeg-0", b"jpeg-1", b"jpeg-2"]


def _decode(url: str) -> bytes:
    import base64

    return base64.b64decode(url.split(",", 1)[1])


@respx.mock
def test_payload_asks_for_json_schema_usage_and_model(client, settings):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    client.analyze(_event())

    sent = _sent(route)
    assert sent["model"] == settings.ai.model
    assert sent["temperature"] == settings.ai.temperature
    assert sent["usage"] == {"include": True}
    schema = sent["response_format"]["json_schema"]["schema"]
    assert sent["response_format"]["type"] == "json_schema"
    assert schema["properties"]["risk"]["enum"] == ["low", "medium", "high"]
    assert set(schema["required"]) == {"risk", "reason", "description", "confidence", "people"}


@respx.mock
def test_authorization_goes_in_the_header(client):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    client.analyze(_event())

    assert route.calls.last.request.headers["authorization"] == "Bearer sk-or-v1-abcdefghijk"


def test_repr_does_not_show_the_key(client):
    assert "sk-or-v1-abcdefghijk" not in repr(client)


# --------------------------------------------------------------------------- response


@respx.mock
def test_parse_returns_content_and_cached_tokens(client):
    respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    result = client.analyze(_event())

    assert result.content == CONTENT
    assert result.model == "google/gemini-2.5-flash"
    assert result.usage == Usage(
        prompt_tokens=1500,
        completion_tokens=60,
        cached_tokens=1200,
        cost=0.00042,
    )
    assert result.latency_seconds >= 0


@respx.mock
def test_content_as_text_blocks(client):
    reply = _reply()
    reply["choices"][0]["message"]["content"] = [{"type": "text", "text": CONTENT}]
    respx.post(URL).mock(return_value=httpx.Response(200, json=reply))

    assert client.analyze(_event()).content == CONTENT


@respx.mock
def test_missing_usage_does_not_break(client):
    reply = _reply()
    del reply["usage"]
    respx.post(URL).mock(return_value=httpx.Response(200, json=reply))

    assert client.analyze(_event()).usage == Usage()


@respx.mock
@pytest.mark.parametrize(
    "body",
    [{"choices": []}, {"choices": [{"message": {}}]}, {"error": {"message": "no credits"}}],
)
def test_empty_response_becomes_ai_unavailable(client, body):
    respx.post(URL).mock(return_value=httpx.Response(200, json=body))

    with pytest.raises(AIUnavailable):
        client.analyze(_event())


@respx.mock
def test_non_json_body_becomes_ai_unavailable(client):
    respx.post(URL).mock(return_value=httpx.Response(200, text="<html>502</html>"))

    with pytest.raises(AIUnavailable):
        client.analyze(_event())


# --------------------------------------------------------------------------- retries


@respx.mock
@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_on_429_and_5xx_until_it_works(client, status):
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(status),
            httpx.Response(status),
            httpx.Response(200, json=_reply()),
        ]
    )

    result = client.analyze(_event())

    assert result.content == CONTENT
    assert route.call_count == 3
    assert client.sleep_calls == [1.0, 2.0]  # doubling backoff


@respx.mock
def test_network_error_is_retried_too(client):
    route = respx.post(URL).mock(
        side_effect=[httpx.ConnectTimeout("timeout"), httpx.Response(200, json=_reply())]
    )

    assert client.analyze(_event()).content == CONTENT
    assert route.call_count == 2


@respx.mock
def test_running_out_of_attempts_becomes_ai_unavailable(client):
    route = respx.post(URL).mock(return_value=httpx.Response(503))

    with pytest.raises(AIUnavailable, match="503"):
        client.analyze(_event())

    assert route.call_count == 3  # 1 + max_retries (2)


@respx.mock
def test_client_4xx_is_not_retried(client):
    route = respx.post(URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid key"}})
    )

    with pytest.raises(AIUnavailable, match="401"):
        client.analyze(_event())

    assert route.call_count == 1
    assert client.sleep_calls == []


@respx.mock
def test_event_without_images_does_not_call_the_ai(client):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=_reply()))

    with pytest.raises(AIUnavailable, match="no images"):
        client.analyze(Event(started_at=NIGHT, images=()))

    assert route.call_count == 0
