from datetime import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ia_watcher.config import (
    AI_SECRETS,
    CAMERA_SECRETS,
    TELEGRAM_SECRETS,
    MissingSecrets,
    RiskLevel,
    load_settings,
    mask_url,
)

ALL_SECRETS = CAMERA_SECRETS + AI_SECRETS + TELEGRAM_SECRETS


def test_load_example(example_config, fake_env):
    s = load_settings(example_config, env=fake_env)
    assert s.camera.port == 554
    assert s.event.frames_per_event == 4
    assert s.notify.min_risk is RiskLevel.MEDIUM
    assert s.secrets.telegram_chat_id == "987654"
    assert s.health.daily_at == time(8, 0)
    assert s.health.camera_offline_minutes == 10


def test_daily_heartbeat_can_be_disabled(tmp_path, example_config, fake_env):
    raw = yaml.safe_load(Path(example_config).read_text(encoding="utf-8"))
    raw["health"] = {"daily_at": None}
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_settings(cfg, env=fake_env).health.daily_at is None


def test_secret_is_required_only_where_it_is_used(example_config, fake_env):
    del fake_env["TELEGRAM_BOT_TOKEN"]
    del fake_env["TELEGRAM_CHAT_ID"]
    s = load_settings(example_config, env=fake_env)  # loads: the camera doesn't use Telegram

    assert s.rtsp_url.startswith("rtsp://ia_watcher:")
    s.secrets.require(*CAMERA_SECRETS)
    with pytest.raises(MissingSecrets, match="TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"):
        s.secrets.require(*ALL_SECRETS)


def test_missing_camera_secret_blocks_rtsp_url(example_config, fake_env):
    del fake_env["TAPO_PASSWORD"]
    s = load_settings(example_config, env=fake_env)
    with pytest.raises(MissingSecrets, match="TAPO_PASSWORD"):
        _ = s.rtsp_url


def test_require_lists_env_names_not_values(example_config, fake_env):
    s = load_settings(example_config, env={})
    with pytest.raises(MissingSecrets) as err:
        s.secrets.require(*ALL_SECRETS)
    assert "tapo_password" not in str(err.value)  # .env names, not field names
    assert "TAPO_PASSWORD" in str(err.value)


def test_secrets_never_in_repr(example_config, fake_env):
    s = load_settings(example_config, env=fake_env)
    dumped = repr(s) + str(s)
    for key in ("TAPO_PASSWORD", "OPENROUTER_API_KEY", "TELEGRAM_BOT_TOKEN"):
        assert fake_env[key] not in dumped
    assert fake_env["TAPO_PASSWORD"] not in s.rtsp_url_safe


def test_rtsp_url_escapes_password(example_config, fake_env):
    s = load_settings(example_config, env=fake_env)
    assert s.rtsp_url.startswith("rtsp://ia_watcher:")
    assert "p4ss%40%3Asecret%2F1@" in s.rtsp_url
    assert s.rtsp_url.endswith(":554/stream1")


def test_risk_order_and_parse():
    assert RiskLevel.LOW < RiskLevel.MEDIUM < RiskLevel.HIGH
    assert RiskLevel.HIGH >= RiskLevel.MEDIUM
    assert RiskLevel.parse(" Medium ") is RiskLevel.MEDIUM
    with pytest.raises(ValueError):
        RiskLevel.parse("critical")


def test_criteria_file_is_static_and_substantial(example_config, fake_env):
    text = load_settings(example_config, env=fake_env).criteria_text()
    # large enough for implicit caching, and no dynamic fields
    assert len(text) > 4000
    assert all(level in text for level in ("low", "medium", "high"))
    assert "{{" not in text


def _write_cfg(tmp_path: Path, example_config: str, patch: dict) -> Path:
    raw = yaml.safe_load(Path(example_config).read_text())
    for section, values in patch.items():
        raw[section].update(values)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_invalid_min_risk(tmp_path, example_config, fake_env):
    p = _write_cfg(tmp_path, example_config, {"notify": {"min_risk": "critical"}})
    with pytest.raises(ValidationError):
        load_settings(p, env=fake_env)


def test_unknown_key_rejected(tmp_path, example_config, fake_env):
    p = _write_cfg(tmp_path, example_config, {"event": {"frames_per_evnt": 3}})
    with pytest.raises(ValidationError):
        load_settings(p, env=fake_env)


def test_mask_url():
    assert mask_url("rtsp://u:p@10.0.0.1:554/stream1") == "rtsp://u:***@10.0.0.1:554/stream1"
    assert mask_url("rtsp://10.0.0.1/stream1") == "rtsp://10.0.0.1/stream1"
    assert mask_url("/tmp/sample.mp4") == "/tmp/sample.mp4"
