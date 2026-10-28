"""Loads and validates the configuration (non-secret YAML + secrets from the environment)."""

from __future__ import annotations

import operator
import os
from datetime import time as dtime
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import quote, urlsplit, urlunsplit

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

_ORDER = {"low": 0, "medium": 1, "high": 2}


def _by_order(op):
    """Compares by severity, not by the alphabetical order inherited from `str`."""

    def compare(self: RiskLevel, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return op(_ORDER[self.value], _ORDER[other.value])

    return compare


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def parse(cls, value: str) -> RiskLevel:
        """Accepts common variations ('Medium', 'HIGH', ' low ')."""
        return cls(value.strip().lower())

    __lt__ = _by_order(operator.lt)
    __le__ = _by_order(operator.le)
    __gt__ = _by_order(operator.gt)
    __ge__ = _by_order(operator.ge)


def mask_url(url: str) -> str:
    """Strips the password from a URL for logging: 'rtsp://u:p@host/s' -> 'rtsp://u:***@host/s'."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    netloc = f"{parts.username or ''}:***@{parts.hostname or ''}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraConfig(_Strict):
    host: str
    port: int = 554
    stream: str = "stream1"
    reconnect_seconds: float = Field(5, gt=0)


class MotionConfig(_Strict):
    detect_width: int = Field(320, ge=64)
    min_area_ratio: float = Field(0.01, gt=0, lt=1)
    trigger_frames: int = Field(2, ge=1, le=10)  # consecutive motion frames to open an event
    warmup_frames: int = Field(30, ge=0)
    ignore_zones: list[list[float]] = Field(default_factory=list)  # validated by MotionDetector


class EventConfig(_Strict):
    frames_per_event: int = Field(4, ge=1, le=10)
    frame_interval_seconds: float = Field(1.0, ge=0)
    cooldown_seconds: float = Field(60, ge=0)
    max_image_width: int = Field(1024, ge=128)
    jpeg_quality: int = Field(80, ge=10, le=100)


class AIConfig(_Strict):
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash"
    criteria_file: Path = Path("config/risk_criteria.md")
    timeout_seconds: float = Field(30, gt=0)
    max_retries: int = Field(2, ge=0)
    temperature: float = Field(0.0, ge=0, le=2)
    enable_cache_control: bool = True


class NotifyConfig(_Strict):
    channel: str = "simulated"  # output channel; see notifier.CHANNELS
    min_risk: RiskLevel = RiskLevel.MEDIUM
    max_images: int = Field(4, ge=0, le=10)
    timezone: str = "America/Sao_Paulo"

    @field_validator("min_risk", mode="before")
    @classmethod
    def _parse_risk(cls, v: object) -> object:
        return RiskLevel.parse(v) if isinstance(v, str) else v


class StorageConfig(_Strict):
    events_dir: Path = Path("data/events")
    db_path: Path = Path("data/ai_security_guard.db")
    keep_hours: int = Field(12, ge=1)  # retention for images of the home


class HealthConfig(_Strict):
    daily_at: dtime | None = dtime(8, 0)  # daily "is running" heartbeat (None disables)
    camera_offline_minutes: float = Field(10, gt=0)  # no frames for longer than this = alert


class LoggingConfig(_Strict):
    level: str = "INFO"


_SECRET_ENV = {
    "tapo_username": "TAPO_USERNAME",
    "tapo_password": "TAPO_PASSWORD",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "TELEGRAM_CHAT_ID",
}

# Secrets each stage needs — required where they are used, not at load time.
CAMERA_SECRETS = ("tapo_username", "tapo_password")
AI_SECRETS = ("openrouter_api_key",)
TELEGRAM_SECRETS = ("telegram_bot_token", "telegram_chat_id")


class MissingSecrets(ValueError):
    """A secret required by the stage in use is missing from .env."""


class Secrets(BaseModel):
    """Secrets from .env. Optional here; each stage calls `require()` with what it needs."""

    tapo_username: str | None = None
    tapo_password: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Self:
        env = dict(os.environ) if env is None else env
        return cls(**{field: env[key] for field, key in _SECRET_ENV.items() if env.get(key)})

    def require(self, *fields: str) -> Self:
        """Ensures the requested secrets exist; the error names the missing .env variables."""
        missing = [_SECRET_ENV[f] for f in fields if getattr(self, f) is None]
        if missing:
            raise MissingSecrets(f"missing variables in .env: {', '.join(missing)}")
        return self


class Settings(_Strict):
    camera: CameraConfig
    motion: MotionConfig = MotionConfig()
    event: EventConfig = EventConfig()
    ai: AIConfig = AIConfig()
    notify: NotifyConfig = NotifyConfig()
    storage: StorageConfig = StorageConfig()
    health: HealthConfig = HealthConfig()
    logging: LoggingConfig = LoggingConfig()
    secrets: Secrets

    @model_validator(mode="after")
    def _check_criteria(self) -> Self:
        if not self.ai.criteria_file.is_file():
            raise ValueError(f"criteria file not found: {self.ai.criteria_file}")
        return self

    @property
    def rtsp_url(self) -> str:
        """Full URL with credentials. Never log it — use rtsp_url_safe."""
        self.secrets.require(*CAMERA_SECRETS)
        user = quote(self.secrets.tapo_username, safe="")
        pwd = quote(self.secrets.tapo_password.get_secret_value(), safe="")
        c = self.camera
        return f"rtsp://{user}:{pwd}@{c.host}:{c.port}/{c.stream}"

    @property
    def rtsp_url_safe(self) -> str:
        """Same URL without the password — the only version that may be logged."""
        return mask_url(self.rtsp_url)

    def criteria_text(self) -> str:
        """The criteria — static, sent as the system prompt (cacheable)."""
        return self.ai.criteria_file.read_text(encoding="utf-8")


def load_settings(
    config_path: str | Path = "config/config.yaml",
    env_file: str | Path | None = ".env",
    env: dict[str, str] | None = None,
) -> Settings:
    """Loads YAML + secrets. An explicit `env` (tests) skips the .env file."""
    if env is None and env_file and Path(env_file).is_file():
        load_dotenv(env_file, override=False)
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    raw["secrets"] = Secrets.from_env(env)
    return Settings.model_validate(raw)
