from pathlib import Path

import pytest
import yaml

FAKE_ENV = {
    "TAPO_USERNAME": "ia_watcher",
    "TAPO_PASSWORD": "p4ss@:secret/1",
    "OPENROUTER_API_KEY": "sk-or-v1-abcdefghijk",
    "TELEGRAM_BOT_TOKEN": "123456:ABCDEF-token",
    "TELEGRAM_CHAT_ID": "987654",
}


@pytest.fixture
def fake_env() -> dict[str, str]:
    return dict(FAKE_ENV)


@pytest.fixture
def example_config(tmp_path: Path) -> str:
    """The example config with the example criteria — the real criteria are not in git."""
    raw = yaml.safe_load(Path("config/config.example.yaml").read_text(encoding="utf-8"))
    raw["ai"]["criteria_file"] = "config/risk_criteria.example.md"
    path = tmp_path / "example.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return str(path)
