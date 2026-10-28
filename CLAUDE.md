# ai_security_guard — AI risk analysis for a Tapo camera

Watches a Tapo camera over RTSP, detects motion **locally**, sends a short batch of frames
to a cheap vision model (OpenRouter → Gemini Flash), classifies the risk as
`low | medium | high` and sends an alert through the configured channel (today simulated;
Telegram is not implemented yet).

## Architecture (pipeline)

```
RTSP (Tapo) → capture.py → motion.py → event.py → ai_client.py → classifier.py → notifier.py
                  │            │           │            │               │             │
               frames       motion?     batch of N   OpenRouter      valid JSON     Alert
               (thread)    (OpenCV)    JPEG frames  (prompt cache)  + fallback   (≥ min_risk)
                                                                                     │
                                                                       channel (notify.channel):
                                                                       simulated → log only
                                                                       telegram  (not implemented)
                                                                                     │
                                                                       storage.py (SQLite + JPEGs)
```

Each stage is an isolated module with a small interface, testable without camera, network or AI.
`pipeline.py` only orchestrates.

## Stack and environment

- Ubuntu 24.04, Python 3.12, **uv** (never `pip install` directly), Docker Compose for 24/7
- Deps: `opencv-python-headless`, `numpy`, `httpx`, `pydantic`, `pyyaml`, `python-dotenv`
- Tests: `pytest`, `respx` (HTTP mocking). Lint/format: `ruff`

## Commands

```bash
uv sync                          # installs dependencies
uv run pytest -q                 # unit tests (no network)
uv run pytest -q -m live         # real AI classification of the reference video — needs .env
uv run python scripts/smoke_camera.py   # real checks: smoke_camera / _ai / _motion / _notify
uv run ruff check --fix . && uv run ruff format .
uv run python -m ai_security_guard           # runs the pipeline
docker compose up -d --build     # 24/7 service; logs: docker compose logs -f
uv run pytest -q -m bench        # benchmark: runs the reference video (no network)
uv run python tests/test_benchmark.py [video.mp4] [--ai]   # same report, for demos
```

## Configuration

- `config/config.yaml` (copy from `config.example.yaml`) — **non-secret** parameters
  (interval between frames, frames per event, cooldown, sensitivity, model, minimum level to notify).
- `.env` (copy from `.env.example`) — **secrets**: RTSP credentials, `OPENROUTER_API_KEY`,
  `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- `config/risk_criteria.md` (copy from `risk_criteria.example.md`; describes the home, kept out of git) —
  classification criteria sent as the system prompt.
  It must stay **static and at the start of the message** to benefit from prompt caching.
  Nothing dynamic (date, time, file name) goes in that block; variable context goes in the user message, after it.

## How to work (IMPORTANT)

0. If the code can be simplified, simplify it. Clean, professional architecture.
1. Write the tests first → implement → `uv run pytest -q` green → ruff clean → **stop and report**.
2. Every external dependency (RTSP, HTTP, clock, sleep) comes in through injection/parameters so it can be mocked.
3. Unit tests **never** use the network or the real camera. Real tests use `@pytest.mark.live`.
4. If something is ambiguous or needs a decision from the user, ask before implementing.
5. If a local `docs/` folder exists (not versioned), it holds the build plan (`PLAN.md`, used by
   `/phase`) and the decision log (`DECISOES.md`). Record new architecture decisions there.

## Security and privacy rules

- **Never** log the RTSP URL with its password, API keys or the Telegram token. Use `mask_url()` / `Settings.rtsp_url_safe`.
- **Never** read, print or commit `.env`. It is in `.gitignore` and blocked in `.claude/settings.json`.
- Frames are private images of the home: they are written only to `data/events/` (gitignored), with a
  retention of `storage.keep_hours` (12 h, cut by the hour). The long history stays on the camera.
- The Telegram bot only answers/sends to `TELEGRAM_CHAT_ID`. Messages from other chats are ignored.
- A notification channel never raises and never logs a secret: inherit from `BaseNotifier` and pass
  the secrets in `redact`.
- An AI failure must **not** silence an event: if the AI fails, notify as `medium` with the warning "analysis unavailable".

## Code conventions

- Type hints everywhere, small functions, `dataclass`/pydantic for data that crosses modules.
- Logging with `logging` (never `print` outside smoke-test scripts).
- Everything in English: identifiers, comments, logs and user-facing text.
- Risk enum: `RiskLevel` in `ai_security_guard/config.py` (`low < medium < high`, comparable).
