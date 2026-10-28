# Image for the 24/7 service. Config and secrets do NOT go into the image: config/ is
# mounted as a volume and secrets arrive through env_file (see docker-compose.yml).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# dependencies first: this layer is only rebuilt when uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
# runs as a regular user; compose swaps in the UID of the owner of ./data
USER 1000:1000
CMD ["python", "-m", "ai_security_guard", "--config", "config/config.yaml"]
