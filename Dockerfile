FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY ai/ ./ai/
COPY data/loader.py ./data/
COPY data/processed/preprocessed_geese_full.csv data/processed/
COPY data/processed/feat_scaler.pkl data/processed/
COPY data/processed/delta_scaler.pkl data/processed/
COPY ai/training/model/weights/bird_best.pt ai/training/model/weights/
COPY contracts/ ./contracts/
COPY tests/ ./tests/

EXPOSE 8080

CMD ["uv", "run", "bird-xai-server"]
