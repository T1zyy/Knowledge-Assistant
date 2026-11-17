FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR=/opt/tiktoken
WORKDIR /srv/assistant

COPY pyproject.toml alembic.ini ./
COPY app ./app
COPY migrations ./migrations
RUN pip install --no-cache-dir . \
    && python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base"); tiktoken.get_encoding("o200k_base")' \
    && useradd --system --uid 10001 --no-create-home assistant

USER assistant
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
