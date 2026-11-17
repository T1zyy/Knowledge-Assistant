# AI Knowledge Assistant

Локальный ассистент для поиска по внутренним документам. Загружает PDF с текстовым слоем, TXT и Markdown, находит релевантные фрагменты и возвращает ответ с указанием файла, версии, страницы PDF и цитаты.

**Стек** · Python 3.12 · FastAPI · HTTPX · PostgreSQL · pgvector · Redis · n8n · Alembic · Docker Compose · pytest

- **Demo** использует локальные хешированные признаки для retrieval и возвращает выдержку из найденного текста. Подходит для демонстрации API, индекса, кеша и n8n без ключа и платных запросов.
- **OpenAI** использует embeddings и Responses API. Ответ проверяется по JSON Schema. Модель может запросить метаданные найденного источника через function calling.

## Запуск

Нужны Docker и Compose. Первая сборка скачивает образы, зависимости и таблицы токенизации.

```bash
cp .env.example .env
```

В `.env` установи разные значения `API_KEY`, `INGESTION_API_KEY` и `POSTGRES_PASSWORD`. Для каждого можно использовать результат отдельного вызова `openssl rand -hex 24`. Пароль PostgreSQL в Compose подставляется в URL, поэтому используй hex-строку.

```bash
docker compose up --build -d
docker compose ps
curl -fsS --retry 10 --retry-connrefused --retry-delay 1 http://127.0.0.1:8000/ready
```

Swagger UI доступен по адресу [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Используй `API_KEY` в кнопке **Authorize**. Alembic применяет миграции при первом запуске через сервис `migrate`.

### Демонстрация за две команды

Подставь значение `API_KEY` из `.env` в переменную текущего терминала.

```bash
export API_KEY='value-from-your-dotenv'

curl -fsS http://127.0.0.1:8000/documents \
  -H "X-API-Key: $API_KEY" \
  -F 'file=@examples/vpn.txt;type=text/plain'

curl -fsS http://127.0.0.1:8000/ask \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  --data '{"question":"Какой срок действия временного VPN доступа?"}'
```

Ответ содержит `14 дней` и фрагмент из `vpn.txt`. Повтори тот же запрос, чтобы увидеть `cache_hit: true`. Вопрос про размер годовой премии вернёт `insufficient_context`. В `examples/` также есть Markdown и PDF с синтетическими регламентами.

Чтобы включить n8n, запусти `docker compose --profile automation up --build -d`, открой [127.0.0.1:5678](http://127.0.0.1:5678) и импортируй workflow. Подробные шаги и команда тестового вебхука приведены в [инструкции n8n](docs/n8n.md).

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Проверка процесса |
| GET | `/ready` | Проверка версии миграции и Redis |
| POST | `/documents` | Загрузка файла multipart. `201` для нового документа, `200` для дубля |
| GET | `/documents?limit=20&offset=0` | Метаданные документов и пагинация |
| GET | `/documents/{id}` | Метаданные одного документа |
| PUT | `/documents/{id}` | Замена файла с увеличением версии при изменении |
| DELETE | `/documents/{id}` | Удаление документа и его фрагментов |
| POST | `/ask` | Запрос JSON с полем `question` и необязательным `allow_external` |

Токен загрузки `INGESTION_API_KEY` работает только на `POST /documents`. Основной `API_KEY` даёт доступ ко всем операциям. Ошибки API содержат код и `request_id`. При превышении лимита вернётся `429` с `Retry-After`.

## OpenAI

Используй синтетические документы и отдельный пустой корпус. Для смены embedding-модели или перехода Demo → OpenAI можно создать отдельный Compose-проект через `docker compose -p knowledge-openai up --build -d` после остановки первого стека, чтобы освободить порт 8000. У каждого проекта свой том PostgreSQL.

```dotenv
AI_PROVIDER=openai
ALLOW_EXTERNAL_AI=true
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

Передача внешнему провайдеру требует подтверждения в каждой операции. При загрузке или замене передай multipart-поле `allow_external=true`. В запросе к `/ask` передай JSON-поле `"allow_external": true`. По умолчанию внешние вызовы выключены. Объём отправляемых данных ограничен вопросом, фрагментами для embeddings и выбранным контекстом для генерации. Переменные и секреты храни в локальном `.env`.

По умолчанию используется размерность 1536. Модель embeddings должна поддерживать параметр `dimensions=1536`, а модель генерации — Responses API, structured outputs и function calling. При смене только модели генерации можно сохранить корпус. Ключ кеша включает модель.

## Как устроено

```text
Файл      → FastAPI / n8n → извлечение → чанкинг → embeddings → PostgreSQL + pgvector
Вопрос    → Redis rate limit → проверка кеша → top-k → генерация → проверка ссылок
Изменение → одна транзакция PostgreSQL → новая версия корпуса → новый ключ кеша
```

В PostgreSQL хранятся метаданные документов и текстовые фрагменты. Сбой embeddings до записи оставляет предыдущую версию доступной. При записи проверяется версия документа, а повторная загрузка дедуплицируется по SHA-256. На небольшом корпусе pgvector выполняет точный поиск по косинусному расстоянию.

Ответ ссылается только на фрагменты из текущей выборки. Приложение само формирует цитаты и проверяет идентификаторы, возвращённые моделью. Сверка идентификаторов защищает структуру источников. Смысловую полноту ответов удобно проверять на контрольных примерах.

Проект рассчитан на локальную демонстрацию с одним корпусом. У PDF должен быть текстовый слой. Для публичного развёртывания потребуются отдельные права пользователей, изоляция обработки файлов, TLS и анализ требований к данным.

## Тесты и оценка

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
pytest -q
ruff check .
```

Unit-тесты используют моки OpenAI. Для интеграционных тестов установи `TEST_DATABASE_URL` для отдельной базы с именем, заканчивающимся на `_test`, и `TEST_REDIS_URL`. Каждый тест создаёт собственную схему PostgreSQL и префикс ключей Redis. В GitHub Actions тестовые сервисы запускаются автоматически.

Скрипт оценки загружает три синтетических документа и проверяет 12 вопросов. Последний локальный запуск Demo прошёл **12 из 12**. Это проверка поведения на демонстрационном корпусе. Результаты моделей можно сравнить одним и тем же скриптом.

```bash
python scripts/evaluate.py --label demo --output eval/results/demo.json
```

Подробности оценки и сценарий сравнения с LangFlow описаны в [документации эксперимента](docs/evaluation.md).

## Код проекта

- `app/api.py` и `app/main.py` — маршруты, авторизация, лимиты тела запроса и обработка ошибок.
- `app/documents.py`, `app/service.py` — загрузка, чанкинг, retrieval и проверка источников.
- `app/ai.py` — локальный провайдер, OpenAI HTTP API, повтор запросов и вызов инструмента.
- `app/repository.py`, `app/cache.py` — транзакции, pgvector, Redis-кеш и rate limiter.
- `workflows/`, `examples/`, `tests/` — n8n, синтетические данные и проверки.

Основные внешние API описаны в документации [pgvector](https://github.com/pgvector/pgvector), [OpenAI Embeddings](https://developers.openai.com/api/docs/guides/embeddings), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) и [Function Calling](https://developers.openai.com/api/docs/guides/function-calling).
