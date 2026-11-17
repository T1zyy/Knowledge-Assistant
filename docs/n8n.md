# Загрузка документов из n8n

Workflow принимает один файл, передаёт его в FastAPI и возвращает результат отправителю вебхука. Содержимое файла и ключи не сохраняются в истории штатных запусков. Для импорта JSON секреты не нужны. [Webhook node](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.webhook/) · [HTTP Request node](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.httprequest/)

1. Запусти `docker compose --profile automation up --build -d`. В интерфейсе `http://127.0.0.1:5678` пройди первичную настройку.
2. Импортируй `workflows/ingest.n8n.json` через **Import from File**.
3. К узлу **Receive document** привяжи credential **Header Auth**. Имя заголовка `X-Webhook-Key`, значение — отдельный секрет, который использует вызывающая сторона.
4. К узлу **Index in FastAPI** привяжи другой credential **Header Auth**. Имя заголовка `X-API-Key`, значение — `INGESTION_API_KEY` из `.env`.
5. Сохрани и опубликуй workflow. Для работы внутри Compose URL узла — `http://api:8000/documents`. Если оба сервиса запущены напрямую, укажи локальный URL FastAPI.
6. Проверь загрузку и её повтор.

```bash
export WEBHOOK_KEY='value-from-n8n-inbound-credential'
curl -sS -i http://127.0.0.1:5678/webhook/knowledge-assistant/documents \
  -H "X-Webhook-Key: $WEBHOOK_KEY" \
  -F 'file=@examples/vpn.txt;type=text/plain'
```

Первая загрузка возвращает `201` с `status: indexed` и метаданными. Для повтора ожидается `200` с тем же ID. Для неподдерживаемого формата ожидается `415` с `status: failed`. Ошибка подключения к FastAPI возвращает `502`. Для разрешённых синтетических данных в режиме OpenAI добавь `-F 'allow_external=true'`.

n8n обрабатывает вебхук и передаёт HTTP-статус. Разбор файла, правила внешней передачи, идемпотентность и запись индекса находятся в FastAPI. Благодаря этому загрузка работает одинаково через Swagger, n8n и клиентский код.
