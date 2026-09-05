.PHONY: install run-api run-worker up down test lint cli

install:
	pip install -r requirements.txt

run-api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-worker:
	celery -A app.orchestration.celery_app.celery_app worker -l info --concurrency=1

up:
	docker compose up --build

down:
	docker compose down -v

test:
	python -m pytest tests/ -q

cli:
	python scripts/cli.py --help
