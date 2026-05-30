PYTHON ?= python3

.PHONY: install test api ui docker-up docker-down lint

install:
	cd api && $(PYTHON) -m pip install -e .[dev]

test:
	PYTHONPATH=api/src $(PYTHON) -m pytest

api:
	cd api && PYTHONPATH=src uvicorn rag_assistant_api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	cd ui && $(PYTHON) -m http.server 3000

docker-up:
	cp -n .env.example .env || true
	docker compose up --build

docker-down:
	docker compose down

lint:
	cd api && $(PYTHON) -m compileall src
