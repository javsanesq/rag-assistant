PYTHON ?= python3

.PHONY: install test api ui db-upgrade db-revision docker-up docker-config smoke docker-down lint

install:
	cd api && $(PYTHON) -m pip install -e .[dev]

test:
	PYTHONPATH=api/src $(PYTHON) -m pytest

api:
	cd api && PYTHONPATH=src uvicorn rag_assistant_api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	cd ui && $(PYTHON) -m http.server 3000

db-upgrade:
	cd api && PYTHONPATH=src alembic upgrade head

db-revision:
	cd api && PYTHONPATH=src alembic revision --autogenerate -m "$(m)"

docker-up:
	cp -n .env.example .env || true
	docker compose up --build

docker-config:
	docker compose config

smoke:
	curl -f http://localhost:8000/health/live
	curl -f http://localhost:8000/health/ready

docker-down:
	docker compose down

lint:
	cd api && $(PYTHON) -m compileall src
