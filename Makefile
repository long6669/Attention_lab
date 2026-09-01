.PHONY: setup dev backend frontend test lint format build

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
	cd frontend && npm ci

dev:
	@echo "Run 'make backend' and 'make frontend' in separate terminals."

backend:
	PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload

frontend:
	cd frontend && npm run dev

test:
	PYTHONPATH=backend .venv/bin/python -m unittest discover -s backend/tests
	cd frontend && npm run build

lint:
	.venv/bin/ruff check backend
	cd frontend && npm run lint
	cd frontend && npm run format:check

format:
	.venv/bin/ruff check --fix backend
	.venv/bin/ruff format backend
	cd frontend && npm run format

build:
	cd frontend && npm run build
