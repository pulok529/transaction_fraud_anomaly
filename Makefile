.PHONY: help install train test lint serve docker-build docker-run clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install       Install development and runtime dependencies"
	@echo "  make train         Run full synthetic training pipeline"
	@echo "  make test          Run pytest test suite"
	@echo "  make lint          Run ruff linter check"
	@echo "  make serve         Start FastAPI development server with auto-reload"
	@echo "  make docker-build  Build production Docker container"
	@echo "  make docker-run    Run production container on port 8000"

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements-dev.txt

train:
	$(PYTHON) -m fraud_anomaly.train --n 50000 --out artifacts/

test:
	pytest -v

lint:
	ruff check .

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t fraud-anomaly:latest .

docker-run:
	docker run --rm -p 8000:8000 --name fraud-anomaly-api fraud-anomaly:latest

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/
