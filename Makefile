.PHONY: install test lint run format clean docker-build

install:
	pip install -r requirements.txt -r requirements-lock.txt

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sarcasm-detector:latest .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
