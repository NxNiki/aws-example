# Makefile for bituslabs_ds project

.PHONY: help install install-dev test test-unit test-integration test-coverage test-fast lint format clean build docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  test-fast      Run fast tests (exclude slow tests)"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  clean          Clean build artifacts"
	@echo "  build          Build the package"
	@echo "  docs           Build documentation"

# Installation
install:
	poetry install --only main

install-dev:
	poetry install --with dev,test

# Testing
test:
	poetry run pytest

test-unit:
	poetry run pytest tests/unit/ -m "not slow"

test-integration:
	poetry run pytest tests/integration/

test-coverage:
	poetry run pytest --cov=bituslabs_ds --cov-report=html --cov-report=term-missing

test-fast:
	poetry run pytest -m "not slow"

test-parallel:
	poetry run pytest -n auto

# Code quality
lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	poetry build

# Documentation
docs:
	cd docs && make html

# Development helpers
dev-setup: install-dev
	poetry run pre-commit install

run-example:
	poetry run python jobs/examples/data_loader_s3_example.py

# CI/CD helpers
ci-test:
	poetry run pytest --cov=bituslabs_ds --cov-report=xml --junitxml=test-results.xml

ci-lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

# Docker helpers (if needed)
docker-build:
	docker build -t bituslabs-ds .

docker-test:
	docker run --rm bituslabs-ds pytest

# AWS helpers (if needed)
aws-test:
	AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test poetry run pytest -m aws

# Performance testing
perf-test:
	poetry run pytest -m "not slow" --durations=10

# Security scanning (if needed)
security-scan:
	poetry run safety check
	poetry run bandit -r src/

# All checks
check-all: lint test-coverage
	@echo "All checks passed!"

