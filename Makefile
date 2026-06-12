# Makefile for bituslabs_ds project

.PHONY: help install install-dev test test-unit test-integration test-coverage test-fast lint lint-new format clean build docs openapi

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
	@echo "  openapi        Regenerate the dashboard_api OpenAPI schema + frontend TS client"
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

# CI lint scope (Phase 0): the new/migrated React-redesign surfaces only. The
# legacy tree has pre-existing flake8 debt (see .flake8) and is NOT gated yet;
# add paths here as the rewrite ports each module so the gate tightens tab by
# tab. `make lint` above still covers the whole repo for local cleanup work.
LINT_PATHS_NEW = src/dashboard_api src/bituslabs_ds/metrics src/bituslabs_ds/confluence src/bituslabs_ds/aws_secrets.py \
	tests/unit/test_dashboard_api.py tests/unit/test_ai_agent_actions.py

lint-new:
	poetry run flake8 $(LINT_PATHS_NEW)
	poetry run mypy $(LINT_PATHS_NEW)
	poetry run black --check $(LINT_PATHS_NEW)
	poetry run isort --check-only $(LINT_PATHS_NEW)

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

# Regenerate the OpenAPI schema from dashboard_api and the frontend's typed
# client from it. Run after changing any dashboard_api request/response model.
openapi:
	bash scripts/gen_openapi_client.sh

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

