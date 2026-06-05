#!/usr/bin/env bash
# Build the dashboard_api image (SPA + API) from the project root.
#
#   bash infra/dashboard_api/build.sh
#
# Run from the repo root so the Docker build context picks up pyproject.toml,
# poetry.lock, src/, and frontend/.
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-dashboard-api}"
TAG="${TAG:-latest}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "error: run from the project root (pyproject.toml not found)" >&2
  exit 1
fi

docker build -f infra/dashboard_api/Dockerfile -t "${IMAGE_NAME}:${TAG}" .
echo "built ${IMAGE_NAME}:${TAG}"
