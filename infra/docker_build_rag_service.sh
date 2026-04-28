#!/usr/bin/env bash
#
# Build the rag_service Docker image.
# Tag/push behavior matches the other docker_build_*.sh scripts.
#
# Usage:
#   ./infra/docker_build_rag_service.sh                  # build only
#   ./infra/docker_build_rag_service.sh push             # build + push to ECR
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="${RAG_IMAGE_NAME:-rag_service}"
TAG="${RAG_IMAGE_TAG:-latest}"

docker build \
    -f infra/Dockerfile.rag_service \
    -t "${IMAGE_NAME}:${TAG}" \
    .

if [[ "${1:-}" == "push" ]]; then
    : "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID must be set}"
    : "${AWS_REGION:?AWS_REGION must be set}"
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}:${TAG}"
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    docker tag "${IMAGE_NAME}:${TAG}" "${ECR_URI}"
    docker push "${ECR_URI}"
    echo "Pushed: ${ECR_URI}"
fi
