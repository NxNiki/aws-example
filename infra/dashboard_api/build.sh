#!/usr/bin/env bash
set -e

# Build and push the dashboard_api image (React SPA + data/report API) to ECR.
# Run: bash infra/dashboard_api/build.sh
# From: project root (so the build context picks up pyproject.toml, poetry.lock,
# src/, and frontend/).

IMAGE_NAME="bituslabs-ds-dashboard-api"
TAG="latest"
REGION="us-west-2"

if [[ ! -f "pyproject.toml" ]]; then
    echo "error: run from the project root (pyproject.toml not found)" >&2
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME"

echo "ECR_URL: $ECR_URL"

# 1. Ensure the repository exists in ECR
if ! aws ecr describe-repositories --repository-names "$IMAGE_NAME" --region "$REGION" > /dev/null 2>&1; then
    echo "Creating repository $IMAGE_NAME..."
    aws ecr create-repository --repository-name "$IMAGE_NAME" --region "$REGION"
else
    echo "Repository $IMAGE_NAME already exists. Skipping creation."
fi

# 2. Login to ECR
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# 3. Build the Docker image (multi-stage: vite build -> uvicorn runtime)
docker build --platform linux/amd64 -t $IMAGE_NAME -f infra/dashboard_api/Dockerfile ./

# 4. Tag and push to ECR
docker tag $IMAGE_NAME "$ECR_URL:$TAG"
docker push "$ECR_URL:$TAG"

echo "Image pushed to: $ECR_URL:$TAG"
echo ""
echo "To run locally: docker run --rm -p 8050:8050 -e AWS_PROFILE -v ~/.aws:/root/.aws:ro $IMAGE_NAME"
echo ""
echo "To deploy to ECS:"
echo "  python infra/dashboard_api/deploy_ecs.py --build-first"
echo "  # Or: bash infra/dashboard_api/build.sh && python infra/dashboard_api/deploy_ecs.py"
