#!/bin/bash
set -e

# run bash infra/docker_build_etl.sh from project root directory to ensure build context is correctly specified.

IMAGE_NAME="bituslabs-ds-etl"
TAG="latest"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-west-2"
ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME"
echo "ECR_URL: $ECR_URL"

# 1. Ensure the repository exists in ECR
if ! aws ecr describe-repositories --repository-names "$IMAGE_NAME" --region "$REGION" > /dev/null 2>&1; then
    echo "Creating repository $IMAGE_NAME..."
    aws ecr create-repository --repository-name "$IMAGE_NAME" --region "$REGION"
else
    echo "Repository $IMAGE_NAME already exists. Skipping creation."
fi

# Login to ECR
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Build the Docker image
docker build --platform linux/amd64 -t $IMAGE_NAME -f infra/Dockerfile.etl ./

# Tag and push to ECR
docker tag $IMAGE_NAME "$ECR_URL:$TAG"
docker push "$ECR_URL:$TAG"

echo "Image pushed to: $ECR_URL:$TAG"
