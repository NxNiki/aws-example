#!/bin/bash
set -e

# run bash infra/shared/build_base.sh from project root directory to ensure build context is correctly specified.

IMAGE_NAME="bituslabs-ds-dl"
TAG="latest"
REGION="us-west-2"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME"

# 1. Ensure the repository exists in ECR
echo "Verifying ECR repository..."
aws ecr describe-repositories --repository-names "$IMAGE_NAME" --region "$REGION" > /dev/null 2>&1 || \
aws ecr create-repository --repository-name "$IMAGE_NAME" --region "$REGION"

# 2. Login to ECR (using variables for consistency)
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# 3. Build the Docker image
docker build --platform linux/amd64 -t $IMAGE_NAME -f infra/shared/Dockerfile.base ./

# 4. Tag and push to ECR
docker tag "$IMAGE_NAME:$TAG" "$ECR_URL:$TAG"
docker push "$ECR_URL:$TAG"

echo "Success! Image pushed to: $ECR_URL:$TAG"
