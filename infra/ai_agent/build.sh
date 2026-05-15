#!/bin/bash
set -e

# Build and push the AI Chat Agent Docker image to ECR.
# Run: bash infra/ai_agent/build.sh
# From: project root directory (to ensure build context is correct).

IMAGE_NAME="bituslabs-ds-ai-agent"
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

# 2. Login to ECR
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# 3. Build the Docker image
docker build --platform linux/amd64 -t $IMAGE_NAME -f infra/ai_agent/Dockerfile ./

# 4. Tag and push to ECR
docker tag $IMAGE_NAME "$ECR_URL:$TAG"
docker push "$ECR_URL:$TAG"

echo "Image pushed to: $ECR_URL:$TAG"
echo ""
echo "To run locally: docker run -p 8051:8051 -e GOOGLE_API_KEY -e OPENAI_API_KEY $IMAGE_NAME"
echo ""
echo "To deploy to ECS:"
echo "  python infra/ai_agent/deploy_ecs.py --build-first"
echo "  # Or: bash infra/ai_agent/build.sh && python infra/ai_agent/deploy_ecs.py"
