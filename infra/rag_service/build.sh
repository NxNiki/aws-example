#!/bin/bash
set -e

# Build and push the RAG service Docker image to ECR.
# Run: bash infra/rag_service/build.sh
# From: project root directory (to ensure build context is correct).

IMAGE_NAME="bituslabs-ds-rag-service"
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

# 3. Build the Docker image (linux/amd64 for Fargate compatibility on Apple Silicon)
docker build --platform linux/amd64 -t $IMAGE_NAME -f infra/rag_service/Dockerfile ./

# 4. Tag and push to ECR
docker tag $IMAGE_NAME "$ECR_URL:$TAG"
docker push "$ECR_URL:$TAG"

echo "Image pushed to: $ECR_URL:$TAG"
echo ""
echo "To run locally: docker run -p 8052:8052 -e GOOGLE_API_KEY $IMAGE_NAME"
echo ""
echo "To deploy to ECS:"
echo "  python infra/rag_service/deploy_ecs.py --build-first"
echo "  # Or: bash infra/rag_service/build.sh && python infra/rag_service/deploy_ecs.py"
