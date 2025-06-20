#!/bin/bash
set -e

# run bash infra/docker_build.sh from project root directory to ensure build context is correctly specified.

IMAGE_NAME="bituslabs-ds-sagemaker"
TAG="latest"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-west-2"
ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME"
echo "ECR_URL: $ECR_URL"

# Login to ECR
#aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 246618743249.dkr.ecr.us-west-2.amazonaws.com
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 338568447110.dkr.ecr.us-west-2.amazonaws.com

# Build the Docker image
docker build --platform linux/amd64 -t $IMAGE_NAME -f infra/Dockerfile ./

# Tag and push to ECR
docker tag $IMAGE_NAME "$ECR_URL:$TAG"
docker push "$ECR_URL:$TAG"

echo "Image pushed to: $ECR_URL:$TAG"
