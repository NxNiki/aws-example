#!/bin/bash
# One-time setup: create ECS cluster, log group, task definition, and EventBridge schedule.
# Run from project root. Requires: AWS CLI, jq, and ECR image already pushed.
#
# Usage:
#   1. Create ECS cluster + log group (once per account/region).
#   2. Create task definition (replace ACCOUNT_ID and ECR_IMAGE_URI in infra/etl/ecs_task_def.json).
#   3. Create EventBridge rule + target (fill SUBNETS and SECURITY_GROUP for your VPC).
#
# Example (fill in your subnet and security group IDs):
#   export SUBNETS="subnet-abc123,subnet-def456"
#   export SECURITY_GROUP="sg-xyz789"
#   bash infra/etl/setup_schedule.sh

set -e
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/bituslabs-ds-etl:latest"
CLUSTER_NAME="${ETL_CLUSTER_NAME:-etl-cluster}"
TASK_FAMILY="${ETL_TASK_FAMILY:-etl-fish-hunter}"
LOG_GROUP="${ETL_LOG_GROUP:-/ecs/etl-fish-hunter}"
RULE_NAME="${ETL_RULE_NAME:-etl-fish-hunter-daily}"
# Required for Fargate task to run (use private subnets that can reach Redshift/bastion if needed)
SUBNETS="${SUBNETS:?Set SUBNETS (e.g. subnet-xxx,subnet-yyy)}"
SECURITY_GROUP="${SECURITY_GROUP:?Set SECURITY_GROUP (e.g. sg-xxx)}"

echo "Region: $REGION, Account: $ACCOUNT_ID"
echo "Cluster: $CLUSTER_NAME, Task: $TASK_FAMILY, Rule: $RULE_NAME"

# 1. ECS cluster
if ! aws ecs describe-clusters --clusters "$CLUSTER_NAME" --region "$REGION" --query 'clusters[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
  echo "Creating ECS cluster $CLUSTER_NAME..."
  aws ecs create-cluster --cluster-name "$CLUSTER_NAME" --region "$REGION"
else
  echo "Cluster $CLUSTER_NAME already exists."
fi

# 2. Log group
if ! aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region "$REGION" --query "logGroups[?logGroupName=='$LOG_GROUP'].logGroupName" --output text 2>/dev/null | grep -q .; then
  echo "Creating log group $LOG_GROUP..."
  aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION"
else
  echo "Log group $LOG_GROUP already exists."
fi

# 3. Task definition (use a temp file with placeholders replaced)
TASK_DEF_FILE=$(mktemp)
sed -e "s|ACCOUNT_ID|$ACCOUNT_ID|g" -e "s|ECR_IMAGE_URI|$ECR_IMAGE|g" infra/etl/ecs_task_def.json > "$TASK_DEF_FILE"
echo "Registering task definition $TASK_FAMILY..."
aws ecs register-task-definition --cli-input-json file://"$TASK_DEF_FILE" --region "$REGION" --query 'taskDefinition.revision' --output text
rm -f "$TASK_DEF_FILE"

# 4. EventBridge rule (daily at 02:00 UTC)
aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "cron(0 2 * * ? *)" \
  --state ENABLED \
  --description "Run ETL fish-hunter daily" \
  --region "$REGION"

# 5. EventBridge target (run ECS Fargate task)
CLUSTER_ARN="arn:aws:ecs:$REGION:$ACCOUNT_ID:cluster/$CLUSTER_NAME"
TASK_DEF_ARN="arn:aws:ecs:$REGION:$ACCOUNT_ID:task-definition/$TASK_FAMILY"
# Use latest task definition revision
REV=$(aws ecs describe-task-definition --task-definition "$TASK_FAMILY" --region "$REGION" --query 'taskDefinition.revision' --output text)
TASK_DEF_ARN="$TASK_DEF_ARN:$REV"

aws events put-targets \
  --rule "$RULE_NAME" \
  --targets "[
    {
      \"Id\": \"1\",
      \"Arn\": \"$CLUSTER_ARN\",
      \"RoleArn\": \"arn:aws:iam::$ACCOUNT_ID:role/ecsEventsRole\",
      \"EcsParameters\": {
        \"TaskDefinitionArn\": \"$TASK_DEF_ARN\",
        \"TaskCount\": 1,
        \"LaunchType\": \"FARGATE\",
        \"NetworkConfiguration\": {
          \"awsvpcConfiguration\": {
            \"Subnets\": $(echo "$SUBNETS" | sed 's/,/","/g' | sed 's/^/["/' | sed 's/$/"]/'),
            \"SecurityGroups\": [\"$SECURITY_GROUP\"],
            \"AssignPublicIp\": \"DISABLED\"
          }
        }
      }
    }
  ]" \
  --region "$REGION"

echo "Done. Schedule: $RULE_NAME runs daily at 02:00 UTC. Ensure IAM role ecsEventsRole exists and can run ECS tasks."
