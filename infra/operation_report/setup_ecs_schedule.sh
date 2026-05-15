#!/bin/bash
# Setup scheduled run of operation daily report (EventBridge + ECS Fargate).
# Runs jobs/operation_daily_report/run_daily_report.py daily and sends the report to Slack.
#
# Prerequisites:
#   1. Build and push ETL image: bash infra/etl/build.sh
#   2. Store secrets in Secrets Manager:
#      - etl/bastion-key (existing)
#      - etl/slack-bot-token: Slack Bot OAuth token (xoxb-...)
#      - etl/slack-channel-id: Slack channel ID (e.g. C01234567) or channel name
#   3. Create ecsEventsRole (one-time) for EventBridge to run ECS tasks
#
# Usage:
#   export SUBNETS="subnet-xxx,subnet-yyy"
#   export SECURITY_GROUP="sg-xxx"
#   bash infra/operation_report/setup_ecs_schedule.sh

set -e
REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/bituslabs-ds-etl:latest"
CLUSTER_NAME="${ETL_CLUSTER_NAME:-etl-cluster}"
TASK_FAMILY="etl-operation-daily-report"
LOG_GROUP="/ecs/etl-operation-daily-report"
RULE_NAME="operation-daily-report-daily"
# Schedule: daily at 14:30 UTC (22:30 Beijing) - adjust as needed
SCHEDULE="${DAILY_REPORT_SCHEDULE:-cron(30 14 * * ? *)}"
SUBNETS="${SUBNETS:?Set SUBNETS (e.g. subnet-xxx,subnet-yyy)}"
SECURITY_GROUP="${SECURITY_GROUP:?Set SECURITY_GROUP (e.g. sg-xxx)}"

echo "Region: $REGION, Account: $ACCOUNT_ID"
echo "Cluster: $CLUSTER_NAME, Task: $TASK_FAMILY, Rule: $RULE_NAME"
echo "Schedule: $SCHEDULE"

# 1. ECS cluster (reuse existing etl-cluster)
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

# 3. Task definition
TASK_DEF_FILE=$(mktemp)
sed -e "s|ACCOUNT_ID|$ACCOUNT_ID|g" -e "s|ECR_IMAGE_URI|$ECR_IMAGE|g" infra/operation_report/ecs_task_def.json > "$TASK_DEF_FILE"
echo "Registering task definition $TASK_FAMILY..."
aws ecs register-task-definition --cli-input-json file://"$TASK_DEF_FILE" --region "$REGION" --query 'taskDefinition.revision' --output text
rm -f "$TASK_DEF_FILE"

# 4. EventBridge rule
aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "$SCHEDULE" \
  --state ENABLED \
  --description "Run operation daily report and send to Slack" \
  --region "$REGION"

# 5. EventBridge target
CLUSTER_ARN="arn:aws:ecs:$REGION:$ACCOUNT_ID:cluster/$CLUSTER_NAME"
REV=$(aws ecs describe-task-definition --task-definition "$TASK_FAMILY" --region "$REGION" --query 'taskDefinition.revision' --output text)
TASK_DEF_ARN="arn:aws:ecs:$REGION:$ACCOUNT_ID:task-definition/$TASK_FAMILY:$REV"
SUBNETS_JSON=$(echo "$SUBNETS" | sed 's/,/","/g' | sed 's/^/["/' | sed 's/$/"]/')

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
            \"Subnets\": $SUBNETS_JSON,
            \"SecurityGroups\": [\"$SECURITY_GROUP\"],
            \"AssignPublicIp\": \"DISABLED\"
          }
        }
      }
    }
  ]" \
  --region "$REGION"

echo "Done. Operation daily report scheduled: $RULE_NAME."
echo "Ensure secrets etl/slack-bot-token and etl/slack-channel-id exist in Secrets Manager."
echo "Grant ecsTaskExecutionRole secretsmanager:GetSecretValue on those secrets."
