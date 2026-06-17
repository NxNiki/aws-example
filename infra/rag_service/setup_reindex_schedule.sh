#!/bin/bash
# One-time setup: daily scheduled RAG reindex.
#
# Creates a Fargate task (rag-service image, CMD overridden to run
# jobs/build_rag_index.py --refresh-ecs-service) and an EventBridge Scheduler
# schedule that fires it daily at 09:00 America/Los_Angeles (DST-aware).
#
# The build runs in its OWN throwaway task (not on the serving task), uploads
# the fresh Faiss index to S3, then force-new-deployments the rag-service so the
# live service reloads it. This avoids the failure mode of calling /reindex on
# the serving task (blocks the event loop -> health checks fail -> task killed).
#
# Prereqs:
#   - rag-service already deployed (cluster, task SG, image in ECR exist).
#     See infra/rag_service/deploy_ecs.py.
#   - AWS CLI v2 (for `aws scheduler`), permission to manage IAM/ECS/Scheduler.
#
# Usage (from project root):
#   bash infra/rag_service/setup_reindex_schedule.sh
#
# Optional env overrides: REGION, CLUSTER_NAME, SUBNETS, SECURITY_GROUP, SCHEDULE_HOUR.

set -euo pipefail

REGION="${REGION:-us-west-2}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/bituslabs-ds-rag-service:latest"

CLUSTER_NAME="${CLUSTER_NAME:-ai-team-dashboard-cluster}"
TASK_FAMILY="rag-service-reindex"
LOG_GROUP="/ecs/rag-service-reindex"
SCHEDULE_NAME="rag-service-reindex-daily"
SCHEDULE_TZ="America/Los_Angeles"
SCHEDULE_HOUR="${SCHEDULE_HOUR:-9}"            # 9 AM Pacific
TASK_ROLE_NAME="ragReindexTaskRole"            # task role: S3 r/w + secrets read + ecs:UpdateService
SCHED_ROLE_NAME="ragReindexSchedulerRole"      # scheduler role: ecs:RunTask + iam:PassRole
EXEC_ROLE_NAME="ecsTaskExecutionRole"          # shared execution role (image pull + logs)

S3_BUCKET="bituslabs-team-ai"
RAG_INDEX_PREFIX="rag"
SECRETS_MANAGER_NAME="ai-dashboard_ai_agent"
SERVICE_NAME="rag-service"

echo "Region: $REGION  Account: $ACCOUNT_ID"
echo "Cluster: $CLUSTER_NAME  Task: $TASK_FAMILY  Schedule: $SCHEDULE_NAME @ ${SCHEDULE_HOUR}:00 $SCHEDULE_TZ"

# --- Discover VPC networking (reuse the service's default-VPC subnets + task SG) ---
VPC_ID="${VPC_ID:-$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text --region "$REGION")}"
if [[ -z "${SUBNETS:-}" ]]; then
  SUBNETS="$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" \
    --query 'Subnets[].SubnetId' --output text --region "$REGION" | tr '\t' ',')"
fi
if [[ -z "${SECURITY_GROUP:-}" ]]; then
  SECURITY_GROUP="$(aws ec2 describe-security-groups \
    --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values="${SERVICE_NAME}-task-sg" \
    --query 'SecurityGroups[0].GroupId' --output text --region "$REGION")"
fi
echo "VPC: $VPC_ID  Subnets: $SUBNETS  SG: $SECURITY_GROUP"
# JSON array of subnet ids: "subnet-a","subnet-b"
SUBNETS_JSON="\"$(echo "$SUBNETS" | sed 's/,/","/g')\""

# --- 1. Log group -----------------------------------------------------------
if ! aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region "$REGION" \
      --query "logGroups[?logGroupName=='$LOG_GROUP'].logGroupName" --output text | grep -q .; then
  echo "Creating log group $LOG_GROUP..."
  aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION"
fi

# --- 2. Reindex task role (S3 read+write + secrets read + ecs:UpdateService) -
TASK_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
if ! aws iam get-role --role-name "$TASK_ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating task role $TASK_ROLE_NAME..."
  aws iam create-role --role-name "$TASK_ROLE_NAME" --assume-role-policy-document "$TASK_TRUST" >/dev/null
fi
aws iam put-role-policy --role-name "$TASK_ROLE_NAME" --policy-name "ragReindex-inline" \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Sid\":\"RwFaissArtifact\",\"Effect\":\"Allow\",
       \"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],
       \"Resource\":[\"arn:aws:s3:::$S3_BUCKET\",\"arn:aws:s3:::$S3_BUCKET/$RAG_INDEX_PREFIX/*\"]},
      {\"Sid\":\"ReadCreds\",\"Effect\":\"Allow\",
       \"Action\":[\"secretsmanager:GetSecretValue\"],
       \"Resource\":[\"arn:aws:secretsmanager:$REGION:$ACCOUNT_ID:secret:$SECRETS_MANAGER_NAME*\"]},
      {\"Sid\":\"RefreshService\",\"Effect\":\"Allow\",
       \"Action\":[\"ecs:UpdateService\",\"ecs:DescribeServices\"],
       \"Resource\":[\"arn:aws:ecs:$REGION:$ACCOUNT_ID:service/$CLUSTER_NAME/$SERVICE_NAME\"]}
    ]
  }"
echo "  task role $TASK_ROLE_NAME ready"

# --- 3. Register task definition --------------------------------------------
TASK_DEF_FILE="$(mktemp)"
sed -e "s|ACCOUNT_ID|$ACCOUNT_ID|g" -e "s|ECR_IMAGE_URI|$ECR_IMAGE|g" \
  infra/rag_service/reindex_task_def.json > "$TASK_DEF_FILE"
echo "Registering task definition $TASK_FAMILY..."
aws ecs register-task-definition --cli-input-json "file://$TASK_DEF_FILE" --region "$REGION" \
  --query 'taskDefinition.revision' --output text
rm -f "$TASK_DEF_FILE"
TASK_DEF_ARN="arn:aws:ecs:$REGION:$ACCOUNT_ID:task-definition/$TASK_FAMILY"

# --- 4. Scheduler execution role (ecs:RunTask + iam:PassRole) ----------------
SCHED_TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
if ! aws iam get-role --role-name "$SCHED_ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating scheduler role $SCHED_ROLE_NAME..."
  aws iam create-role --role-name "$SCHED_ROLE_NAME" --assume-role-policy-document "$SCHED_TRUST" >/dev/null
fi
aws iam put-role-policy --role-name "$SCHED_ROLE_NAME" --policy-name "ragReindexScheduler-inline" \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Sid\":\"RunTask\",\"Effect\":\"Allow\",\"Action\":[\"ecs:RunTask\"],
       \"Resource\":[\"$TASK_DEF_ARN:*\",\"$TASK_DEF_ARN\"]},
      {\"Sid\":\"PassRoles\",\"Effect\":\"Allow\",\"Action\":[\"iam:PassRole\"],
       \"Resource\":[\"arn:aws:iam::$ACCOUNT_ID:role/$EXEC_ROLE_NAME\",\"arn:aws:iam::$ACCOUNT_ID:role/$TASK_ROLE_NAME\"]}
    ]
  }"
echo "  scheduler role $SCHED_ROLE_NAME ready"
SCHED_ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$SCHED_ROLE_NAME"
CLUSTER_ARN="arn:aws:ecs:$REGION:$ACCOUNT_ID:cluster/$CLUSTER_NAME"

# --- 5. EventBridge Scheduler schedule (daily, Pacific, DST-aware) ----------
TARGET="{
  \"Arn\": \"$CLUSTER_ARN\",
  \"RoleArn\": \"$SCHED_ROLE_ARN\",
  \"EcsParameters\": {
    \"TaskDefinitionArn\": \"$TASK_DEF_ARN\",
    \"TaskCount\": 1,
    \"LaunchType\": \"FARGATE\",
    \"NetworkConfiguration\": {
      \"awsvpcConfiguration\": {
        \"Subnets\": [$SUBNETS_JSON],
        \"SecurityGroups\": [\"$SECURITY_GROUP\"],
        \"AssignPublicIp\": \"ENABLED\"
      }
    }
  }
}"
CRON="cron(0 $SCHEDULE_HOUR * * ? *)"

if aws scheduler get-schedule --name "$SCHEDULE_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "Updating schedule $SCHEDULE_NAME..."
  ACTION=update-schedule
else
  echo "Creating schedule $SCHEDULE_NAME..."
  ACTION=create-schedule
fi
aws scheduler "$ACTION" \
  --name "$SCHEDULE_NAME" \
  --schedule-expression "$CRON" \
  --schedule-expression-timezone "$SCHEDULE_TZ" \
  --flexible-time-window '{"Mode":"OFF"}' \
  --target "$TARGET" \
  --region "$REGION" >/dev/null

echo ""
echo "Done. '$SCHEDULE_NAME' runs $TASK_FAMILY daily at ${SCHEDULE_HOUR}:00 $SCHEDULE_TZ."
echo "Manual test run:"
echo "  aws ecs run-task --cluster $CLUSTER_NAME --launch-type FARGATE \\"
echo "    --task-definition $TASK_FAMILY --region $REGION \\"
echo "    --network-configuration 'awsvpcConfiguration={subnets=[$SUBNETS_JSON],securityGroups=[$SECURITY_GROUP],assignPublicIp=ENABLED}'"
echo "Logs: aws logs tail $LOG_GROUP --since 15m --region $REGION --follow"
