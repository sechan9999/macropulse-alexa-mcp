#!/usr/bin/env bash
# aws/deploy_cfn.sh
# ─────────────────────────────────────────────────────────────────
# One-click AWS CloudFormation deploy script for MacroPulse
# Deploys MacroPulse to AWS ECS Fargate with ALB and Secrets Manager.
#
# Usage:
#   ./deploy_cfn.sh [STACK_NAME] [REGION]
# Example:
#   ./deploy_cfn.sh macropulse-prod us-east-1

set -e

STACK_NAME=${1:-macropulse-quant}
AWS_REGION=${2:-us-east-1}
TEMPLATE_FILE="$(dirname "$0")/cloudformation.yaml"

echo "==================================================================="
echo "🚀 MacroPulse AWS Marketplace 1-Click CloudFormation Deployer"
echo "Stack Name: ${STACK_NAME}"
echo "Region:     ${AWS_REGION}"
echo "==================================================================="

# Validate template
echo "Validating CloudFormation template..."
aws cloudformation validate-template \
  --template-body "file://${TEMPLATE_FILE}" \
  --region "${AWS_REGION}"

# Deploy stack
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file "${TEMPLATE_FILE}" \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      EnvironmentName="${STACK_NAME}" \
  --no-fail-on-empty-changeset

echo "Querying Outputs..."
aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${AWS_REGION}" \
  --query "Stacks[0].Outputs" \
  --output table

echo "🎉 Deployment complete!"
