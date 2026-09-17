# MacroPulse — AWS Marketplace & 1-Click Deployment Guide

Deploy **MacroPulse** on AWS with dedicated enterprise infrastructure in under 5 minutes using the automated AWS CloudFormation template.

[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=macropulse-quant&templateURL=https://s3.amazonaws.com/macropulse-public/aws/cloudformation.yaml)

---

## 🏗️ Architecture on AWS

* **Compute**: AWS ECS Fargate (serverless container execution, 1 vCPU, 2 GiB RAM, auto-recovering).
* **Networking**: Dedicated VPC across 2 Availability Zones, Public Subnets, Internet Gateway, and Route Tables.
* **Traffic Routing**: AWS Application Load Balancer (ALB) terminating HTTP/HTTPS and performing automatic health checks.
* **Security & Secrets**: AWS Secrets Manager storing optional FRED API keys, Gemini AI keys, and read-only Brokerage secrets.
* **Observability**: Amazon CloudWatch Logs for real-time quantitative engine diagnostics and latency monitoring.
* **Voice & MCP Interface**: Public endpoint exposed on port 8000 for Alexa Skills Kit webhooks and Model Context Protocol (Streamable HTTP & SSE).

---

## 🚀 1-Click Launch via AWS CLI

```bash
# 1. Clone the repository
git clone https://github.com/sechan9999/hf-macro-dashboard.git
cd hf-macro-dashboard/aws

# 2. Run the deployment script
chmod +x deploy_cfn.sh
./deploy_cfn.sh macropulse-quant us-east-1
```

Or deploy directly with `aws cloudformation deploy`:

```bash
aws cloudformation deploy \
  --template-file cloudformation.yaml \
  --stack-name macropulse-quant \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      EnvironmentName="macropulse-quant" \
      FredApiKey="YOUR_FRED_KEY" \
      GeminiApiKey="YOUR_GEMINI_KEY" \
  --region us-east-1
```

---

## ⚙️ Stack Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `EnvironmentName` | String | `macropulse-quant` | Identifier prefix for all generated AWS resources |
| `ContainerImage` | String | `public.ecr.aws/macropulse/macro-pulse:latest` | URI of container image |
| `TaskCpu` | String | `1024` | Fargate vCPU allocation (512, 1024, 2048, 4096) |
| `TaskMemory` | String | `2048` | Fargate Memory allocation in MiB |
| `FredApiKey` | Secret | `""` (Optional) | St. Louis Fed FRED API key for corporate credit spreads |
| `GeminiApiKey` | Secret | `""` (Optional) | Google Gemini API key for Senior Analyst commentary |
| `ReadOnlyBrokerageSecret` | Secret | `""` (Optional) | Alpaca / IBKR read-only API credentials |

---

## 🩺 Health Check & Post-Deployment Verification

Once CloudFormation completes (`CREATE_COMPLETE`), query the stack output URL:

```bash
# Verify ALB Health Check
curl http://<ALB-DNS-NAME>/healthz

# Verify Alexa MCP Server
curl http://<ALB-DNS-NAME>:8000/health
```
