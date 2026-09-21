<#
.SYNOPSIS
  One-time IAM setup for hosting the MacroPulse Alexa+ MCP server on Amazon ECS Express Mode
  and for using the Alexa+ CLI.

  Run it YOURSELF, in your own terminal, with an AWS profile that is allowed to create IAM
  users and roles (for example the root login profile).

  Creates (and touches nothing else):
    roles  ecsTaskExecutionRole
           ecsInfrastructureRoleForExpressServices        (both required by ECS Express Mode)
    user   macropulse-deployer   deploys the service; signs in with `aws login` (no access keys)
    user   alexa-ai-tools        Alexa+ CLI; may ONLY assume Amazon's AddOn3PDeveloperToolsRead role

  It never creates passwords or access keys. You do that in the IAM console (steps are printed
  at the end), so no secret ever passes through a script or a chat.

  Safe to re-run: existing roles/users are kept and their policies are re-applied.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File aws\setup-express-iam.ps1 -Profile tcgyver
#>
param([string]$Profile = "tcgyver")
$ErrorActionPreference = "Stop"

# This window's PATH can be stale (for example opened before the AWS CLI was installed).
# Re-read PATH from the registry for THIS process only; nothing is changed permanently.
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + $env:Path
$awsCmd = Get-Command aws -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $awsCmd) { throw "AWS CLI ('aws') not found. Install AWS CLI v2, or open a NEW PowerShell window and re-run." }
Write-Host "AWS CLI: $($awsCmd.Source)" -ForegroundColor DarkGray

function Invoke-Aws {
    $ErrorActionPreference = "Continue"   # local to this function: stderr must not abort before we read the exit code
    $out = & aws @args --profile $Profile 2>&1
    if ($LASTEXITCODE -ne 0) { throw "aws $($args -join ' ') failed:`n$out" }
    $out
}
function Test-Aws {
    $ErrorActionPreference = "Continue"
    & aws @args --profile $Profile *> $null
    return ($LASTEXITCODE -eq 0)
}
function Write-Json($name, $json) {
    $path = Join-Path $env:TEMP $name
    [IO.File]::WriteAllText($path, $json, (New-Object Text.UTF8Encoding $false))
    return "file://" + ($path -replace '\\', '/')
}

$account = (Invoke-Aws sts get-caller-identity --query Account --output text).Trim()
$masked = ("*" * ($account.Length - 4)) + $account.Substring($account.Length - 4)
Write-Host "Using profile '$Profile' (account $masked)" -ForegroundColor Cyan

# ---------------------------------------------------------------- ECS Express Mode roles
function Ensure-Role($name, $trustJson, $managedPolicyArn) {
    if (Test-Aws iam get-role --role-name $name) {
        Write-Host "  role $name already exists - keeping it"
    } else {
        Invoke-Aws iam create-role --role-name $name --assume-role-policy-document (Write-Json "$name-trust.json" $trustJson) | Out-Null
        Write-Host "  created role $name"
    }
    Invoke-Aws iam attach-role-policy --role-name $name --policy-arn $managedPolicyArn | Out-Null
}

Write-Host "`n[1/3] ECS Express Mode roles" -ForegroundColor Cyan
$trustTasks = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
$trustInfra = '{"Version":"2012-10-17","Statement":[{"Sid":"AllowAccessInfrastructureForECSExpressServices","Effect":"Allow","Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
Ensure-Role "ecsTaskExecutionRole" $trustTasks "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
Ensure-Role "ecsInfrastructureRoleForExpressServices" $trustInfra "arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"

# ---------------------------------------------------------------- deploy user
function Ensure-User($name) {
    if (Test-Aws iam get-user --user-name $name) { Write-Host "  user $name already exists - keeping it" }
    else { Invoke-Aws iam create-user --user-name $name | Out-Null; Write-Host "  created user $name" }
}

Write-Host "`n[2/3] user macropulse-deployer (deploys the service)" -ForegroundColor Cyan
Ensure-User "macropulse-deployer"
foreach ($arn in @(
        "arn:aws:iam::aws:policy/AmazonECS_FullAccess",
        "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
        "arn:aws:iam::aws:policy/CloudWatchLogsReadOnlyAccess",
        "arn:aws:iam::aws:policy/SignInLocalDevelopmentAccess")) {
    Invoke-Aws iam attach-user-policy --user-name macropulse-deployer --policy-arn $arn | Out-Null
}
$deployerInline = @"
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":["iam:GetRole","iam:PassRole"],
   "Resource":["arn:aws:iam::${account}:role/ecsTaskExecutionRole","arn:aws:iam::${account}:role/ecsInfrastructureRoleForExpressServices"]},
  {"Effect":"Allow","Action":"iam:CreateServiceLinkedRole","Resource":"*",
   "Condition":{"StringEquals":{"iam:AWSServiceName":["ecs.amazonaws.com","elasticloadbalancing.amazonaws.com","ecs.application-autoscaling.amazonaws.com"]}}}
]}
"@
Invoke-Aws iam put-user-policy --user-name macropulse-deployer --policy-name macropulse-deployer-passrole --policy-document (Write-Json "deployer-inline.json" $deployerInline) | Out-Null
Write-Host "  attached: ECS, ECR, CloudWatch Logs (read), SignInLocalDevelopmentAccess + PassRole for the two roles only"

# ---------------------------------------------------------------- Alexa+ CLI user
Write-Host "`n[3/3] user alexa-ai-tools (Alexa+ CLI, AssumeRole only)" -ForegroundColor Cyan
Ensure-User "alexa-ai-tools"
$alexaInline = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Resource":"arn:aws:iam::372468808636:role/AddOn3PDeveloperToolsRead"}]}'
Invoke-Aws iam put-user-policy --user-name alexa-ai-tools --policy-name assume-alexa-addon-tools --policy-document (Write-Json "alexa-inline.json" $alexaInline) | Out-Null
Write-Host "  inline policy set: may only call sts:AssumeRole on Amazon's AddOn3PDeveloperToolsRead role"

Write-Host @"

Done. Two manual steps remain (they involve secrets, so you do them in the console):

  A) IAM console -> Users -> macropulse-deployer -> Security credentials -> Enable console access
     -> set a password of your choice. (No access key needed; 'aws login' gives short-term credentials.)
  B) IAM console -> Users -> alexa-ai-tools -> Security credentials -> Create access key
     -> "Command Line Interface (CLI)". Then, in your terminal:  aws configure --profile alexa-ai-user

Then tell Claude, and sign in as the deployer:
  aws login --region us-east-1 --profile deployer
(the browser sign-in asks for the account ID or alias, the IAM user name macropulse-deployer, and its password)

Remove everything later (after judging) by deleting the two users and two roles in the IAM console.
"@ -ForegroundColor Green
