# AWS Lambda Deployment Script for Windows PowerShell
param (
    [string]$AwsRegion = "us-east-1",
    [string]$AwsAccountId = "",
    [string]$EcrRepoName = "flood-risk-api",
    [string]$LambdaFunctionName = "flood-risk-api"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($AwsAccountId)) {
    Write-Host "Fetching AWS Account ID..." -ForegroundColor Cyan
    $AwsAccountId = (aws sts get-caller-identity --query Account --output text)
}

$EcrUri = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com/$EcrRepoName"

Write-Host "1. Building Docker image locally..." -ForegroundColor Cyan
docker build --provenance=false -t $EcrRepoName .

Write-Host "2. Logging into AWS ECR ($AwsRegion)..." -ForegroundColor Cyan
aws ecr get-login-password --region $AwsRegion | docker login --username AWS --password-stdin "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com"

Write-Host "3. Ensuring ECR Repository '$EcrRepoName' exists..." -ForegroundColor Cyan
try {
    aws ecr describe-repositories --repository-names $EcrRepoName --region $AwsRegion 2>$null
} catch {
    aws ecr create-repository --repository-name $EcrRepoName --region $AwsRegion
}

Write-Host "4. Tagging and Pushing image to ECR ($EcrUri)..." -ForegroundColor Cyan
docker tag ${EcrRepoName}:latest "${EcrUri}:latest"
docker push "${EcrUri}:latest"

Write-Host "5. Deploying to AWS Lambda '$LambdaFunctionName'..." -ForegroundColor Cyan
try {
    aws lambda update-function-code `
        --function-name $LambdaFunctionName `
        --image-uri "${EcrUri}:latest" `
        --region $AwsRegion
    Write-Host "Successfully updated Lambda function!" -ForegroundColor Green
} catch {
    Write-Host "Lambda function '$LambdaFunctionName' does not exist yet. Please create it using AWS CLI or AWS Console." -ForegroundColor Yellow
}
