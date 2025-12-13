$REGION = "us-east-1"
$REPO_NAME = "agent-runtime-backend"

# Fetch Account ID and trim whitespace/newlines which caused the previous error
$ACCOUNT_ID = (aws sts get-caller-identity --query "Account" --output text).Trim()

$ECR_URL = "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
$IMAGE_URI = "${ECR_URL}/${REPO_NAME}:latest"

Write-Host "Account ID: $ACCOUNT_ID"
Write-Host "Target Image URI: $IMAGE_URI"

Write-Host "Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URL

if ($LASTEXITCODE -ne 0) {
    Write-Error "Login failed!"
    exit 1
}

Write-Host "Building Docker Image..."
docker build -t "$REPO_NAME" .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed!"
    exit 1
}

Write-Host "Tagging Image..."
# We explicitly tag it to match our target URI
docker tag "${REPO_NAME}:latest" "$IMAGE_URI"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Tag failed!"
    exit 1
}

Write-Host "Pushing Image to ECR..."
docker push "$IMAGE_URI"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Push failed!"
    exit 1
}

Write-Host "Done! You can now run 'tofu apply' to deploy the Lambda function."
