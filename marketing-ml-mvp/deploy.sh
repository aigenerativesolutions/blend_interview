#!/bin/bash

# Marketing ML API Deployment Script for GCP
# Make sure you have gcloud CLI installed and authenticated

set -e  # Exit on any error

# Configuration
PROJECT_ID=${PROJECT_ID:-"your-project-id"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"marketing-ml-api"}

echo "🚀 Deploying Marketing ML API to GCP..."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi

# Set the project
echo "📋 Setting project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Option 1: Deploy using Cloud Build + Cloud Run
echo "🏗️  Building and deploying with Cloud Build..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME .

echo "☁️  Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 10 \
    --timeout 300 \
    --set-env-vars API_HOST=0.0.0.0,API_PORT=8080

echo "✅ Deployment completed!"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
echo "🌐 Service URL: $SERVICE_URL"
echo "📊 Health check: $SERVICE_URL/health"
echo "📖 API docs: $SERVICE_URL/docs"

# Test the service
echo "🧪 Testing the service..."
curl -f "$SERVICE_URL/health" && echo "✅ Health check passed!" || echo "❌ Health check failed!"

echo "🎉 Deployment script completed!"
echo ""
echo "Next steps:"
echo "1. Test your API: $SERVICE_URL/docs"
echo "2. Train and upload your model to the models/ directory"
echo "3. Monitor logs: gcloud logs tail --follow --log-filter=\"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\""