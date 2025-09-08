#!/bin/bash

# Startup script for training VM on GCP Compute Engine
# Professional MLOps training pipeline execution

set -e

# Configuration
PROJECT_ID=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)
BUCKET_NAME="blend-mlops-models-${PROJECT_ID}"
ZONE=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d'/' -f4)
INSTANCE_NAME=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)

echo "🚀 Starting MLOps training pipeline on ${INSTANCE_NAME}"
echo "📊 Project: ${PROJECT_ID}, Zone: ${ZONE}"

# Update system and install dependencies
apt-get update
apt-get install -y python3 python3-pip git curl

# Install Python packages
pip3 install pandas scikit-learn xgboost numpy matplotlib seaborn shap scipy optuna google-cloud-storage

# Create working directory
mkdir -p /opt/mlops-training
cd /opt/mlops-training

# Download training scripts from Cloud Storage
echo "📥 Downloading training scripts from GCS..."
gsutil -m cp -r gs://${BUCKET_NAME}/scripts/* .

# Download data from Cloud Storage (assuming data is uploaded)
echo "📥 Downloading training data..."
gsutil cp gs://${BUCKET_NAME}/data/marketing_campaign.csv ./data/

# Create artifacts directory
mkdir -p artifacts

# Execute training pipeline
echo "🏋️ Starting model training..."
python3 pipeline/run_pipeline.py \
    --data-path data/marketing_campaign.csv \
    --artifacts-dir artifacts \
    --test-size 0.2 \
    --val-split 0.2

# Upload trained models and artifacts to Cloud Storage
echo "📤 Uploading models to Cloud Storage..."

# Create versioned directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_VERSION="v_${TIMESTAMP}"

# Upload models with versioning
gsutil -m cp -r artifacts/ gs://${BUCKET_NAME}/models/${MODEL_VERSION}/
gsutil -m cp -r artifacts/ gs://${BUCKET_NAME}/models/latest/

# Create model metadata
cat > model_metadata.json << EOF
{
    "version": "${MODEL_VERSION}",
    "training_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "instance_name": "${INSTANCE_NAME}",
    "zone": "${ZONE}",
    "project_id": "${PROJECT_ID}",
    "data_size": "$(wc -l < data/marketing_campaign.csv) rows",
    "pipeline_status": "completed",
    "artifacts_location": "gs://${BUCKET_NAME}/models/${MODEL_VERSION}/"
}
EOF

# Upload metadata
gsutil cp model_metadata.json gs://${BUCKET_NAME}/models/latest/
gsutil cp model_metadata.json gs://${BUCKET_NAME}/models/${MODEL_VERSION}/

# Send completion notification (optional webhook)
echo "✅ Training completed successfully"
echo "📦 Model version: ${MODEL_VERSION}"
echo "🏆 Artifacts uploaded to: gs://${BUCKET_NAME}/models/${MODEL_VERSION}/"

# Clean shutdown - delete instance after successful completion
echo "🔄 Training completed. Shutting down instance..."
gcloud compute instances delete ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --project=${PROJECT_ID} \
    --quiet

exit 0