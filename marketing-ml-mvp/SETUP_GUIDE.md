# 🚀 GitHub → GCP → Local Pipeline Setup Guide

Complete setup guide for the hybrid MLOps pipeline that trains models in GCP and uses them locally.

## 📋 **Overview**

This pipeline provides:
- 🤖 **Automated training** triggered by GitHub pushes
- ☁️ **Cloud training** on GCP Compute Engine VMs  
- 📦 **Model storage** in Cloud Storage with versioning
- 🔄 **Auto-sync** to your local Streamlit app

---

## 🛠️ **1. GCP Setup**

### **Enable APIs**
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
```

### **Create Service Account**
```bash
# Create service account
gcloud iam service-accounts create mlops-pipeline \
    --description="MLOps pipeline service account" \
    --display-name="MLOps Pipeline"

# Add necessary roles
PROJECT_ID="your-project-id"
SA_EMAIL="mlops-pipeline@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/compute.instanceAdmin.v1"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/cloudfunctions.invoker"

# Download service account key
gcloud iam service-accounts keys create mlops-sa-key.json \
    --iam-account=$SA_EMAIL
```

### **Create Cloud Storage Bucket**
```bash
PROJECT_ID="your-project-id"
REGION="us-central1"

# Create bucket
gsutil mb -p $PROJECT_ID -l $REGION gs://blend-mlops-models-$PROJECT_ID

# Upload training data
gsutil cp data/marketing_campaign.csv gs://blend-mlops-models-$PROJECT_ID/data/

# Upload training scripts
gsutil cp -r marketing-ml-mvp/pipeline/ gs://blend-mlops-models-$PROJECT_ID/scripts/
gsutil cp -r marketing-ml-mvp/src/ gs://blend-mlops-models-$PROJECT_ID/scripts/
gsutil cp marketing-ml-mvp/deployment/training-vm-startup.sh gs://blend-mlops-models-$PROJECT_ID/scripts/
```

### **Deploy Cloud Function**
```bash
cd marketing-ml-mvp

# Deploy training trigger function
gcloud functions deploy training-trigger \
    --gen2 \
    --runtime python39 \
    --region $REGION \
    --source deployment/training-trigger \
    --entry-point create_training_instance \
    --trigger-http \
    --allow-unauthenticated \
    --memory 512MB \
    --set-env-vars GCP_PROJECT=$PROJECT_ID,COMPUTE_ZONE=us-central1-a

# Get the function URL
FUNCTION_URL=$(gcloud functions describe training-trigger --region=$REGION --format='value(url)')
echo "Training Function URL: $FUNCTION_URL"
```

---

## 🔐 **2. GitHub Setup**

### **Add Repository Secrets**
Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `GCP_PROJECT_ID` | your-gcp-project-id | Your GCP project ID |
| `GCP_SA_KEY` | [JSON content] | Service account key JSON (entire file content) |
| `TRAINING_FUNCTION_URL` | https://... | Cloud Function URL from above |

### **Service Account Key Format**
For `GCP_SA_KEY`, copy the **entire contents** of `mlops-sa-key.json`:
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "mlops-pipeline@your-project-id.iam.gserviceaccount.com",
  ...
}
```

---

## 🖥️ **3. Local Setup**

### **Install Dependencies**
```bash
cd marketing-ml-mvp

# Install Streamlit dependencies
pip install -r requirements_streamlit.txt

# Or install additional packages
pip install google-cloud-storage
```

### **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
# Required:
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/mlops-sa-key.json

# Optional (for LLM assistant):
OPENAI_API_KEY=your-openai-key
# or
GROQ_API_KEY=your-groq-key
```

### **Test Model Sync**
```bash
# Test model sync manually
python model_sync.py --status

# Check for updates
python model_sync.py --check

# Force sync
python model_sync.py --sync --force
```

---

## 🚀 **4. Pipeline Usage**

### **Automatic Training**
1. **Make changes** to your ML pipeline code
2. **Push to main branch**
3. **GitHub Action triggers** automatically
4. **GCP trains model** (15-30 minutes)
5. **Model uploaded** to Cloud Storage
6. **Local app auto-syncs** new model

### **Manual Training**
```bash
# Trigger training via GitHub Actions
gh workflow run train-model.yml

# Or call Cloud Function directly
curl -X POST $FUNCTION_URL \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -d '{"trigger_source": "manual"}'
```

### **Run Streamlit App**
```bash
# Start the app
streamlit run app_streamlit.py

# App will:
# 1. Auto-check for model updates on startup
# 2. Download latest model if available
# 3. Show sync status in the UI
# 4. Provide manual sync buttons
```

---

## 🔧 **5. Troubleshooting**

### **Common Issues**

**GitHub Action Fails:**
```bash
# Check GitHub Actions logs
# Common fixes:
# 1. Verify GCP_SA_KEY secret is valid JSON
# 2. Check GCP_PROJECT_ID matches your project
# 3. Ensure Cloud Function URL is correct
```

**Model Sync Fails:**
```bash
# Check authentication
python -c "from google.cloud import storage; print(storage.Client().list_buckets())"

# Check bucket exists
gsutil ls gs://blend-mlops-models-your-project-id/

# Check permissions
gsutil iam get gs://blend-mlops-models-your-project-id/
```

**Cloud Function Timeout:**
```bash
# Check VM creation in GCP Console
# Compute Engine → VM instances
# Look for instances with "mlops-training" prefix
```

### **Debug Commands**
```bash
# Check model sync status
python model_sync.py --status

# View Streamlit logs
streamlit run app_streamlit.py --logger.level=debug

# Monitor GCP training
gcloud compute instances list --filter="name~mlops-training"
```

### **Performance Optimization**
```bash
# Use preemptible VMs (already configured)
# Auto-shutdown after training (already configured)
# Model caching (already configured - 1 hour)
```

---

## 📊 **6. Architecture Diagram**

```
GitHub Repository
│
├── Push to main branch
│   ├── .github/workflows/train-model.yml (GitHub Action)
│   ├── Authenticates with GCP
│   └── Calls Cloud Function
│
└── Cloud Function (training-trigger)
    ├── Creates Compute Engine VM
    ├── VM runs training pipeline
    ├── Uploads model to Cloud Storage
    └── Auto-destroys VM
│
Cloud Storage (gs://blend-mlops-models-project/)
├── models/latest/ (current model)
├── models/v_timestamp/ (versioned models)
├── scripts/ (training code)
└── data/ (training data)
│
Local Streamlit App
├── Auto-syncs on startup
├── Downloads latest model
├── Provides manual sync controls
└── Uses model for predictions
```

---

## ✅ **7. Success Validation**

### **Pipeline Working Signs:**
1. ✅ GitHub Action completes successfully
2. ✅ Cloud Function creates VM
3. ✅ VM trains model and uploads to GCS
4. ✅ Streamlit app syncs new model
5. ✅ Predictions use latest model

### **Test the Full Pipeline:**
```bash
# 1. Make a small change to the model
echo "# Test change" >> marketing-ml-mvp/pipeline/train_final_fixed.py

# 2. Commit and push
git add -A
git commit -m "Test MLOps pipeline"
git push origin main

# 3. Monitor GitHub Actions
# 4. Check GCP Console for VM
# 5. Wait for training completion
# 6. Restart Streamlit app
# 7. Verify new model version
```

---

## 🎯 **Perfect for BLEND Demo!**

This setup demonstrates:
- ✅ **End-to-end MLOps** automation
- ✅ **Hybrid cloud/local** architecture  
- ✅ **Cost optimization** (VMs auto-destroy)
- ✅ **Professional CI/CD** with GitHub Actions
- ✅ **Modern ML stack** (XGBoost + LLM + Streamlit)
- ✅ **Production-ready** patterns

🚀 **Your pipeline is ready to impress!**