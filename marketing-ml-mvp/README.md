# Marketing Campaign Response Prediction API

## 🎯 Overview

This project provides a production-ready machine learning API for predicting customer responses to marketing campaigns. The model uses XGBoost with advanced features including probability calibration and SHAP explanations.

## 🏗️ Architecture

```
marketing-ml-mvp/
├── src/
│   ├── api/                 # FastAPI REST API
│   ├── data/               # Data preprocessing
│   ├── models/             # ML models and training
│   ├── explainability/     # SHAP analysis
│   ├── utils/              # Utilities
│   └── config/             # Configuration
├── models/                 # Trained model artifacts
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── deploy.sh             # GCP deployment script
```

## 🚀 Quick Start

### 1. Local Development

```bash
# Clone and navigate to the project
cd marketing-ml-mvp

# Install dependencies
pip install -r requirements.txt

# Train the model (requires data file)
python train_model.py

# Start the API server
uvicorn src.api.main:app --reload --port 8080
```

### 2. Docker Deployment

```bash
# Build the Docker image
docker build -t marketing-ml-api .

# Run the container
docker run -p 8080:8080 marketing-ml-api
```

### 3. GCP Cloud Run Deployment

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"

# Deploy using the script
./deploy.sh
```

## 📊 Model Features

### Core ML Pipeline
- **XGBoost Classifier** with 3-stage hyperparameter tuning
- **Temperature Scaling** for probability calibration
- **SHAP Analysis** for model explainability
- **Threshold Optimization** using precision-recall curves

### Key Features
- Customer demographics (age, income, education, marital status)
- Purchase behavior (spending patterns, channel preferences)
- Campaign history (previous campaign responses)
- Derived features (total spend, customer tenure)

## 🔗 API Endpoints

### Base URL
- **Local**: `http://localhost:8080`
- **Production**: Your Cloud Run service URL

### Main Endpoints

#### Health Check
```http
GET /health
```

#### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "customer": {
    "Education": "Graduation",
    "Marital_Status": "Married",
    "Income": 58138.0,
    "Kidhome": 0,
    "Teenhome": 0,
    "MntWines": 635,
    "MntFruits": 88,
    "MntMeatProducts": 546,
    "MntFishProducts": 172,
    "MntSweetProducts": 88,
    "MntGoldProds": 88,
    "NumDealsPurchases": 3,
    "NumWebPurchases": 8,
    "NumCatalogPurchases": 10,
    "NumStorePurchases": 4,
    "NumWebVisitsMonth": 7,
    "AcceptedCmp1": 0,
    "AcceptedCmp2": 0,
    "AcceptedCmp3": 0,
    "AcceptedCmp4": 0,
    "AcceptedCmp5": 0,
    "Recency": 58,
    "Complain": 0
  },
  "use_calibration": true,
  "threshold": null
}
```

#### Batch Predictions
```http
POST /predict/batch
```

#### SHAP Explanations
```http
POST /explain
```

#### Model Information
```http
GET /model/info
GET /model/feature-importance
GET /model/sample-input
```

### Interactive Documentation
- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

## 🛠️ Technical Details

### Model Performance
- **Algorithm**: XGBoost with optimized hyperparameters
- **Validation**: 5-fold stratified cross-validation
- **Calibration**: Temperature scaling for reliable probabilities
- **Explainability**: SHAP values for prediction interpretation

### Production Features
- **Health checks** and monitoring endpoints
- **Request validation** with Pydantic schemas
- **Error handling** with detailed error responses
- **CORS support** for web applications
- **Logging** and metrics for observability

### Deployment Options
1. **Cloud Run**: Serverless, auto-scaling (recommended)
2. **App Engine**: Managed platform with automatic scaling
3. **Compute Engine**: Full control over infrastructure
4. **Local Docker**: Development and testing

## 📋 Requirements

### Data Requirements
Your training data CSV should include these columns:
- **Demographics**: Education, Marital_Status, Income, Kidhome, Teenhome
- **Spending**: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
- **Purchases**: NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth
- **Campaign History**: AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5
- **Other**: Recency, Complain
- **Target**: Response (0/1)

### System Requirements
- **Python 3.10+**
- **Memory**: 2GB+ recommended
- **CPU**: 2+ cores for optimal performance

## 🔧 Configuration

### Environment Variables
```bash
API_HOST=0.0.0.0          # API host
API_PORT=8080             # API port
API_TITLE=Marketing ML API # API title
API_VERSION=1.0.0         # API version
```

### Model Parameters
Configure in `src/config/settings.py`:
- XGBoost hyperparameters
- Cross-validation settings
- Threshold optimization parameters

## 🧪 Testing

```bash
# Test the health endpoint
curl http://localhost:8080/health

# Test with sample data
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## 📈 Monitoring and Logging

### Health Checks
- **Liveness**: `/health` endpoint
- **Readiness**: Model loading status
- **Performance**: Response time monitoring

### Logging
- Request/response logging
- Model prediction metrics
- Error tracking and alerting

## 🔒 Security

- Input validation with Pydantic
- Rate limiting (configurable)
- CORS policy configuration
- No sensitive data logging

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints consistently

## 📞 Support

For questions about this implementation:
1. Check the API documentation at `/docs`
2. Review the model training pipeline in `train_model.py`
3. Examine the code structure in `src/`

## 🏆 Interview Notes

This project demonstrates:
- **Production ML deployment** patterns
- **Modern API development** with FastAPI
- **Cloud-native architecture** for GCP
- **MLOps best practices** (versioning, monitoring, validation)
- **Model interpretability** with SHAP
- **Probability calibration** for reliable predictions
- **Containerization** and deployment automation