# Marketing Campaign MLOps Pipeline

## Overview

Production-ready MLOps pipeline for marketing campaign response prediction using XGBoost with Bayesian optimization via Optuna. Implements complete machine learning workflow with automated feature engineering, hyperparameter tuning, model training, probability calibration, and SHAP analysis.

## Architecture

### Pipeline Components

1. **Feature Engineering**: Automated data preprocessing and feature creation
2. **Hyperparameter Optimization**: Bayesian optimization using Optuna with 100 trials
3. **Model Training**: XGBoost classifier with optimized parameters
4. **Probability Calibration**: Temperature scaling for improved probability estimates
5. **Model Interpretation**: SHAP analysis with feature importance and interactions
6. **Containerized Deployment**: Docker-ready for cloud deployment

### Directory Structure

```
marketing-ml-mvp/
├── data/                       # Dataset storage
├── pipeline/                   # Core ML pipeline components
│   ├── run_pipeline.py        # Main orchestrator
│   ├── third_tuning.py        # Optuna optimization
│   ├── train_final_fixed.py   # Final model training
│   ├── temperature_calibration.py  # Probability calibration
│   └── shap_analysis_pipeline_fixed.py  # Model interpretation
├── src/                       # Source code modules
│   └── data/                  # Data processing utilities
├── artifacts/                 # Generated models and results
├── requirements.txt           # Python dependencies
└── Dockerfile.pipeline        # Container configuration
```

## Technical Specifications

### Model Performance
- **Algorithm**: XGBoost Classifier with categorical feature support
- **Optimization**: Bayesian hyperparameter tuning (100 trials)
- **Cross-validation**: 5-fold stratified CV
- **Metrics**: F1-score optimization with ROC-AUC validation

### Feature Engineering
- Age calculation from birth year (reference: 2014)
- Customer tenure calculation with day adjustment
- Total children aggregation (Kidhome + Teenhome)
- Total spending across all product categories
- Education level encoding (ordinal: 0-4)
- Marital status as categorical feature

### Key Features
- **Reproducible**: Fixed random seeds and versioned artifacts
- **Scalable**: Containerized deployment ready for cloud platforms
- **Interpretable**: Comprehensive SHAP analysis with feature interactions
- **Calibrated**: Temperature-scaled probability estimates
- **Automated**: Complete end-to-end pipeline execution

## Usage

### Local Execution

```bash
# Build container
docker build -f Dockerfile.pipeline -t marketing-ml-pipeline .

# Run complete pipeline
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  marketing-ml-pipeline \
  python pipeline/run_pipeline.py \
    --data-path data/marketing_campaign.csv \
    --artifacts-dir artifacts
```

### Pipeline Configuration

- **Test Size**: 20% holdout for final evaluation
- **Validation Split**: 20% of training data for hyperparameter tuning
- **Optuna Trials**: 100 trials with TPE sampler
- **Early Stopping**: 50 rounds for final model training
- **SHAP Analysis**: Top 20 positive/negative impacts per feature, top 7 feature interactions

## Output Artifacts

### Model Files
- `final_model.pkl`: Trained XGBoost classifier
- `feature_pipeline.pkl`: Data preprocessing pipeline
- `best_model_optuna.pkl`: Best model from optimization

### Configuration Files
- `best_params_optuna.json`: Optimal hyperparameters
- `optimal_threshold_optuna.json`: Calibrated decision threshold
- `final_model_metadata.json`: Complete model metadata

### Analysis Results
- `optuna_results.json`: Hyperparameter tuning summary
- `optuna_trials.csv`: Detailed trial history
- `top_shap_values_by_feature.json`: Feature impact analysis
- `feature_interactions_top7.json`: Feature interaction analysis

## Production Deployment

### Requirements
- Docker or Kubernetes environment
- Minimum 2GB RAM for full dataset processing
- Python 3.9+ with scientific computing libraries

### Performance Characteristics
- Training time: ~30-45 minutes (full dataset, 100 trials)
- Memory usage: ~1.5GB peak during training
- Model size: ~50MB serialized
- Inference: <10ms per prediction

## Model Validation

### Cross-Validation Strategy
- 5-fold stratified cross-validation during hyperparameter tuning
- Class imbalance handling via calculated scale_pos_weight
- F1-score optimization for imbalanced dataset performance

### Quality Assurance
- Automated hyperparameter bounds validation
- Early stopping to prevent overfitting
- Temperature calibration for probability reliability
- Comprehensive SHAP analysis for model transparency

## Data Schema

### Input Features (27 total)
- Customer demographics (Age, Education, Marital_Status)
- Purchase behavior (spending across 6 product categories)
- Engagement metrics (web visits, store purchases, catalog purchases)
- Campaign responses (historical campaign acceptance)
- Derived features (Total_Spent, Total_Kids, Months_As_Customer)

### Target Variable
- Binary response (0: No response, 1: Response to marketing campaign)
- Class distribution: ~15% positive class (imbalanced)

## Development

### Adding New Features
1. Modify `src/data/preprocessor_unified.py`
2. Update feature names list
3. Adjust XGBoost categorical feature handling if needed
4. Re-run hyperparameter optimization

### Extending Analysis
1. Add new SHAP plot types in `shap_analysis_pipeline_fixed.py`
2. Implement additional calibration methods in `temperature_calibration.py`
3. Include new optimization metrics in `third_tuning.py`

## Dependencies

Core libraries:
- `xgboost>=1.7.0`: Gradient boosting framework
- `optuna>=3.1.0`: Hyperparameter optimization
- `shap>=0.41.0`: Model interpretation
- `scikit-learn>=1.0.0`: Machine learning utilities
- `pandas>=1.3.0`: Data manipulation
- `numpy>=1.21.0`: Numerical computing