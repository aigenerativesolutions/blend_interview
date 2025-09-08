"""
Configuration settings for the Marketing ML MVP
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"

# Data file
MARKETING_DATA_FILE = "marketing_campaign.csv"

# Model settings
MODEL_NAME = "xgboost_marketing_model.pkl"
CALIBRATOR_NAME = "temperature_scaler.pkl"
MODEL_METADATA_NAME = "model_metadata.json"

# Feature engineering settings
CURRENT_YEAR = 2014  # Based on notebook analysis
SPENDING_COLUMNS = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]

# Columns to drop (from notebook analysis)
COLUMNS_TO_DROP = [
    'ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'
]

# Model hyperparameters (from notebook tuning results)
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# Cross-validation settings
CV_FOLDS = 5
STRATIFIED_CV = True

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Marketing Response Prediction API"
API_VERSION = "1.0.0"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")