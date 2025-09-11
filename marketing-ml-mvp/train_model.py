#!/usr/bin/env python3
"""
Training script for the Marketing Campaign Response Prediction model.
This script trains the model and saves it to the models/ directory.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.preprocessor import MarketingDataPreprocessor
from src.models.train import XGBoostTrainer
from src.models.calibration import TemperatureScaling
from src.utils.model_utils import save_model_artifacts
from src.config.settings import MODELS_PATH, XGBOOST_PARAMS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline"""
    
    # Create models directory
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting model training pipeline...")
    
    # Load and preprocess data
    logger.info("📊 Loading and preprocessing data...")
    preprocessor = MarketingDataPreprocessor()
    
    # Load training data (80% of original dataset)
    data_path = Path("data/train_data.csv")
    
    if not data_path.exists():
        logger.error(f"❌ Train data file not found: {data_path}")
        logger.info("Please run split_data.py first to generate train_data.csv from the original dataset")
        logger.info("Expected columns: Education, Marital_Status, Income, Kidhome, etc.")
        return
    
    # Load the data
    X_processed, y, feature_names = preprocessor.load_and_preprocess(data_path)
    
    # Initialize trainer
    logger.info("🎯 Initializing XGBoost trainer...")
    trainer = XGBoostTrainer()
    
    # Train the model with 3-stage tuning
    logger.info("🔧 Training model with 3-stage hyperparameter tuning...")
    model, training_metadata = trainer.train(
        X_processed, y, 
        feature_names=feature_names,
        save_path=MODELS_PATH
    )
    
    # Train calibrator
    logger.info("📊 Training probability calibrator...")
    calibrator = TemperatureScaling()
    
    # Get validation predictions for calibration
    X_val = training_metadata['validation_data']['X_val']
    y_val = training_metadata['validation_data']['y_val']
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Fit calibrator
    calibrator.fit(val_probs, y_val)
    
    # Save calibrator
    calibrator_path = MODELS_PATH / "temperature_scaling_calibrator.pkl"
    calibrator.save(calibrator_path)
    
    # Create complete model metadata
    complete_metadata = {
        **training_metadata,
        'has_calibrator': True,
        'calibrator_path': str(calibrator_path),
        'model_version': '1.0.0',
        'training_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save complete metadata
    metadata_path = MODELS_PATH / "model_metadata.json"
    save_model_artifacts(
        model=None,  # Model already saved by trainer
        metadata=complete_metadata,
        model_path=MODELS_PATH / "xgboost_model.pkl",  # Already exists
        metadata_path=metadata_path
    )
    
    logger.info(" Training completed successfully!")
    logger.info(f"📁 Model saved to: {MODELS_PATH}")
    logger.info(f"📊 Model accuracy: {complete_metadata.get('final_metrics', {}).get('accuracy', 'N/A'):.3f}")
    logger.info(f"🎯 Optimal threshold: {complete_metadata.get('optimal_threshold', 'N/A'):.3f}")
    
    return model, complete_metadata

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f" Training failed: {str(e)}")
        sys.exit(1)