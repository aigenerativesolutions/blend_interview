"""
Feature Engineering - EXACTO DEL NOTEBOOK
Implementación exacta del código que el usuario proporcionó
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def prepare_data_for_training_exact(data_path: Path, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Preparar datos con EXACTAMENTE el código del notebook proporcionado
    """
    logger.info(f"📊 Cargando datos desde {data_path}")
    
    # Cargar datos con separador correcto
    df = pd.read_csv(data_path, sep=';')
    logger.info(f"Datos cargados: {df.shape}")
    
    # EXACTO DEL NOTEBOOK - Convertir Dt_Customer a datetime
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    reference_date = pd.Timestamp('2014-09-01')
    
    months = (
        (reference_date.year - df['Dt_Customer'].dt.year) * 12 +
        (reference_date.month - df['Dt_Customer'].dt.month)
    )
    
    # -1 if day dont match with reference_date
    adjust = (reference_date.day < df['Dt_Customer'].dt.day).astype(int)
    df['Months_As_Customer'] = (months - adjust).astype(int)
    
    # Calculate Age
    df['Age'] = 2014 - df['Year_Birth']
    
    # Combine Kidhome and Teenhome to reduce columns
    df['Total_Kids'] = df['Kidhome'] + df['Teenhome']
    
    # Calculate Total_Spent to see if total amount tell something 
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spent'] = df[spending_cols].sum(axis=1)
    
    # Drop columns irrelevant, age already calculated up there
    df.drop(columns=['ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer'], inplace=True)
    
    # Map Education to ordered cardinality, cause XGboost, even when do efficients categorical partitions, may dont capture
    education_order = {
        'Basic': 0,
        '2n Cycle': 1,
        'Graduation': 2,
        'Master': 3,
        'PhD': 4
    }
    df['Education'] = df['Education'].map(education_order).astype('int')
    
    # Convert Marital_Status to categorical for native XGBoost handling
    df['Marital_Status'] = df['Marital_Status'].astype('category')
    
    logger.info(f"✅ Feature engineering exacto del notebook completado")
    logger.info(f"Forma final: {df.shape}")
    
    # Define X and y
    X = df.drop(columns=['Response'])
    y = df['Response']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Feature names
    feature_names = X.columns.tolist()
    
    return {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'original_shape': df.shape
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Script de Feature Engineering - EXACTO DEL NOTEBOOK")
    print("Implementa exactamente el código proporcionado por el usuario")