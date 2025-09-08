"""
Feature Engineering Pipeline - Serializable
Reproduce exactamente las transformaciones del notebook
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import json
import logging
from typing import Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketingFeaturePipeline:
    """
    Pipeline de feature engineering para marketing campaign prediction.
    Reproduce exactamente las transformaciones del notebook ganador.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features derivadas exactamente como en el notebook
        """
        df = df.copy()
        
        # 1. Convert Dt_Customer to datetime and calculate Months_As_Customer
        if 'Dt_Customer' in df.columns:
            df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
            reference_date = pd.Timestamp('2014-09-01')
            
            months = (
                (reference_date.year - df['Dt_Customer'].dt.year) * 12 +
                (reference_date.month - df['Dt_Customer'].dt.month)
            )
            
            # -1 if day dont match with reference_date
            adjust = (reference_date.day < df['Dt_Customer'].dt.day).astype(int)
            df['Months_As_Customer'] = (months - adjust).astype(int)
        
        # 2. Calculate Age using fixed year 2014
        if 'Year_Birth' in df.columns:
            df['Age'] = 2014 - df['Year_Birth']
        
        # 3. Combine Kidhome and Teenhome to reduce columns
        if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
            df['Total_Kids'] = df['Kidhome'] + df['Teenhome']
        
        # 4. Calculate Total_Spent (suma de todos los Mnt*)
        spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        df['Total_Spent'] = df[spending_cols].sum(axis=1)
        
        # 5. Drop irrelevant columns
        columns_to_drop = ['ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        logger.info(f"Features creadas: Age, Total_Kids, Total_Spent, Months_As_Customer")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Codificar variables categóricas exactamente como en el notebook
        """
        df = df.copy()
        
        # Map Education to ordered cardinality (como en el notebook)
        if 'Education' in df.columns:
            education_order = {
                'Basic': 0,
                '2n Cycle': 1,
                'Graduation': 2,
                'Master': 3,
                'PhD': 4
            }
            
            if fit:
                # Store the mapping for later use
                self.label_encoders['Education'] = education_order
                df['Education'] = df['Education'].map(education_order).astype('int')
                logger.info(f"Education mapped con orden: {education_order}")
            else:
                # Use stored mapping
                if 'Education' in self.label_encoders:
                    df['Education'] = df['Education'].map(self.label_encoders['Education']).fillna(1).astype('int')
                else:
                    df['Education'] = df['Education'].map(education_order).fillna(1).astype('int')
        
        # Convert Marital_Status to categorical for native XGBoost handling
        if 'Marital_Status' in df.columns:
            if fit:
                # Store unique values for consistency
                self.label_encoders['Marital_Status'] = df['Marital_Status'].unique().tolist()
            df['Marital_Status'] = df['Marital_Status'].astype('category')
            logger.info(f"Marital_Status converted to categorical")
        
        return df
    
    # XGBoost no requiere escalado de features - función eliminada para reproducir notebook exacto
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Response') -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Ajustar y transformar datos de entrenamiento
        """
        logger.info("🔧 Iniciando feature engineering pipeline...")
        
        # 1. Crear features derivadas
        df_processed = self.create_features(df)
        
        # 2. Separar features y target
        if target_col in df_processed.columns:
            X = df_processed.drop(columns=[target_col])
            y = df_processed[target_col].values
        else:
            X = df_processed
            y = None
        
        # 3. Codificar categóricas
        X = self.encode_categorical_features(X, fit=True)
        
        # 4. Guardar nombres de features (sin escalado, XGBoost no lo requiere)
        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"✅ Pipeline fitted. Features finales: {len(self.feature_names)}")
        logger.info(f"Features: {self.feature_names}")
        
        return X.values, y, self.feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transformar nuevos datos usando pipeline fitted
        """
        if not self.is_fitted:
            raise ValueError("Pipeline debe ser fitted antes de transform")
        
        logger.info("🔄 Transformando nuevos datos...")
        
        # 1. Crear features derivadas
        df_processed = self.create_features(df)
        
        # 2. Remover target si existe
        X = df_processed.drop(columns=['Response'], errors='ignore')
        
        # 3. Codificar categóricas
        X = self.encode_categorical_features(X, fit=False)
        
        # 4. Asegurar orden correcto de features (sin escalado)
        X = X[self.feature_names]
        
        logger.info(f"✅ Datos transformados: {X.shape}")
        
        return X.values
    
    def save(self, filepath: Path) -> None:
        """
        Serializar pipeline completo
        """
        pipeline_data = {
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"💾 Pipeline guardado en {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'MarketingFeaturePipeline':
        """
        Cargar pipeline serializado
        """
        pipeline_data = joblib.load(filepath)
        
        pipeline = cls()
        pipeline.label_encoders = pipeline_data['label_encoders']
        pipeline.feature_names = pipeline_data['feature_names']
        pipeline.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"📂 Pipeline cargado desde {filepath}")
        logger.info(f"Features: {len(pipeline.feature_names)}")
        
        return pipeline

def prepare_data_for_training(data_path: Path, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Preparar datos para entrenamiento con feature engineering
    """
    logger.info(f"📊 Cargando datos desde {data_path}")
    
    # Cargar datos con separador correcto
    df = pd.read_csv(data_path, sep=';')
    logger.info(f"Datos cargados: {df.shape}")
    
    # Remover columnas innecesarias
    columns_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    logger.info(f"Datos después de limpiar columnas: {df.shape}")
    
    # Crear pipeline
    pipeline = MarketingFeaturePipeline()
    
    # Aplicar transformaciones
    X, y, feature_names = pipeline.fit_transform(df)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'pipeline': pipeline,
        'original_shape': df.shape
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Ejemplo de uso
    data_path = Path("data/marketing_campaign_data.csv")
    if data_path.exists():
        data = prepare_data_for_training(data_path)
        
        # Guardar pipeline
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        pipeline = data['pipeline']
        pipeline.save(artifacts_dir / "feature_pipeline.pkl")
        
        print(f"✅ Pipeline guardado. Features: {len(data['feature_names'])}")
    else:
        print(f"❌ Archivo no encontrado: {data_path}")