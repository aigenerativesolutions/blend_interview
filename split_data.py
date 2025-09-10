#!/usr/bin/env python3
"""
Script para dividir el dataset en train (80%) y test (20%)
El train se sube al repo para el pipeline, el test se queda local para validación
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

def split_marketing_data():
    """Divide el dataset de marketing en train/test 80/20"""
    
    # Leer el dataset original
    data_path = "marketing-ml-mvp/data/marketing_campaign.csv"
    
    if not os.path.exists(data_path):
        print(f"ERROR: No se encontró el archivo: {data_path}")
        print("Asegúrate de estar en el directorio raíz del proyecto")
        return False
    
    print(f"Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path, sep=";")
    
    print(f"Dataset original: {len(df)} registros, {len(df.columns)} columnas")
    
    # Verificar que existe la columna Response
    if 'Response' not in df.columns:
        print("ERROR: No se encontró la columna 'Response' en el dataset")
        print(f"Columnas disponibles: {list(df.columns)}")
        return False
    
    # Mostrar distribución de clases original
    response_counts = df['Response'].value_counts()
    response_props = df['Response'].value_counts(normalize=True)
    
    print(f"\nDistribución de clases original:")
    print(f"  Clase 0 (No responde): {response_counts[0]} ({response_props[0]:.1%})")
    print(f"  Clase 1 (Responde): {response_counts[1]} ({response_props[1]:.1%})")
    
    if response_props[1] < 0.5:
        print(f"WARNING: Dataset desbalanceado detectado - usando stratified split")
    
    # Split 80/20 con stratified split para mantener proporciones
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,  # Semilla fija para reproducibilidad
        stratify=df['Response']  # CRÍTICO: mantener proporciones de clases
    )
    
    print(f"Split completado:")
    print(f"  Train set: {len(train_df)} registros ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(test_df)} registros ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verificar proporciones después del split
    train_response_props = train_df['Response'].value_counts(normalize=True)
    test_response_props = test_df['Response'].value_counts(normalize=True)
    
    print(f"\nVerificación de proporciones (stratified split):")
    print(f"  Train set - Clase 0: {train_response_props[0]:.1%}, Clase 1: {train_response_props[1]:.1%}")
    print(f"  Test set  - Clase 0: {test_response_props[0]:.1%}, Clase 1: {test_response_props[1]:.1%}")
    print(f"  Original  - Clase 0: {response_props[0]:.1%}, Clase 1: {response_props[1]:.1%}")
    
    # Verificar que las proporciones son similares (tolerancia 1%)
    train_diff = abs(train_response_props[1] - response_props[1])
    test_diff = abs(test_response_props[1] - response_props[1])
    
    if train_diff < 0.01 and test_diff < 0.01:
        print("SUCCESS: Proporciones mantenidas correctamente!")
    else:
        print("WARNING: Proporciones pueden variar ligeramente")
    
    # Guardar train set (para subir a GitHub y entrenar en la nube)
    train_path = "data/train_data.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Train set guardado en: {train_path}")
    
    # Guardar test set (local, NO se sube a GitHub)
    test_path = "data/test_data_local.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Test set guardado en: {test_path} (solo local)")
    
    # Mostrar estadísticas básicas
    print(f"\nEstadísticas del Train Set:")
    print(train_df.describe().round(2))
    
    print(f"\nPrimeras 3 filas del Train Set:")
    print(train_df.head(3))
    
    print(f"\nSplit estratificado completado exitosamente!")
    print(f"Las proporciones de clases se mantuvieron en train y test")
    print(f"Ahora puedes hacer commit del train_data.csv para entrenar en la nube")
    print(f"El test_data_local.csv se quedará local para validación objetiva")
    
    return True

if __name__ == "__main__":
    print("Iniciando división de dataset 80/20...")
    success = split_marketing_data()
    
    if success:
        print("\n" + "="*60)
        print("PROCESO COMPLETADO - STRATIFIED SPLIT")
        print("="*60)
        print("Archivos generados:")
        print("  • data/train_data.csv (80% - para GitHub/pipeline)")
        print("  • data/test_data_local.csv (20% - solo local)")
        print("\nBENEFICIOS del Stratified Split:")
        print("  • Proporciones de clases balanceadas")
        print("  • Entrenamiento sin sesgo")
        print("  • Validación más confiable")
        print("\nPróximos pasos:")
        print("  1. git add data/train_data.csv")
        print("  2. git commit -m 'feat: Add stratified train dataset for pipeline'")
        print("  3. git push origin main")
        print("  4. Esperar que el pipeline entrene el modelo")
        print("  5. Ejecutar test_local.py para validar con 20% no visto")
    else:
        print("ERROR: Error al procesar el dataset")