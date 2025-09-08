# 🐛 Pipeline Debug Guide

Esta guía te ayuda a debuggear y ejecutar el pipeline MLOps paso a paso.

## ✅ Quick Start (Ejecutar Todo)

```bash
# Opción 1: Script de inicio rápido
python quick_start.py

# Opción 2: Pipeline manual
python pipeline/run_pipeline.py --data-path data/marketing_campaign.csv
```

## 🔍 Debugging Paso a Paso

### 1. Validación de Componentes

```bash
# Probar todos los componentes individualmente
python validate_pipeline.py
```

Este script valida:
- ✅ Carga de datos (formato semicolon separator)
- ✅ Feature engineering pipeline
- ✅ Third tuning module
- ✅ Temperature calibration (FIXED VERSION)
- ✅ Pipeline orchestrator

### 2. Tests Básicos

```bash
# Tests de dependencias y datos básicos
python debug_test.py
```

### 3. Pipeline Manual por Pasos

Si necesitas ejecutar componentes individuales:

```python
# 1. Preparar datos
from pipeline.feature_engineering import prepare_data_for_training
data = prepare_data_for_training("data/marketing_campaign.csv")

# 2. Third Tuning
from pipeline.third_tuning import run_third_tuning
results = run_third_tuning(X_train, y_train, X_val, y_val, artifacts_dir)

# 3. Entrenamiento final
from pipeline.train_final import train_final_model_pipeline
model_results = train_final_model_pipeline(X_train, y_train, X_test, y_test, feature_names, artifacts_dir)

# 4. Calibración
from pipeline.temperature_calibration import calibrate_model_probabilities
calibrator = calibrate_model_probabilities(model, X_val, y_val, artifacts_dir)

# 5. SHAP Analysis
from pipeline.shap_analysis_pipeline import run_complete_shap_analysis
shap_results = run_complete_shap_analysis(model, X_test, feature_names, artifacts_dir)
```

## 🛠️ Problemas Conocidos y Soluciones

### Error: "no se encontró Python"
```bash
# Verificar Python path
python --version
# O usar python3
python3 quick_start.py
```

### Error: ModuleNotFoundError
```bash
# Instalar dependencias
pip install -r requirements.txt

# O individual:
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib fastapi uvicorn
```

### Error: Dataset not found
```bash
# Verificar que el dataset esté en la ubicación correcta:
data/marketing_campaign.csv
```

### Error: Separator issues
El dataset original usa separador semicolon (`;`), esto ya está configurado en el pipeline.

### Error: Temperature calibration
**✅ FIXED**: El archivo `temperature_calibration.py` ha sido reparado y ahora funciona correctamente.

## 📊 Estructura del Pipeline

```
1. Feature Engineering → 
2. Third Tuning (metodología ganadora) → 
3. Training Final → 
4. Temperature Calibration → 
5. SHAP Analysis
```

## 📁 Artifacts Generados

Después de ejecutar el pipeline, encontrarás:

```
artifacts/
├── feature_pipeline.pkl          # Feature transformer
├── final_model.pkl              # Modelo XGBoost entrenado
├── temperature_calibrator.pkl   # Calibrador de temperatura
├── calibration_metrics.json     # Métricas de calibración
├── pipeline_summary.json        # Resumen completo
├── plots/                       # Gráficos de calibración
│   ├── reliability_diagram.png
│   └── probability_distributions.png
└── shap_values/                 # Análisis SHAP
    ├── shap_explainer.pkl
    ├── shap_values.pkl
    └── plots/
```

## 🎯 Métricas Esperadas

Un pipeline exitoso debe generar métricas similares a:
- **ROC-AUC**: ~0.85-0.90
- **F1 Score**: ~0.60-0.70
- **Temperatura**: ~0.8-1.2

## 🆘 Si Todo Falla

1. **Ejecuta validación**: `python validate_pipeline.py`
2. **Revisa logs**: Busca errores específicos en la salida
3. **Verifica datos**: Asegúrate que `data/marketing_campaign.csv` existe
4. **Prueba componentes**: Usa `debug_test.py` para tests básicos

## 🚀 Deploy a Producción

Una vez que el pipeline funcione:

```bash
# Servir modelo con FastAPI
python src/app.py

# Acceder a la API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [values...]}'
```

## 📝 Logs

Los logs del pipeline se guardan en `pipeline.log` para debugging detallado.