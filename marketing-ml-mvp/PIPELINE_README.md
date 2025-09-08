# 🤖 MLOps Pipeline Completo - Marketing Campaign Prediction

## 🎯 Overview

Este es un pipeline MLOps completo que implementa **exactamente tu metodología ganadora del notebook**, automatizado con **GitHub Actions** para CI/CD. Reproduce el flujo completo: Features → 3er Tuning → Training → Calibration → SHAP, todo serializado y versionado.

## 📊 ¿Qué es MLOps?

**MLOps = DevOps para Machine Learning**

- **Automatiza** el entrenamiento cuando cambias datos o código
- **Versiona** modelos, datos y código
- **Despliega** automáticamente si las métricas mejoran
- **Monitorea** el rendimiento en producción
- **Reproduce** experimentos exactamente

## 🏗️ Arquitectura del Pipeline

```
📁 marketing-ml-mvp/
├── 🤖 pipeline/
│   ├── feature_engineering.py      # Features serializables (Age, Total_Spent, etc.)
│   ├── third_tuning.py            # SOLO 3er tuning (metodología ganadora)
│   ├── train_final.py             # Modelo final con mejores params
│   ├── temperature_calibration.py # Temperature Scaling
│   ├── shap_analysis_pipeline.py  # SHAP completo + plots
│   └── run_pipeline.py            # Orquestador principal
├── 🔄 .github/workflows/
│   ├── ml-pipeline.yml           # CI/CD para entrenamiento
│   └── deploy-api.yml            # Despliegue automático
├── 📦 artifacts/                  # Modelos y resultados serializados
├── 🌐 src/api/                   # API con pipeline integrado
└── 📚 docs/                      # Documentación
```

## 🚀 Flujo de Trabajo Automático

### 1. **Haces Push a GitHub**
```bash
git add .
git commit -m "Update data or code"
git push origin main
```

### 2. **GitHub Actions se activa automáticamente**
- Detecta cambios en `data/` o `pipeline/`
- Ejecuta el pipeline completo
- Valida calidad del modelo
- Despliega si mejoran las métricas

### 3. **Pipeline MLOps ejecuta:**

#### Step 1: **Feature Engineering** 🔧
```python
# Reproduce EXACTAMENTE tu notebook:
- Age = current_year - Year_Birth
- Total_Spent = suma de Mnt*
- Customer_Days desde Dt_Customer
- Codificación de Education/Marital_Status
- Scaling de features numéricas

# ✅ Serializa: feature_pipeline.pkl
```

#### Step 2: **3er Tuning (Metodología Ganadora)** 🎯
```python
# TU configuración exacta que ganó:
third_param_grid = {
    'n_estimators': [290, 300, 310],
    'max_depth': [5],
    'learning_rate': [0.01],
    'subsample': [0.9],
    'colsample_bytree': [0.8],
    'gamma': [0.05, 0.1],
    'reg_alpha': [0.5],
    'reg_lambda': [2.5, 3.0]
}

# ✅ Serializa: best_params_tuning3.json, optimal_threshold.json
```

#### Step 3: **Entrenamiento Final** 🏋️
```python
# Entrena con los mejores parámetros del 3er tuning
# Calcula threshold óptimo con PR curve
# Genera plots de evaluación

# ✅ Serializa: final_model.pkl, final_model_metadata.json
```

#### Step 4: **Temperature Calibration** 🌡️
```python
# Tu implementación exacta de Temperature Scaling
# Reliability diagrams antes/después
# Mejora del Brier Score

# ✅ Serializa: temperature_calibrator.pkl
```

#### Step 5: **SHAP Analysis** 🔍
```python
# Análisis completo como en tu notebook:
- SHAP values globales
- Feature importance
- Dependence plots (top 10 features)
- Waterfall plots (samples representativos)

# ✅ Serializa: shap_explainer.pkl, shap_values.npz
```

### 4. **API Actualizada Automáticamente** 🌐
- Carga todos los artifacts del pipeline
- Endpoints con SHAP integrado
- Predicciones con calibración
- Feature engineering automático

## 📁 Artifacts Serializados

Todos los componentes se guardan automáticamente:

```
artifacts/
├── feature_pipeline.pkl           # Transformador de features fitted
├── best_params_tuning3.json      # Mejores parámetros del 3er tuning
├── final_model.pkl               # Modelo XGBoost entrenado
├── temperature_calibrator.pkl    # Calibrador de temperatura
├── optimal_threshold.json        # Threshold óptimo encontrado
├── shap_values/
│   ├── shap_explainer.pkl        # SHAP explainer
│   ├── shap_values.npz          # Valores SHAP
│   └── global_feature_importance.json
├── plots/                        # Todos los plots generados
└── pipeline_summary.json        # Resumen completo
```

## 🎮 Cómo Usar

### Opción 1: Ejecutar Pipeline Localmente

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar tus datos
cp tu_archivo.csv data/marketing_campaign_data.csv

# 3. Ejecutar pipeline completo
python pipeline/run_pipeline.py \
  --data-path data/marketing_campaign_data.csv \
  --artifacts-dir artifacts

# 4. Ver resultados
cat artifacts/pipeline_summary.json
```

### Opción 2: GitHub Actions Automático

```bash
# 1. Subir datos al repo
git add data/marketing_campaign_data.csv
git commit -m "Add training data"
git push origin main

# 2. Ver workflow en GitHub Actions
# - Va a https://github.com/tu-repo/actions
# - El workflow se ejecuta automáticamente
# - Descarga artifacts cuando termine
```

### Opción 3: API en Producción

```bash
# Después del pipeline, la API ya tiene todo:
curl -X POST "https://your-api-url/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "Education": "Graduation",
      "Marital_Status": "Married", 
      "Income": 58138.0,
      "Year_Birth": 1985,
      "Dt_Customer": "2015-01-01",
      ...
    },
    "use_calibration": true
  }'
```

## 🔄 GitHub Actions Workflow

### `ml-pipeline.yml` - Entrenamiento Automático

```yaml
# Se activa cuando:
- Push a main/develop
- Cambios en pipeline/ o data/
- Cada domingo (reentrenamiento programado)
- Manualmente con workflow_dispatch

# Ejecuta:
1. Valida datos
2. Ejecuta pipeline completo  
3. Valida métricas (ROC-AUC > 0.75, F1 > 0.60)
4. Sube artifacts
5. Triggea despliegue si pasa calidad
```

### `deploy-api.yml` - Despliegue Automático

```yaml
# Se activa cuando:
- El pipeline de entrenamiento termina exitoso
- Manualmente para staging/production

# Ejecuta:
1. Build Docker con artifacts
2. Deploy a Cloud Run staging
3. Health checks y API tests
4. Deploy a production con traffic split gradual
```

## 📊 Monitoreo y Métricas

### Métricas de Calidad
- **ROC-AUC**: Debe ser ≥ 0.75
- **F1-Score**: Debe ser ≥ 0.60
- **Brier Improvement**: Mejora con calibración
- **Temperature**: Parámetro de calibración

### Artifacts Versionados
- Cada run genera artifacts con timestamp
- Modelos se guardan en GCS con versión
- Registry de modelos automático
- Rollback fácil a versiones anteriores

## 🚨 Qué Pasa si Falla

### Fallos Comunes y Soluciones

1. **"Data file not found"**
   ```bash
   # Solución: Subir archivo de datos
   cp tu_archivo.csv data/marketing_campaign_data.csv
   git add data/marketing_campaign_data.csv
   git commit -m "Add data"
   git push
   ```

2. **"Model quality below thresholds"**
   ```bash
   # El modelo no pasa calidad (ROC-AUC < 0.75)
   # Revisa los datos o ajusta thresholds en .github/workflows/ml-pipeline.yml
   ```

3. **"Pipeline artifacts not found"**
   ```bash
   # API no encuentra artifacts
   # Ejecuta pipeline localmente o revisa GitHub Actions
   python pipeline/run_pipeline.py --data-path data/marketing_campaign_data.csv
   ```

## 🎯 Para tu Entrevista

### Lo que puedes mostrar:

1. **Pipeline Completo**: "Implementé MLOps end-to-end"
2. **Metodología Ganadora**: "Usé mi 3er tuning que dio mejores resultados"
3. **CI/CD Automático**: "Push = entrenamiento + despliegue automático"
4. **Serialización**: "Todo versionado y reproducible"
5. **Monitoreo**: "Validación automática de calidad"
6. **SHAP Integrado**: "Explicabilidad en la API"

### Preguntas que puedes responder:

- **"¿Cómo garantizas reproducibilidad?"** → "Versionado de código, datos y modelos"
- **"¿Cómo manejas el modelo drift?"** → "Reentrenamiento automático programado"
- **"¿Cómo despliegas modelos?"** → "CI/CD con validación automática"
- **"¿Cómo explicas predicciones?"** → "SHAP integrado en la API"

## 🔧 Configuración Avanzada

### Secrets de GitHub (Requeridos para GCP)

```bash
# En GitHub repo → Settings → Secrets:
GCP_SA_KEY: [Service Account Key JSON]
GCP_PROJECT_ID: tu-proyecto-gcp  
GCS_BUCKET: tu-bucket-models
```

### Configuración del Pipeline

```python
# En pipeline/run_pipeline.py puedes ajustar:
- test_size: Proporción test set
- val_split: Proporción validation
- Thresholds de calidad
- Parámetros del tuning
```

### Triggers Personalizados

```yaml
# En .github/workflows/ml-pipeline.yml:
schedule:
  - cron: '0 2 * * 0'  # Cada domingo 2:00 AM
  
# Cambiar por tu horario preferido
```

## 🏆 Ventajas de este Approach

1. **✅ Reproduce tu metodología exacta**
2. **🚀 Automatización completa**
3. **📦 Versionado de todo**
4. **🔍 Trazabilidad total**
5. **⚡ Despliegue sin downtime**
6. **📊 Monitoreo continuo**
7. **🔄 Rollback fácil**
8. **🎯 Production-ready**

## 📞 Troubleshooting

### Ver logs del pipeline:
```bash
# Localmente:
tail -f pipeline.log

# En GitHub Actions:
# Ve a Actions tab → Select workflow run → Expand steps
```

### Test manual de la API:
```bash
# Health check
curl https://your-api-url/health

# Predict con datos sample
curl https://your-api-url/model/sample-input
curl -X POST https://your-api-url/predict -d @sample.json
```

### Debug artifacts:
```bash
# Ver qué se generó
ls -la artifacts/
python -c "import json; print(json.load(open('artifacts/pipeline_summary.json')))"
```

---

## 🎉 ¡Tu pipeline MLOps está listo!

Este setup te da **credibilidad técnica máxima** para tu entrevista. Muestra conocimiento de:

- **ML Engineering** (tu metodología optimizada)
- **MLOps** (CI/CD, versionado, monitoreo)  
- **Cloud Native** (Docker, GCP, APIs)
- **Best Practices** (testing, serialización, documentación)

**¡Buena suerte en tu entrevista! 🚀**