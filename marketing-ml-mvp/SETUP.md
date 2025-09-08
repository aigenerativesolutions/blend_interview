# 🚀 Setup Rápido - MLOps Pipeline

## ⚡ Quick Start (5 minutos)

### 1. **Preparar el Entorno**

```bash
# Clonar/navegar al proyecto
cd marketing-ml-mvp

# Instalar dependencias
pip install -r requirements.txt

# Crear directorio de datos
mkdir -p data
```

### 2. **Colocar tus Datos**

```bash
# Copiar tu archivo de datos
cp /ruta/a/tu/archivo.csv data/marketing_campaign_data.csv

# O crear archivo de ejemplo para testing:
echo "Education,Marital_Status,Income,Year_Birth,Dt_Customer,..." > data/marketing_campaign_data.csv
```

### 3. **Ejecutar Pipeline Localmente**

```bash
# Ejecutar pipeline completo (15-30 minutos)
python pipeline/run_pipeline.py \
  --data-path data/marketing_campaign_data.csv \
  --artifacts-dir artifacts \
  --test-size 0.2 \
  --val-split 0.2

# Ver resultados
cat artifacts/pipeline_summary.json
```

### 4. **Probar API**

```bash
# En otra terminal, iniciar API
python -m uvicorn src.api.main:app --reload --port 8080

# En tu navegador: http://localhost:8080/docs
# O curl:
curl http://localhost:8080/health
curl http://localhost:8080/model/info
```

## 🔧 Setup Completo para GitHub Actions

### 1. **Crear Repositorio GitHub**

```bash
# Inicializar git (si no está inicializado)
git init
git add .
git commit -m "Initial MLOps pipeline setup"

# Crear repo en GitHub y conectar
git remote add origin https://github.com/tu-usuario/marketing-ml-mvp.git
git push -u origin main
```

### 2. **Configurar Secrets para GCP (Opcional)**

Si quieres desplegar a Google Cloud Platform:

1. Ve a **GitHub repo → Settings → Secrets and variables → Actions**

2. Agrega estos secrets:
   ```
   GCP_SA_KEY: [Tu Service Account Key JSON completo]
   GCP_PROJECT_ID: tu-proyecto-gcp-id
   GCS_BUCKET: tu-bucket-para-modelos
   ```

3. Para obtener Service Account Key:
   ```bash
   # En Google Cloud Console:
   # IAM & Admin → Service Accounts → Create/Select Account
   # Add roles: Cloud Run Admin, Storage Admin, Cloud Build Editor
   # Create Key → JSON → Download
   ```

### 3. **Activar GitHub Actions**

Los workflows ya están configurados en `.github/workflows/`. Se activarán automáticamente cuando:

- Hagas push a `main` o `develop`
- Cambies archivos en `pipeline/` o `data/`
- Manualmente desde GitHub Actions tab
- Cada domingo (reentrenamiento programado)

## 📊 Estructura de Datos Esperada

Tu archivo CSV debe tener estas columnas:

```csv
Education,Marital_Status,Income,Year_Birth,Dt_Customer,Kidhome,Teenhome,
MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,
NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,
AcceptedCmp1,AcceptedCmp2,AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,
Recency,Complain,Response
```

**Importante**: `Response` es la columna target (0/1).

## 🎯 Variables de Configuración

### En `pipeline/run_pipeline.py`:

```python
# Ajustar estos parámetros según necesites:
--data-path: Ruta al archivo CSV
--artifacts-dir: Donde guardar modelos (default: artifacts)
--test-size: Proporción para test (default: 0.2)
--val-split: Proporción para validation (default: 0.2)
```

### En `.github/workflows/ml-pipeline.yml`:

```yaml
# Thresholds de calidad del modelo:
min_roc_auc = 0.75  # ROC-AUC mínimo
min_f1_score = 0.60 # F1-Score mínimo

# Horario de reentrenamiento automático:
schedule:
  - cron: '0 2 * * 0'  # Domingo 2:00 AM
```

## 🔍 Verificar que Todo Funciona

### 1. **Pipeline Local**

```bash
# Debe generar estos artifacts:
ls artifacts/
# Expected:
# feature_pipeline.pkl
# final_model.pkl
# temperature_calibrator.pkl
# pipeline_summary.json
# plots/
# shap_values/
```

### 2. **GitHub Actions**

```bash
# Hacer push y ver workflow:
git add .
git commit -m "Test pipeline"
git push origin main

# En GitHub: Actions tab → Ver workflow running
# Debe pasar todos los steps sin errores
```

### 3. **API Funcionando**

```bash
# Todos estos endpoints deben responder:
curl http://localhost:8080/health
curl http://localhost:8080/model/info
curl http://localhost:8080/model/sample-input
curl http://localhost:8080/model/feature-importance
```

## ⚠️ Troubleshooting Común

### "ModuleNotFoundError"
```bash
# Asegúrate de estar en el directorio correcto:
cd marketing-ml-mvp
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -r requirements.txt
```

### "Data file not found"
```bash
# Verificar que el archivo existe:
ls -la data/marketing_campaign_data.csv

# Si no existe, crear archivo de ejemplo o copiar tus datos
```

### "Model quality below thresholds"
```bash
# El modelo no pasa los thresholds de calidad
# Opciones:
# 1. Mejorar datos/features
# 2. Ajustar thresholds en ml-pipeline.yml
# 3. Revisar hiperparámetros en third_tuning.py
```

### "Pipeline artifacts not found"
```bash
# La API no encuentra los modelos
# Ejecutar pipeline primero:
python pipeline/run_pipeline.py --data-path data/marketing_campaign_data.csv

# Verificar que se generaron artifacts:
ls -la artifacts/
```

### GitHub Actions falla
```bash
# Ver logs detallados en GitHub:
# Actions tab → Select failed run → Expand failed step

# Problemas comunes:
# - Secrets mal configurados (GCP)
# - Archivo de datos no encontrado
# - Dependencies missing
```

## 🔄 Workflow de Desarrollo

### Para desarrollo local:
```bash
# 1. Modificar código
vim pipeline/feature_engineering.py

# 2. Ejecutar pipeline localmente
python pipeline/run_pipeline.py --data-path data/marketing_campaign_data.csv

# 3. Probar API
python -m uvicorn src.api.main:app --reload --port 8080

# 4. Commit cuando esté listo
git add . && git commit -m "Update feature engineering"
```

### Para deploy automático:
```bash
# 1. Push a main para trigger automático
git push origin main

# 2. Ver GitHub Actions ejecutarse
# 3. Si pasa calidad → deploy automático
# 4. API actualizada en producción
```

## 📚 Próximos Pasos

Una vez que tienes el setup básico funcionando:

1. **Personalizar Hiperparámetros**: Edita `pipeline/third_tuning.py`
2. **Ajustar Features**: Modifica `pipeline/feature_engineering.py`  
3. **Monitoreo**: Configura alertas en GitHub/GCP
4. **Scaling**: Configura auto-scaling en Cloud Run
5. **Datos**: Setup pipeline de datos automatizado

## 🎉 ¡Ya estás listo!

Con este setup tienes un **pipeline MLOps completo** que impresionará en cualquier entrevista técnica. Muestra dominio de:

- ✅ ML Engineering
- ✅ MLOps/CI-CD  
- ✅ Cloud Computing
- ✅ API Development
- ✅ Best Practices

**¡Éxito en tu entrevista! 🚀**