# 🎯 Marketing Campaign Predictor - Streamlit App

## Professional MLOps Demo para BLEND

Una aplicación completa de **Machine Learning + LLM** que combina:
- 🤖 **Modelo XGBoost** para predicción de respuesta a campañas
- 🧠 **LLM Assistant** para estrategias de marketing  
- 📊 **Dashboard interactivo** con métricas y visualizaciones
- 🎯 **Interfaz profesional** desarrollada en Streamlit

---

## 🚀 Quick Start - Deployment Local

### **1. Instalar Dependencias**
```bash
# Opción A: Instalar solo lo necesario para Streamlit
pip install -r requirements_streamlit.txt

# Opción B: Instalar todo (si ya tienes requirements.txt)  
pip install -r requirements.txt
pip install streamlit plotly openai groq anthropic
```

### **2. Ejecutar la Aplicación**
```bash
# Desde el directorio marketing-ml-mvp
streamlit run app_streamlit.py
```

### **3. Acceder a la App**
- **URL**: `http://localhost:8501`
- **Documentación**: Incluida en la interfaz

---

## 🤖 Configurar LLM Assistant (Opcional)

Para habilitar el **Marketing Assistant** con IA, configura una API key:

### **Opción 1: OpenAI (Recomendado)**
```bash
# Windows
set OPENAI_API_KEY=tu-openai-api-key

# Linux/Mac  
export OPENAI_API_KEY="tu-openai-api-key"
```

### **Opción 2: Groq (Rápido y gratuito)**
```bash
# Registrar en: https://console.groq.com/
set GROQ_API_KEY=tu-groq-api-key
```

### **Opción 3: Local con Ollama (Gratis)**
```bash
# Instalar: https://ollama.ai/
ollama pull llama3:8b
ollama serve
```

### **Sin API Key**
La app funcionará perfectamente en **modo demo** con respuestas predefinidas.

---

## 📱 Funcionalidades

### **🏠 Dashboard Principal**
- Métricas clave del modelo y campañas
- Visualizaciones interactivas con Plotly
- Feature importance del modelo XGBoost

### **🎯 Predicción de Clientes**
- Formulario completo para características del cliente
- Predicción en tiempo real con probabilidades
- Gauge de confianza y recomendaciones

### **🤖 Marketing Assistant**  
- Chat IA especializado en marketing
- Contexto específico del modelo y datos
- Estrategias personalizadas y insights

### **📊 Análisis del Modelo**
- Métricas de performance detalladas
- Feature importance interactiva
- Insights técnicos del modelo

---

## 🏗️ Arquitectura de la App

```
app_streamlit.py          # Aplicación principal
├── Dashboard             # Métricas y visualizaciones  
├── Predicción           # Interfaz de predicción
├── Marketing Assistant  # Chat LLM
└── Model Analytics     # Análisis técnico

llm_assistant.py         # LLM Marketing Assistant
├── OpenAI Integration   # ChatGPT integration
├── Groq Integration     # Fast LLM option
├── Local Ollama         # Self-hosted option
└── Demo Mode           # Fallback responses

requirements_streamlit.txt # Dependencies optimizadas
```

---

## 🎨 Screenshots y Demo

### **Dashboard Principal**
- **Métricas**: Response rate, Model accuracy, ROI improvement
- **Visualizaciones**: Feature importance, Response distribution

### **Predicción Interactiva**
- **Formulario**: 25+ campos de características del cliente
- **Resultado**: Probability gauge + Recomendaciones

### **Chat Assistant**
- **Contexto**: Entrenado con datos del modelo XGBoost
- **Especialización**: Marketing campaigns, segmentación, ROI

---

## 🛠️ Troubleshooting

### **Error: "Module not found"**
```bash
# Instalar dependencias faltantes
pip install streamlit plotly pandas numpy scikit-learn xgboost
```

### **Error: "Model not loaded"**  
La app funciona en modo demo si no encuentra el modelo entrenado. Para usar el modelo real:
1. Ejecutar el pipeline: `python pipeline/run_pipeline.py`
2. Verificar que existe `artifacts/final_model.pkl`

### **LLM Assistant no responde**
1. Verificar API key configurada
2. Comprobar conexión a internet
3. La app usará modo demo automáticamente

### **Puerto ocupado**
```bash
# Usar puerto diferente
streamlit run app_streamlit.py --server.port 8502
```

---

## 🎯 Para la Demo BLEND

### **Flujo Recomendado:**

1. **Abrir Dashboard** - Mostrar métricas y model performance
2. **Hacer Predicción** - Usar datos de ejemplo, mostrar probability gauge
3. **Usar Marketing Assistant** - Preguntar sobre estrategias de segmentación  
4. **Model Analytics** - Explicar feature importance y insights técnicos

### **Preguntas Sugeridas para LLM:**
- "¿Cómo puedo mejorar la tasa de respuesta de mis campañas?"
- "¿Qué características predicen mejor la respuesta del cliente?"
- "¿Cuál es la mejor estrategia para clientes de alto valor?"

### **Puntos Clave a Destacar:**
- ✅ **MLOps Pipeline completo** - Desde datos hasta deployment
- ✅ **Modelo explicable** - SHAP values y feature importance
- ✅ **IA Generativa integrada** - LLM para insights de marketing
- ✅ **Interfaz profesional** - Lista para producción
- ✅ **Escalabilidad** - Compatible con GCP deployment

---

## 📞 Soporte

**Demo Mode**: La app está diseñada para funcionar sin configuración adicional.

**Production Mode**: Con modelo entrenado y API keys configuradas.

¡Perfecto para impresionar en BLEND! 🎯