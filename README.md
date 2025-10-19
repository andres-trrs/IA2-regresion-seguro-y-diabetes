# 🩺 Seguro Médico & 🧪 Diabetes – IA2

## 📋 Descripción
Aplicación completa (ML + API + Front) que:
- Entrena dos modelos:
  - **Regresión (Ridge)** para costos de **seguro médico**.
  - **Clasificación (LogisticRegression)** para **predicción de diabetes**.
- Expone ambos modelos vía **FastAPI** (deploy en Render).
- Frontend **HTML + JS** (deploy en Vercel) para usar la API.
- Incluye **evaluación**, **umbral óptimo**, **importancias de variables (RandomForest)**, **optimización**, **sesgos** y **contexto de datos**.

---

## 🌐 Producción

- **Backend (FastAPI, Swagger UI):**  
  `https://ia2-regresion-seguro-y-diabetes.onrender.com/docs`

- **Frontend (Vercel):**  
  `https://<TU_FRONTEND_VERCEL>.vercel.app`  
  > Reemplaza con tu URL real de Vercel.

---

## 🗂️ Estructura del Proyecto

```
.
├── ml-app/                      # Backend (FastAPI) + entrenamiento
│   ├── api/
│   │   └── main.py              # Endpoints /predict/*
│   ├── data/
│   │   ├── insurance.csv
│   │   └── diabetes.csv
│   ├── models/
│   │   ├── insurance_model.pkl
│   │   ├── diabetes_model.pkl
│   │   ├── rf_insurance.pkl
│   │   ├── rf_diabetes.pkl
│   │   └── diabetes_threshold.json
│   ├── reports/
│   │   ├── insurance_feature_importance.csv/.png
│   │   ├── diabetes_feature_importance.csv/.png
│   │   ├── diabetes_F1_vs_threshold.png
│   │   ├── diabetes_ROC.png
│   │   └── diabetes_PR.png
│   ├── requirements.txt
│   └── train.py                 # Script de entrenamiento
│
├── web/
│   └── index.html               # Frontend estático (HTML + JS)
│
├── .python-version              # 3.10.13 (forzado para Render)
└── README.md
```

---

## 🚀 Cómo ejecutar localmente

### 1) Clonar y crear entorno
```bash
git clone https://github.com/andres-trrs/IA2-regresion-seguro-y-diabetes
cd IA2-regresion-seguro-y-diabetes
cd ml-app
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) (Opcional) Re-entrenar modelos
```bash
python train.py
```
> Deja los artefactos en `ml-app/models/` y gráficos/reportes en `ml-app/reports/`.

### 3) Levantar la API
```bash
uvicorn api.main:app --reload
```
- Docs locales: `http://127.0.0.1:8000/docs`

### 4) Frontend local rápido (opcional)
- Abre `web/index.html` con Live Server (VSCode) o cualquier servidor estático.

---

## 🔌 Endpoints

### GET `/`
Resumen del servicio y umbral de diabetes.

### POST `/predict/insurance`
**Body JSON**
```json
{
  "age": 30,
  "bmi": 25.0,
  "children": 0,
  "sex": "male",
  "smoker": "no",
  "region": "southwest"
}
```
**Respuesta**
```json
{ "prediction": 3157.31 }
```

### POST `/predict/diabetes`
**Body JSON**
```json
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 72,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 28.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 45
}
```
**Respuesta**
```json
{ "probability": 0.431, "threshold": 0.4, "prediction": 1 }
```

---

## 🧪 Probar desde consola

```bash
# Seguro
curl -X POST https://ia2-regresion-seguro-y-diabetes.onrender.com/predict/insurance   -H "Content-Type: application/json"   -d '{"age":30,"bmi":25.0,"children":0,"sex":"male","smoker":"no","region":"southwest"}'

# Diabetes
curl -X POST https://ia2-regresion-seguro-y-diabetes.onrender.com/predict/diabetes   -H "Content-Type: application/json"   -d '{"Pregnancies":2,"Glucose":130,"BloodPressure":72,"SkinThickness":20,"Insulin":80,"BMI":28.0,"DiabetesPedigreeFunction":0.5,"Age":45}'
```

---

## 📊 Respuestas a las Preguntas del Enunciado

### 1) ¿Cuál es el **umbral ideal** para el modelo de predicción de diabetes?
- Se evaluó **F1 vs. umbral** (gráfico en `reports/diabetes_F1_vs_threshold.png`).
- **Resultado:** **0.40** (balancea precisión y recall mejor que el 0.50 clásico).
- La API usa ese umbral al predecir (`prediction = prob >= 0.4`).

---

### 2) ¿Cuáles son los **factores que más influyen** en el precio del seguro?
Con **RandomForestRegressor** (solo para interpretar), las **top features** fueron:
- **Fumador (smoker)**: impacto dominante (eleva mucho la prima).
- **BMI (IMC)**: mayor IMC incrementa costos.
- **Edad (age)**: costos tienden a subir con la edad.
- (Menor influencia) **children**, **region**, **sex**.  
Ver `reports/insurance_feature_importance.png` y `.csv`.

---

### 3) Análisis comparativo de **importancia de características** con RandomForest
- **Seguro (regresión):** `smoker` > `bmi` > `age` > `children/region/sex`.
- **Diabetes (clasificación):** típicamente `Glucose` > `BMI` > `Age` > `Pregnancies` > `DiabetesPedigreeFunction` > `BloodPressure`/`SkinThickness`.  
  (Ver `reports/diabetes_feature_importance.png` y `.csv`).

> Nota: RF se usa aquí **para interpretación/benchmark**, el servicio en producción sirve los modelos base (Ridge/LogReg).

---

### 4) ¿Qué **técnica de optimización** mejora el rendimiento de ambos modelos?
- **Seguro (Ridge Regression):**
  - Regularización **Ridge** estabiliza coeficientes y reduce overfitting frente a OLS.
  - Sugerido: **GridSearchCV** en `alpha` (e.g. `[0.1, 1, 10]`) con CV.
- **Diabetes (Logistic Regression):**
  - **Ajuste de umbral** (0.40) mejora **F1** vs. umbral 0.50.
  - Sugerido: **`class_weight='balanced'`** si hay desbalance y **GridSearchCV** sobre `C` y penalización (l2).
- **Ambos:** **Validación cruzada** + Pipeline con **escalamiento** si se agregan variables con distinta escala.

---

### 5) **Contexto de los datos**
- **Insurance** (`insurance.csv`): dataset clásico de Kaggle con costos médicos anuales y atributos (edad, bmi, hijos, sexo, fumador, región). Población de EE. UU. (no necesariamente representativa de otros países).
- **Diabetes** (`diabetes.csv` – PIMA Indians Diabetes Database): registros médicos de mujeres de la etnia Pima de 21+ años (Kaggle/UCI). Útil para aprendizaje, **pero no** generalizable a toda la población.

---

### 6) **Sesgos** de los modelos y por qué
- **Diabetes (PIMA):**
  - **Cobertura limitada** (mujeres Pima) → sesgo de selección.  
  - Riesgo de **mala generalización** a otros grupos demográficos.
  - **Clase positiva menor** (diabetes) → desbalance; afecta umbrales/metricas (por eso se ajustó a 0.40).
- **Seguro:**
  - **Factor “smoker”** concentra gran varianza; puede **sobrerreaccionar** si la etiqueta “smoker” está mal reportada.
  - Variables como **region/sex** pueden capturar **efectos socioeconómicos**/sistémicos no causalmente relacionados con costos.
- **Mitigaciones**: validación en nuevos datos, revisión de drift, reportes de fairness si se escala a producción, y evitar interpretaciones causales.

---

## 🔄 Funcionamiento Paso a Paso (API)

### 🩺 Seguro – Flujo
```
Input JSON (age, bmi, children, sex, smoker, region)
   ↓
Preprocesamiento (coincide con entrenamiento)
   ↓
Ridge.predict → costo estimado
   ↓
Respuesta: {"prediction": nnnn.nn}
```

### 🧪 Diabetes – Flujo
```
Input JSON (8 features PIMA)
   ↓
LogisticRegression.predict_proba → probabilidad
   ↓
Comparación con umbral=0.40
   ↓
Respuesta: {"probability": p, "threshold": 0.4, "prediction": 0/1}
```

---

## 🛠️ Deploy (resumen)

### Backend – Render
- **Root Directory:** `ml-app`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port 10000`
- **Python:** `.python-version` con `3.10.13`
- **CORS:** en `main.py` agrega tu dominio de Vercel en `allow_origins`.

### Frontend – Vercel
- **Root Directory:** `web`
- **Framework Preset:** *Other*
- En `web/index.html`, setea:
  ```js
  const BASE = "https://ia2-regresion-seguro-y-diabetes.onrender.com";
  ```

---

## 📎 Referencias (bases usadas)
- Kaggle – **Medical Insurance Cost with Linear Regression**
- Kaggle – **Diabetes Logistic Regression**
- Datasets: `insurance.csv` (Kaggle) y **PIMA Indians Diabetes** (Kaggle/UCI).

---

## BACKEND Y FRONTEND
- backend: https://ia2-regresion-seguro-y-diabetes.onrender.com
- frontend: https://ia-2-regresion-seguro-y-diabetes.vercel.app
