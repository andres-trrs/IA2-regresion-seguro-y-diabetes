from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib, json, pandas as pd

app = FastAPI(title="Insurance & Diabetes API")

MODELS = Path("models")
INS_MODEL_PATH = MODELS / "insurance_model.pkl"
DIA_MODEL_PATH = MODELS / "diabetes_model.pkl"
DIA_THR_PATH   = MODELS / "diabetes_threshold.json"

# carga de modelos
insurance_model = joblib.load(INS_MODEL_PATH)
diabetes_model  = joblib.load(DIA_MODEL_PATH)
with open(DIA_THR_PATH, "r") as f:
    diabetes_threshold = json.load(f)["threshold"]

# definicion datos de entrada
class InsuranceInput(BaseModel):
    age: int = Field(30, ge=0, le=120)
    bmi: float = Field(25.0, ge=10, le=70)
    children: int = Field(0, ge=0, le=10)
    sex: str = Field("male", pattern="^(male|female)$")
    smoker: str = Field("no", pattern="^(yes|no)$")
    region: str = Field("southwest", pattern="^(southwest|southeast|northwest|northeast)$")

class DiabetesInput(BaseModel):
    Pregnancies: int = Field(2, ge=0, le=20)
    Glucose: int = Field(130, ge=0, le=300)
    BloodPressure: int = Field(72, ge=0, le=200)
    SkinThickness: int = Field(20, ge=0, le=100)
    Insulin: int = Field(80, ge=0, le=900)
    BMI: float = Field(28.0, ge=10, le=70)
    DiabetesPedigreeFunction: float = Field(0.5, ge=0, le=3.0)
    Age: int = Field(45, ge=0, le=120)

@app.get("/")
def root():
    return {
        "msg": "OK",
        "docs": "/docs",
        "endpoints": ["/predict/insurance", "/predict/diabetes"],
        "threshold_diabetes": diabetes_threshold
    }

@app.post("/predict/insurance")
def predict_insurance(payload: InsuranceInput):
    X = pd.DataFrame([payload.dict()])
    yhat = float(insurance_model.predict(X)[0])
    return {"prediction": round(yhat, 2)}

@app.post("/predict/diabetes")
def predict_diabetes(payload: DiabetesInput):
    X = pd.DataFrame([payload.dict()])
    prob = float(diabetes_model.predict_proba(X)[0, 1])
    pred = int(prob >= diabetes_threshold)
    return {"probability": round(prob, 4), "threshold": diabetes_threshold, "prediction": pred}
