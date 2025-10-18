import json, joblib
from pathlib import Path
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, f1_score

DATA_INS = Path("data/insurance.csv")
DATA_DIA = Path("data/diabetes.csv")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True, parents=True)
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True, parents=True)

def train_insurance():
    df = pd.read_csv(DATA_INS)
    X = df.drop(columns=["charges"])
    y = df["charges"]
    num = ["age","bmi","children"]
    cat = ["sex","smoker","region"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

    ridge = Pipeline([("pre", pre), ("model", Ridge(random_state=42))])
    grid = GridSearchCV(ridge, {"model__alpha":[0.1,1,10]}, scoring="neg_mean_absolute_error", cv=5)
    grid.fit(X, y)
    best = grid.best_estimator_
    joblib.dump(best, MODELS/"insurance_model.pkl")

    rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=300, random_state=42))])
    rf.fit(X, y)
    joblib.dump(rf, MODELS/"rf_insurance.pkl")

    ohe = rf.named_steps["pre"].named_transformers_["cat"]
    feat_names = num + ohe.get_feature_names_out(cat).tolist()
    imps = rf.named_steps["model"].feature_importances_
    pd.DataFrame({"feature":feat_names,"importance":imps}).sort_values("importance", ascending=False).to_csv(REPORTS/"insurance_feature_importance.csv", index=False)

def best_threshold_f1(y_true, y_prob):
    ths = np.linspace(0.1, 0.9, 17)
    best_t, best_f1 = 0.5, -1
    for t in ths:
        f1 = f1_score(y_true, (y_prob>=t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)

def train_diabetes():
    df = pd.read_csv(DATA_DIA)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    pre = ColumnTransformer([("num", StandardScaler(), X.columns.tolist())])

    logit = Pipeline([("pre", pre),
                      ("model", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logit.fit(Xtr, ytr)
    prob = logit.predict_proba(Xte)[:,1]
    auc = float(roc_auc_score(yte, prob))
    thr, f1opt = best_threshold_f1(yte.values, prob)
    joblib.dump(logit, MODELS/"diabetes_model.pkl")
    (MODELS/"diabetes_threshold.json").write_text(json.dumps({"threshold": thr}, indent=2))

    rf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=300, random_state=42))])
    rf.fit(X, y)
    joblib.dump(rf, MODELS/"rf_diabetes.pkl")
    imps = rf.named_steps["model"].feature_importances_
    pd.DataFrame({"feature":X.columns, "importance":imps}).sort_values("importance", ascending=False).to_csv(REPORTS/"diabetes_feature_importance.csv", index=False)

if __name__ == "__main__":
    train_insurance()
    train_diabetes()
    print("Modelos y reportes guardados en /models y /reports")
