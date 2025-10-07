from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Loan Default Predictor", version="1.0.0")

# ---------- Pydantic request/response models ----------
class PredictRequest(BaseModel):
    age: float = Field(..., description="Borrower age (years)")
    loan_tenure_months: float = Field(..., description="Loan tenure in months")
    number_of_open_accounts: float
    credit_utilization_ratio: float
    loan_to_income: float
    delinquency_ratio: float
    avg_dpd_per_delinquency: float
    residence_type_Owned: bool
    residence_type_Rented: bool
    loan_purpose_Education: bool
    loan_purpose_Home: bool
    loan_purpose_Personal: bool
    loan_type_Unsecured: bool

class PredictResponse(BaseModel):
    probability_score: float
    predicted_class: int

# ---------- Load the trained pipeline ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "model_data.joblib")
MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}") from e


# ---------- Utilities ----------
FEATURES = [
    "age", "loan_tenure_months", "number_of_open_accounts",
    "credit_utilization_ratio", "loan_to_income", "delinquency_ratio",
    "avg_dpd_per_delinquency", "residence_type_Owned",
    "residence_type_Rented", "loan_purpose_Education", "loan_purpose_Home",
    "loan_purpose_Personal", "loan_type_Unsecured"
]

def _row_from_request(req: PredictRequest) -> pd.DataFrame:
    """
    Build a single-row DataFrame from the request. Your pipeline handles
    scaling and column ordering, so we just pass the fields through.
    """
    d = req.model_dump()
    # Ensure all expected features are present; cast bools to native bool
    for k in FEATURES:
        if k not in d:
            raise ValueError(f"Missing required feature: {k}")
    # Create DataFrame with one row; FastAPI -> Pydantic already typed values
    df = pd.DataFrame([d], columns=FEATURES)
    return df


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        X = _row_from_request(payload)
        # pipe = [preprocess -> order_cols -> model]
        proba = pipe.predict_proba(X)[:, 1]
        pred = pipe.predict(X).astype(int)
        return PredictResponse(
            probability_score=round(float(proba[0]), 3),
            predicted_class=int(pred[0])
        )
    except Exception as e:
        # Bubble up a clean 400 for user/data issues, else 500
        msg = str(e)
        # simple heuristic: bad input -> 400
        if "Missing required feature" in msg or "could not convert" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=500, detail=f"Inference error: {msg}")

