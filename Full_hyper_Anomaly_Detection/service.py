from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

import numpy as np
import bentoml
from bentoml.io import JSON
from fastapi import FastAPI
import os
# from constants import MODEL_NAME, SERVICE_NAME

# ---------- FastAPI (for OpenAPI/Swagger) ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SensorData(BaseModel):
    temperature: Optional[float] = 48.35
    humidity: Optional[float] = 36.10
    lux: Optional[float] = 0.14
    soil: Optional[float] = 82.20

MODEL_NAME = os.getenv("MODEL_NAME_Anomaly")
SERVICE_NAME = f"{MODEL_NAME}_Service"

# ---------- Bento: load model & runner ----------
# Runner is recommended for concurrency and scaling
if_runner = bentoml.sklearn.get(MODEL_NAME).to_runner()
svc = bentoml.Service(name=SERVICE_NAME, runners=[if_runner])

# Also load the model object to access decision_function for scoring detail
if_model = bentoml.sklearn.load_model(MODEL_NAME)

# ---------- helpers ----------
def _to_numpy_row(data: SensorData) -> np.ndarray:
    # shape (1, 4)
    return np.array([[data.temperature, data.humidity, data.lux, data.soil]], dtype=float)

def _confidence_from_score(decision_score: float) -> float:
    """
    IsolationForest decision_function:
      > 0 for inliers (normal), < 0 for outliers (anomaly).
    We map to a [0,1] "anomaly confidence" using a simple sigmoid on the negated score.
    """
    # Higher -> more anomalous
    anomaly_score = -decision_score
    # Sigmoid; you can tune the slope (k) if needed
    k = 1.0
    conf = 1.0 / (1.0 + np.exp(-k * anomaly_score))
    return float(conf)

async def _predict_isolation_forest(data: SensorData) -> Dict[str, Any]:
    X = _to_numpy_row(data)

    # y_pred: +1 = normal, -1 = anomaly
    # Use runner for prediction (async-friendly)
    y_pred = await if_runner.predict.async_run(X)
    y_pred = int(y_pred[0])

    # Raw decision score via model object
    # > 0 normal, < 0 anomaly
    decision_val = float(if_model.decision_function(X)[0])

    is_anomaly = 1 if y_pred == -1 else 0
    label = "anomaly" if is_anomaly == 1 else "normal"
    confidence = _confidence_from_score(decision_val)

    return {
        "label": label,
        "anomaly": is_anomaly,
        "confidence": confidence,     # ~probability that this is an anomaly
        "score": decision_val,        # raw decision_function (debug/thresholding)
    }

# ---------- Bento API ----------
user_input = JSON(pydantic_model=SensorData, validate_json=True)

@svc.api(input=user_input, output=JSON(), route="/")
async def predict(payload: SensorData) -> Dict[str, Any]:
    """
    Predict anomaly for a single sensor record using IsolationForest.
    Returns label ('anomaly'|'normal'), anomaly (1|0), confidence in [0,1], and raw score.
    """
    try:
        return await _predict_isolation_forest(payload)
    except Exception as e:
        # Return a structured error response
        return {"error": str(e)}

