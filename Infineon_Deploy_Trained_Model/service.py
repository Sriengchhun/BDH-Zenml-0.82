from starlette.middleware.cors import CORSMiddleware
from bentoml.io import Image, NumpyNdarray
from constants import MODEL_NAME, SERVICE_NAME
from cgi import test
import json
import os
from unicodedata import category
from bentoml.io import JSON
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
# Header file
import bentoml

from pydantic import BaseModel
from typing import Optional
from typing import Dict, Any
from fastapi import FastAPI 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SensorData(BaseModel):
    accx: float
    accy: float
    accz: float
    gyrox: float
    gyroy: float
    gyroz: float


# Load model and build service
xgb_runner = bentoml.sklearn.get(MODEL_NAME).to_runner()
svc = bentoml.Service(name=SERVICE_NAME, runners=[xgb_runner])

# def loadModel():
model_loaded = bentoml.sklearn.load_model(MODEL_NAME)

async def predict_servity(data):
    className = {1: 'Circle', 2: 'Side-to-Side', 3: 'Square', 4: 'negative'}    
    X = np.array([[data.accx, data.accy, data.accz, data.gyrox, data.gyroy, data.gyroz]])
    
    # Make prediction using the BentoML model
    pred = model_loaded.predict(X)
    res = pred[0].astype(int)
    category = className[res]
    
    pred_proba = model_loaded.predict_proba(X)
    max_value_score = pred_proba.argmax(1).item()
    confidence = float(pred_proba[0, max_value_score])
    
    return category, confidence

# Variable for Input
user_input = JSON(
    pydantic_model=SensorData,
    validate_json=True,
)

@svc.api(
    input=user_input,
    output=JSON(),
    route='',
)

async def predict(data: SensorData) -> dict:
    try:
        category, confidence = await predict_servity(data)
        return {"category": category, "confidence": confidence}
        return {'results': result}
    except Exception as e:
        return {"error": str(e)}