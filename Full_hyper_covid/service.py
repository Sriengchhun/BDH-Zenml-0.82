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
class Data(BaseModel):
    gender: str
    age: Optional[int] = 0
    diseases: list
    bodyTemp: Optional[float] = 0
    preSpO2: Optional[float] = 0
    prePR: Optional[float] = 0
    preDyspnea: Optional[int] = 0
    fever: Optional[str] = 0
    cough: Optional[str] = 0
    runnyNose: Optional[str] = 0
    soreThroat: Optional[str] = 0
    smell: Optional[str] = 0
    diarrhea: Optional[str] = 0


def convertdata(data):
    if data is None:
        data = 0
    else:
        data = data
    return data

async def store_data(data):
    df = pd.DataFrame(columns=['gender', 'age', 'diseases',
                               'bodytemp', 'oxygen', 'pulse', 'dyspnea', 'cough', 'diarrhea',
                               'fever', 'runnyNose', 'smell', 'soreThroat'])
    new_index = len(df)

    df.loc[new_index] = {
        'gender': data.gender,
        'age': convertdata(data.age),
        'diseases': data.diseases,
        'bodytemp': convertdata(data.bodyTemp),
        'oxygen': convertdata(data.preSpO2),
        'pulse': convertdata(data.prePR),
        'dyspnea': convertdata(data.preDyspnea),
        'cough': convertdata(data.cough),
        'diarrhea': convertdata(data.diarrhea),
        'fever': convertdata(data.fever),
        'runnyNose': convertdata(data.runnyNose),
        'smell': convertdata(data.smell),
        'soreThroat': convertdata(data.soreThroat)
    }

    df_01 = df.copy()
    return df_01
    # Extract feature from disease feature for 1 report only


async def extract_feature(data):
    df = data.copy()
    df = df.dropna()
    # Convert str to list
    df["list_diseases"] = df["diseases"]
    df['HT_d'] = df['list_diseases'].apply(lambda x: 1 if "HT" in x else 0)
    df['BW_d'] = df['list_diseases'].apply(lambda x: 1 if "BW>90" in x else 0)
    df['DM_d'] = df['list_diseases'].apply(lambda x: 1 if "DM" in x else 0)
    df['DLP_d'] = df['list_diseases'].apply(lambda x: 1 if "DLP" in x else 0)
    df['HIV'] = df['list_diseases'].apply(lambda x: 1 if "HIV" in x else 0)
    df['hypothyroid'] = df['list_diseases'].apply(
        lambda x: 1 if "hypothyroid" in x else 0)
    df['PhumPer'] = df['list_diseases'].apply(
        lambda x: 1 if "ภูมิแพ้อากาศ" in x else 0)
    df = df.drop(['diseases', 'list_diseases'], axis=1)
    return df.copy()

    # Convert object type to float/int type


async def covert_obj(data):
    data = data.drop(data[(data['gender'] == 'other')].index)

    data = data.replace(to_replace=['none', 'stable', 'decrease',
                                    'increase', 'male', 'female', 'ชาย', 'หญิง', 'undefined'],
                        value=[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0, 0])
    return data.copy()

# Load model and build service
xgb_runner = bentoml.sklearn.get(MODEL_NAME).to_runner()
svc = bentoml.Service(name=SERVICE_NAME, runners=[xgb_runner])

# def loadModel():
xgb_model_loaded = bentoml.sklearn.load_model(MODEL_NAME)



async def predict_servity(data):
    className = {0: 'green', 1: 'yellow'}
    X = await store_data(data)
    X = await extract_feature(X)
    X = await covert_obj(X)
    pred = xgb_model_loaded.predict(X)
    res = pred[0].astype(int)
    catagory = className[res]
    pred_proba = xgb_model_loaded.predict_proba(X)
    Max_value_score = pred_proba.argmax(1).item()
    confidence = float(pred_proba[0, Max_value_score])
    return catagory, confidence

# Variable for Input
user_input = JSON(
    pydantic_model=Data,
    validate_json=True,
)

@svc.api(
    input=user_input,
    output=JSON(),
    route='',
)

async def predict(data: Data) -> json:
    try:
        catagory, confidence = await predict_servity(data)
        result = {'class': catagory, 'confidence': confidence}
        return {'results': result}
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        raise e
