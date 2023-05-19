from typing import Dict, List
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.Model import Model, getModel

class PredictionRequest(BaseModel):
    tweet: List

class PredictionResponse(BaseModel):
    res: Dict

repanalysisAPI = FastAPI()

repanalysisAPI.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

@repanalysisAPI.post('/predict', response_model = PredictionResponse)

def predict(request: PredictionRequest, model: Model = Depends(getModel)):
    res = model.predict(request.tweet)

    return PredictionResponse(res = res)