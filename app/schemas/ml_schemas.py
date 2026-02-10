from pydantic import BaseModel
from typing import List, Optional, Dict, Union

# --- Train ---
class TrainRequest(BaseModel):
    dataset_id: str
    model_type: str  # "logreg" ou "rf"

class TrainResponse(BaseModel):
    model_id: str
    metrics: Dict[str, float]
    features: List[str]

# --- Predict ---
class PredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Union[float, str]]] 

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]] = None

# --- Model info ---
class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    hyperparams: Dict
    dataset_id: str
    preprocessing: Dict
    date_trained: str

# --- Metrics ---
class MetricsResponse(BaseModel):
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None
