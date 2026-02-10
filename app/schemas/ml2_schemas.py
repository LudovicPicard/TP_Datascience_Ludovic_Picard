from pydantic import BaseModel
from typing import List, Optional, Dict, Union

# --- Tune ---
class TuneRequest(BaseModel):
    dataset_id: str
    model_type: str  # "logreg" ou "rf"
    search: str = "grid"
    cv: int = 3

class TuneResponse(BaseModel):
    best_model_id: str
    best_params: Dict
    cv_results_summary: List[Dict]

# --- Feature importance ---
class FeatureImportanceResponse(BaseModel):
    feature: str
    importance: float

# --- Permutation importance ---
class PermutationImportanceRequest(BaseModel):
    model_id: str
    n_repeats: int = 5

class PermutationImportanceResponse(BaseModel):
    feature: str
    importance: float

# --- Explain instance ---
class ExplainInstanceRequest(BaseModel):
    model_id: str
    data: Dict[str, Union[str, float, int]]  # peut contenir des strings (catégories)

class ExplainInstanceResponse(BaseModel):
    feature: str
    contribution: float
