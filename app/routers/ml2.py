from fastapi import APIRouter
from typing import List  # <-- il manquait ça
from app.schemas.ml2_schemas import (
    TuneRequest, TuneResponse,
    FeatureImportanceResponse,
    PermutationImportanceRequest, PermutationImportanceResponse,
    ExplainInstanceRequest, ExplainInstanceResponse
)
from app.services.ml2_service import (
    tune_model, get_feature_importance,
    permutation_importance_model, explain_instance
)
import pandas as pd

router = APIRouter(prefix="/ml2", tags=["ML Avancé"])

# --- Tuning ---
@router.post("/tune", response_model=TuneResponse)
def tune(request: TuneRequest):
    model_id, best_params, cv_summary = tune_model(
        request.dataset_id,
        request.model_type,
        request.search,
        request.cv
    )
    return {"best_model_id": model_id, "best_params": best_params, "cv_results_summary": cv_summary}

# --- Feature importance ---
@router.get("/feature-importance/{model_id}", response_model=List[FeatureImportanceResponse])
def feature_importance(model_id: str):
    feat_imp = get_feature_importance(model_id)
    # feat_imp est déjà une liste de dicts {"feature": ..., "importance": ...}
    return feat_imp

# --- Permutation importance ---
@router.post("/permutation-importance", response_model=List[PermutationImportanceResponse])
def permutation_importance(request: PermutationImportanceRequest):
    feat_imp = permutation_importance_model(request.model_id, request.n_repeats)
    return feat_imp

# --- Explain instance ---
@router.post("/explain-instance", response_model=List[ExplainInstanceResponse])
def explain_instance_endpoint(request: ExplainInstanceRequest):
    # convertit le dict en dict simple pour le service
    feat_imp = explain_instance(request.model_id, request.data)
    return [{"feature": f, "contribution": val} for f, val in feat_imp.items()]
