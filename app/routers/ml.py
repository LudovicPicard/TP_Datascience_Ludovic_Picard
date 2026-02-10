from fastapi import APIRouter
from app.schemas.ml_schemas import TrainRequest, TrainResponse, PredictRequest, PredictResponse, ModelInfoResponse, MetricsResponse
from app.services.ml_service import train_model, get_model_info, predict_model

router = APIRouter(prefix="/ml", tags=["ML Baseline"])

@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest):
    model_id, metrics, features = train_model(request.dataset_id, request.model_type)
    return {"model_id": model_id, "metrics": metrics, "features": features}

@router.get("/model-info/{model_id}", response_model=ModelInfoResponse)
def model_info(model_id: str):
    return get_model_info(model_id)

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    import pandas as pd
    data = pd.DataFrame(request.data)
    preds, probas = predict_model(request.model_id, data)
    return {"predictions": preds, "probabilities": probas}

@router.get("/metrics/{model_id}", response_model=MetricsResponse)
def metrics(model_id: str):
    info = get_model_info(model_id)
    metrics = info["pipeline"].named_steps["model"].__dict__.get("metrics", {})
    # Pour simplifier, utiliser MLService pour recalculer
    from app.services.ml_service import MODELS
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    model_data = MODELS[model_id]
    dataset_id = model_data["dataset_id"]
    from app.dataset_manager import DATASETS
    df = DATASETS[dataset_id]
    X = df.drop(columns=["target"])
    y = df["target"]
    preds = model_data["pipeline"].predict(X)
    if model_data["model_type"] == "logreg":
        probas = model_data["pipeline"].predict_proba(X)[:,1]
        auc = roc_auc_score(y, probas)
    else:
        auc = None
    return {
        "model_id": model_id,
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "auc": auc
    }
