from fastapi import APIRouter
from app.schemas.eda_schemas import EDARequest, EDAResponse
from app.services import eda_service

router = APIRouter()

@router.post("/eda/summary", response_model=EDAResponse)
def eda_summary(request: EDARequest):
    result = eda_service.summary_statistics(request.meta.dataset_id)
    return {
        "meta": request.meta.dict(),
        "result": result,
        "report": "Résumé statistiques descriptives",
        "artifacts": {}
    }

@router.post("/eda/groupby", response_model=EDAResponse)
def eda_groupby(request: EDARequest):
    by = request.params.get("by")
    metrics = request.params.get("metrics", ["mean","median"])
    result = eda_service.groupby_statistics(request.meta.dataset_id, by, metrics)
    return {
        "meta": request.meta.dict(),
        "result": result,
        "report": f"Grouper par {by}",
        "artifacts": {}
    }

@router.post("/eda/correlation", response_model=EDAResponse)
def eda_correlation(request: EDARequest):
    result = eda_service.correlation_matrix(request.meta.dataset_id)
    return {
        "meta": request.meta.dict(),
        "result": result,
        "report": "Matrice de corrélation et top 5 corrélations",
        "artifacts": {}
    }

@router.post("/eda/plots", response_model=EDAResponse)
def eda_plots(request: EDARequest):
    artifacts = eda_service.generate_plots(request.meta.dataset_id)
    return {
        "meta": request.meta.dict(),
        "result": {},
        "report": "Artefacts pour plots",
        "artifacts": artifacts
    }
