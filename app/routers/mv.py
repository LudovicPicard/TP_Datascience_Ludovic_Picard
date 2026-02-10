from fastapi import APIRouter, HTTPException
from app.schemas.mv_schemas import PCAFitTransformRequest, PCAFitTransformResponse, KMeansRequest, ClusterResponse, MVReportResponse
from app.services.mv_service import fit_transform_pca, fit_kmeans, get_mv_report

router = APIRouter(prefix="/mv", tags=["Multivariate"])

# ==== PCA endpoint ====
@router.post("/pca/fit_transform", response_model=PCAFitTransformResponse)
def pca_fit_transform(request: PCAFitTransformRequest):
    try:
        return fit_transform_pca(request.dataset_id, request.n_components, request.scale)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==== KMeans endpoint ====
@router.post("/cluster/kmeans", response_model=ClusterResponse)
def cluster_kmeans(request: KMeansRequest):
    try:
        return fit_kmeans(request.dataset_id, request.k, request.scale)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==== Report endpoint ====
@router.get("/report/{dataset_id}", response_model=MVReportResponse)
def mv_report(dataset_id: str):
    try:
        # Les résultats sont récupérés depuis MV_RESULTS dans le service
        return get_mv_report(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
