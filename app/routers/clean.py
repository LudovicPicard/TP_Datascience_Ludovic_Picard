from fastapi import APIRouter, HTTPException
from app.services.clean_service import fit_cleaner, transform_dataset, get_report
from app.schemas.clean_schemas import CleanRequest, TransformRequest, FitResponse, TransformResponse, CleanReportResponse

router = APIRouter(prefix="/clean", tags=["clean"])

@router.post("/fit", response_model=FitResponse)
def fit_clean(request: CleanRequest):
    try:
        cleaner_id, report = fit_cleaner(
            dataset_id=request.dataset_id,
            impute_strategy=request.impute_strategy,
            outlier_strategy=request.outlier_strategy,
            categorical_strategy=request.categorical_strategy
        )
        return FitResponse(cleaner_id=cleaner_id, report=CleanReportResponse(**report))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@router.post("/transform", response_model=TransformResponse)
def transform_data(request: TransformRequest):
    try:
        result = transform_dataset(
            cleaner_id=request.cleaner_id,
            dataset_id=request.dataset_id
        )
        return TransformResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@router.get("/report/{dataset_id}", response_model=CleanReportResponse)
def report(dataset_id: str):
    try:
        report_data = get_report(dataset_id)
        return CleanReportResponse(**report_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")
