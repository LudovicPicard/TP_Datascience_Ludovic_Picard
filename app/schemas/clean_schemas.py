from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class CleanRequest(BaseModel):
    dataset_id: str
    impute_strategy: str = "mean"
    outlier_strategy: str = "clip"
    categorical_strategy: str = "one_hot"

class TransformRequest(BaseModel):
    cleaner_id: str
    dataset_id: str

class CleanReportResponse(BaseModel):
    missing_values: Dict[str, int]
    duplicates: int
    outliers: Dict[str, int]
    types: Dict[str, str]

class TransformResponse(BaseModel):
    dataset_id: str
    data: List[Dict[str, Any]]
    duplicates_removed: int
    imputed: int

class FitResponse(BaseModel):
    cleaner_id: str
    report: CleanReportResponse
