from typing import List, Optional
from pydantic import BaseModel

# PCA

class PCAFitTransformRequest(BaseModel):
    dataset_id: str
    n_components: int = 2  # entre 2 et 5
    scale: bool = True

class PCALoadings(BaseModel):
    variable: str
    contribution: float

class PCAFitTransformResponse(BaseModel):
    projection: List[dict]  # liste de dict par ligne {"PC1":..., "PC2":..., ...}
    explained_variance_ratio: List[float]
    loadings: List[PCALoadings]

# Clustering

class KMeansRequest(BaseModel):
    dataset_id: str
    k: int = 3  # 2 à 6
    scale: bool = True

class ClusterResponse(BaseModel):
    labels: List[int]
    centroids: List[dict]  # {"x1": ..., "x2": ...}
    silhouette: Optional[float]

# Report

class MVReportResponse(BaseModel):
    top_variables_PC1: List[str]
    top_variables_PC2: List[str]
    cluster_sizes: dict  # {"0": 10, "1": 15, ...}
