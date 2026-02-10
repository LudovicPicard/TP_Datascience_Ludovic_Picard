from pydantic import BaseModel
from typing import List, Dict

class MetaInfoEDA(BaseModel):
    dataset_id: str

class EDARequest(BaseModel):
    meta: MetaInfoEDA
    params: Dict = {}  # paramètre libre pour groupby, plots, etc.

class EDAResponse(BaseModel):
    meta: dict
    result: dict
    report: str
    artifacts: dict
