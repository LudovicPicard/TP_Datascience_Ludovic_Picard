from fastapi import APIRouter
from pydantic import BaseModel
from app.dataset_manager import generate_dataset

router = APIRouter()

class DatasetRequest(BaseModel):
    phase: str = "clean"
    seed: int = 42
    n: int = 100

@router.post("/dataset/generate")
def dataset_generate(request: DatasetRequest):
    return generate_dataset(request.phase, request.seed, request.n)
