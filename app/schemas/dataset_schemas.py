from pydantic import BaseModel

class DatasetRequest(BaseModel):
    phase: str
    seed: int
    n: int
