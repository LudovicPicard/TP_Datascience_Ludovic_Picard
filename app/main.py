from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import des routers TP1/TP2
from app.routers import dataset, clean, eda
# Import TP3 – Multivariée
from app.routers import mv
# Import TP4 – ML
from app.routers import ml
# Import TP5 – ML Avancé / tuning & explicabilité
from app.routers import ml2

app = FastAPI(
    title="DataScience API - TP1/TP2/TP3/TP4/TP5",
    description="API pour le TP1 (dataset/clean), TP2 (EDA), TP3 (PCA & Clustering), TP4 (ML) et TP5 (ML avancé) - Fixed Persistence",
    version="1.1.0"
)

# === Middleware CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Routers ===
app.include_router(dataset.router)   # TP1 dataset
app.include_router(clean.router)     # TP1 clean
app.include_router(eda.router)       # TP2 EDA
app.include_router(mv.router)        # TP3 Multivariée
app.include_router(ml.router)        # TP4 ML
app.include_router(ml2.router)       # TP5 ML avancé

# === Racine ===
@app.get("/")
def root():
    return {
        "message": "Bienvenue sur la DataScience API !",
        "endpoints": {
            "TP1": ["/dataset/generate", "/clean/..."],
            "TP2": ["/eda/..."],
            "TP3": ["/mv/pca/fit_transform", "/mv/cluster/kmeans", "/mv/report/{dataset_id}"],
            "TP4": ["/ml/train", "/ml/predict", "/ml/model-info/{model_id}"],
            "TP5": [
                "/ml2/tune",
                "/ml2/feature-importance/{model_id}",
                "/ml2/permutation-importance",
                "/ml2/explain-instance"
            ]
        }
    }
