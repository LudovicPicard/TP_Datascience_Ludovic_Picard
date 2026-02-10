import os, uuid, pickle, datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from app.dataset_manager import DATASETS

MODEL_DIR = "app/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Stockage local des modèles et infos
MODELS = {}

def train_model(dataset_id: str, model_type: str):
    if dataset_id not in DATASETS:
        raise ValueError("Dataset inconnu")
    df = DATASETS[dataset_id].copy()
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Colonnes numériques et catégorielles
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=object).columns.tolist()
    
    # Pipeline preprocessing
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
    ])
    
    # Choix du modèle
    if model_type == "logreg":
        model = LogisticRegression()
    elif model_type == "rf":
        model = RandomForestClassifier()
    else:
        raise ValueError("model_type doit être 'logreg' ou 'rf'")
    
    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("model", model)
    ])
    
    # Split train/valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # Prédictions
    y_pred = pipeline.predict(X_valid)
    if model_type == "logreg":
        y_proba = pipeline.predict_proba(X_valid)[:,1]
        auc = roc_auc_score(y_valid, y_proba)
    else:
        y_proba = None
        auc = None
    
    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred),
        "recall": recall_score(y_valid, y_pred),
        "f1": f1_score(y_valid, y_pred),
        "auc": auc
    }
    
    model_id = str(uuid.uuid4())
    MODELS[model_id] = {
        "pipeline": pipeline,
        "model_type": model_type,
        "dataset_id": dataset_id,
        "preprocessing": {"numeric": numeric_cols, "categorical": cat_cols},
        "date_trained": datetime.datetime.now().isoformat()
    }
    
    # Sérialisation
    with open(f"{MODEL_DIR}/{model_id}.pkl", "wb") as f:
        pickle.dump(MODELS[model_id], f)
    
    return model_id, metrics, numeric_cols + cat_cols

def get_model_info(model_id: str):
    if model_id not in MODELS:
        # tenter de charger depuis fichier
        path = f"{MODEL_DIR}/{model_id}.pkl"
        if not os.path.exists(path):
            raise ValueError("model_id inconnu")
        with open(path, "rb") as f:
            MODELS[model_id] = pickle.load(f)
    info = MODELS[model_id]
    return {
        "model_id": model_id,
        "model_type": info["model_type"],
        "hyperparams": info["pipeline"].get_params(),
        "dataset_id": info["dataset_id"],
        "preprocessing": info["preprocessing"],
        "date_trained": info["date_trained"]
    }

def predict_model(model_id: str, data):
    # Charger modèle si nécessaire
    if model_id not in MODELS:
        path = f"{MODEL_DIR}/{model_id}.pkl"
        if not os.path.exists(path):
            raise ValueError("model_id inconnu")
        with open(path, "rb") as f:
            MODELS[model_id] = pickle.load(f)
    
    pipeline = MODELS[model_id]["pipeline"]
    preprocessing = MODELS[model_id]["preprocessing"]
    
    # Convertir en DataFrame si nécessaire
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Assurer que toutes les colonnes numériques sont présentes
    for col in preprocessing["numeric"]:
        if col not in df.columns:
            df[col] = 0.0  # valeur par défaut
    
    # Assurer que toutes les colonnes catégorielles sont présentes et en str
    for col in preprocessing["categorical"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    
    # Prédiction
    preds = pipeline.predict(df)
    if MODELS[model_id]["model_type"] == "logreg":
        proba = pipeline.predict_proba(df)[:,1]
    else:
        proba = None
    
    return preds.tolist(), proba.tolist() if proba is not None else None
