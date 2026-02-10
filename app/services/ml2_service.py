import os, uuid, pickle, datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from app.dataset_manager import DATASETS

MODEL_DIR = "app/models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODELS = {}

# -------------------- Utils --------------------
def _build_pipeline(model_type: str, numeric_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
    ])
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier()
    pipeline = Pipeline([("preproc", preprocessor), ("model", model)])
    return pipeline

def _save_model(pipeline, model_type, dataset_id, numeric_cols, cat_cols):
    model_id = str(uuid.uuid4())
    MODELS[model_id] = {
        "pipeline": pipeline,
        "model_type": model_type,
        "dataset_id": dataset_id,
        "preprocessing": {"numeric": numeric_cols, "categorical": cat_cols},
        "date_trained": datetime.datetime.now().isoformat()
    }
    with open(f"{MODEL_DIR}/{model_id}.pkl", "wb") as f:
        pickle.dump(MODELS[model_id], f)
    return model_id

def _load_dataset(dataset_id):
    if dataset_id not in DATASETS:
        raise ValueError("Dataset inconnu")
    df = DATASETS[dataset_id].copy()
    X = df.drop(columns=["target"])
    y = df["target"]
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=object).columns.tolist()
    return X, y, numeric_cols, cat_cols

# -------------------- TP5 Functions --------------------
def tune_model(dataset_id: str, model_type: str, search: str = "grid", cv: int = 3):
    X, y, numeric_cols, cat_cols = _load_dataset(dataset_id)
    pipeline = _build_pipeline(model_type, numeric_cols, cat_cols)
    
    # Hyperparameter grid
    if model_type == "logreg":
        param_grid = {"model__C": [0.01, 0.1, 1, 10]}
    else:
        param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5]
        }
    
    if search == "grid":
        searcher = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1)
    else:
        searcher = RandomizedSearchCV(pipeline, param_grid, cv=cv, n_iter=5, n_jobs=-1, random_state=42)
    
    searcher.fit(X, y)
    best_pipeline = searcher.best_estimator_
    model_id = _save_model(best_pipeline, model_type, dataset_id, numeric_cols, cat_cols)
    
    # Top 5 configs
    if hasattr(searcher, "cv_results_"):
        df_results = pd.DataFrame(searcher.cv_results_)
        top5 = df_results.sort_values("rank_test_score").head(5)
        cv_results_summary = top5[["params", "mean_test_score", "rank_test_score"]].to_dict(orient="records")
    else:
        cv_results_summary = []
    
    return model_id, searcher.best_params_, cv_results_summary

def get_feature_importance(model_id: str):
    if model_id not in MODELS:
        path = f"{MODEL_DIR}/{model_id}.pkl"
        if not os.path.exists(path):
            raise ValueError("model_id inconnu")
        with open(path, "rb") as f:
            MODELS[model_id] = pickle.load(f)
    pipeline = MODELS[model_id]["pipeline"]
    numeric = MODELS[model_id]["preprocessing"]["numeric"]
    categorical = MODELS[model_id]["preprocessing"]["categorical"]
    feature_names = numeric + list(pipeline.named_steps["preproc"].named_transformers_["cat"].get_feature_names_out(categorical))
    
    model_type = MODELS[model_id]["model_type"]
    if model_type == "rf":
        importances = pipeline.named_steps["model"].feature_importances_
    else:
        importances = pipeline.named_steps["model"].coef_[0]
        # Standardiser les coefficients
        importances = importances / np.std(importances)
    
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(10).to_dict(orient="records")

def permutation_importance_model(model_id: str, n_repeats: int = 5):
    if model_id not in MODELS:
        path = f"{MODEL_DIR}/{model_id}.pkl"
        if not os.path.exists(path):
            raise ValueError("model_id inconnu")
        with open(path, "rb") as f:
            MODELS[model_id] = pickle.load(f)
    pipeline = MODELS[model_id]["pipeline"]
    dataset_id = MODELS[model_id]["dataset_id"]
    X, y, _, _ = _load_dataset(dataset_id)
    
    result = permutation_importance(pipeline, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    df = pd.DataFrame({"feature": X.columns, "importance": result.importances_mean})
    return df.sort_values("importance", ascending=False).to_dict(orient="records")

def explain_instance(model_id: str, instance: dict):
    if model_id not in MODELS:
        path = f"{MODEL_DIR}/{model_id}.pkl"
        if not os.path.exists(path):
            raise ValueError("model_id inconnu")
        with open(path, "rb") as f:
            MODELS[model_id] = pickle.load(f)
    pipeline = MODELS[model_id]["pipeline"]
    preprocessing = MODELS[model_id]["preprocessing"]
    
    # Convertir instance en DataFrame
    df = pd.DataFrame([instance])
    for col in preprocessing["numeric"]:
        if col not in df.columns:
            df[col] = 0.0
    for col in preprocessing["categorical"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)
    
    model_type = MODELS[model_id]["model_type"]
    if model_type == "logreg":
        # contribution ≈ coef × valeur
        coefs = pipeline.named_steps["model"].coef_[0]
        feature_names = preprocessing["numeric"] + list(pipeline.named_steps["preproc"].named_transformers_["cat"].get_feature_names_out(preprocessing["categorical"]))
        values = df[preprocessing["numeric"] + preprocessing["categorical"]].values[0]
        contrib = dict(zip(feature_names, coefs * values))
    else:
        # approximation RF via perturbation (Mean/Mode Imputation)
        dataset_id = MODELS[model_id]["dataset_id"]
        X_train, y_train, _, _ = _load_dataset(dataset_id)
        
        # Calculer baseline (moyenne/mode)
        baseline = {}
        for col in preprocessing["numeric"]:
            baseline[col] = X_train[col].mean()
        for col in preprocessing["categorical"]:
            baseline[col] = X_train[col].mode()[0]
            
        base_pred = pipeline.predict_proba(df)[0, 1]  # Probabilité de classe 1 (plus informel pour RF)
        
        contrib = {}
        for col in df.columns:
            tmp = df.copy()
            # Remplacer la valeur par la baseline
            if col in baseline:
                tmp[col] = baseline[col]
            
            # Nouvelle prédiction
            new_pred = pipeline.predict_proba(tmp)[0, 1]
            
            # Contribution = Différence (combien la valeur actuelle éloigne de la moyenne)
            contrib[col] = base_pred - new_pred

    return contrib
