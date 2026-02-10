import pandas as pd
from app.dataset_manager import DATASETS
import numpy as np

def summary_statistics(dataset_id: str):
    df = DATASETS[dataset_id]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].describe().to_dict()
    missing_rate = df.isna().mean().to_dict()
    return {"summary": summary, "missing_rate": missing_rate}

def groupby_statistics(dataset_id: str, by: str, metrics: list):
    df = DATASETS[dataset_id]
    if by not in df.columns:
        return {"error": f"Column {by} not in dataset"}
    grouped = df.groupby(by).agg(metrics)
    return {"grouped": grouped.reset_index().to_dict(orient="records")}

def correlation_matrix(dataset_id: str):
    df = DATASETS[dataset_id]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr().to_dict()
    # Top 5 paires corrélées
    corr_df = df[numeric_cols].corr().abs()
    top_pairs = []
    for i, col1 in enumerate(corr_df.columns):
        for j, col2 in enumerate(corr_df.columns):
            if i < j:
                top_pairs.append((col1, col2, corr_df.iloc[i, j]))
    top_pairs = sorted(top_pairs, key=lambda x: x[2], reverse=True)[:5]
    return {"correlation_matrix": corr, "top_pairs": top_pairs}

def generate_plots(dataset_id: str):
    # Placeholder simple: renvoyer les données pour plots
    df = DATASETS[dataset_id]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    artifacts = {
        "histograms": {col: df[col].dropna().tolist() for col in numeric_cols},
        "boxplots": {col: df[col].dropna().tolist() for col in numeric_cols},
        "barplots": {col: df[col].value_counts().to_dict() for col in categorical_cols}
    }
    return artifacts
