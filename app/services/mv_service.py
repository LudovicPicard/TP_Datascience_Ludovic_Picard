import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from app.dataset_manager import DATASETS, MV_RESULTS  # réutilisation TP2

# ==== PCA ====

def fit_transform_pca(dataset_id: str, n_components: int, scale: bool):
    df = DATASETS[dataset_id].copy()
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(X_scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    projection_records = [
        {f"PC{i+1}": float(proj[j, i]) for i in range(n_components)}
        for j in range(len(df))
    ]

    loadings_records = [
        {"variable": numeric_cols[i], "contribution": float(loadings[i, j])}
        for j in range(n_components) for i in range(len(numeric_cols))
    ]

    result = {
        "projection": projection_records,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "loadings": loadings_records
    }
    
    # Persistance
    if dataset_id not in MV_RESULTS:
        MV_RESULTS[dataset_id] = {}
    
    # Note: PersistentStore (dict) ne détecte pas les modifs internes d'un dict mutables
    # Il faut réassigner tout l'objet pour déclencher __setitem__ et le pickle.dump
    current_res = MV_RESULTS[dataset_id]
    current_res["pca_results"] = result
    MV_RESULTS[dataset_id] = current_res

    return result

# ==== KMeans ====

def fit_kmeans(dataset_id: str, k: int, scale: bool):
    df = DATASETS[dataset_id].copy()
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    X = df[numeric_cols].fillna(df[numeric_cols].mean())

    if scale:
        X = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    try:
        sil_score = float(silhouette_score(X, labels))
    except:
        sil_score = None

    centroids = [
        {numeric_cols[i]: float(kmeans.cluster_centers_[j, i]) for i in range(len(numeric_cols))}
        for j in range(k)
    ]

    result = {
        "labels": labels.tolist(),
        "centroids": centroids,
        "silhouette": sil_score
    }
    
    # Persistance
    if dataset_id not in MV_RESULTS:
        MV_RESULTS[dataset_id] = {}
        
    current_res = MV_RESULTS[dataset_id]
    current_res["cluster_results"] = result
    MV_RESULTS[dataset_id] = current_res

    return result

# ==== Rapport ====

def get_mv_report(dataset_id: str, top_n=3):
    # Récupération depuis MV_RESULTS
    mv_data = MV_RESULTS.get(dataset_id)
    if not mv_data:
        raise ValueError("Aucune analyse multivariée trouvée pour ce dataset")
        
    pca_results = mv_data.get("pca_results")
    cluster_results = mv_data.get("cluster_results")

    if pca_results is None or cluster_results is None:
        raise ValueError("PCA et Clustering doivent être calculés avant le rapport")

    loadings_df = pd.DataFrame(pca_results["loadings"])
    top_pc1 = loadings_df[loadings_df["variable"].notnull()].sort_values(
        by="contribution", ascending=False
    ).groupby("variable").first().sort_values("contribution", ascending=False).head(top_n).index.tolist()

    top_pc2 = loadings_df[loadings_df["variable"].notnull()].sort_values(
        by="contribution", ascending=False
    ).groupby("variable").nth(1).dropna().sort_values("contribution", ascending=False).head(top_n).index.tolist()

    cluster_sizes = pd.Series(cluster_results["labels"]).value_counts().to_dict()

    return {
        "top_variables_PC1": top_pc1,
        "top_variables_PC2": top_pc2,
        "cluster_sizes": cluster_sizes
    }
