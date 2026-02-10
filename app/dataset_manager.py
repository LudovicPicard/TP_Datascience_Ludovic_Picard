import pandas as pd
import numpy as np
import uuid
import os
import pickle

# --- Stockage Persistant ---
class PersistentStore(dict):
    def __init__(self, key_name: str):
        super().__init__()
        self.storage_dir = os.path.join(os.getcwd(), "data", key_name)
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_path(self, key):
        return os.path.join(self.storage_dir, f"{key}.pkl")

    def __getitem__(self, key):
        if key not in super().keys():
            filepath = self._get_path(key)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    val = pickle.load(f)
                super().__setitem__(key, val)
            else:
                raise KeyError(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        filepath = self._get_path(key)
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
            
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
            
    def __contains__(self, key):
        return super().__contains__(key) or os.path.exists(self._get_path(key))

# Stockage global pour les datasets (persistant)
DATASETS = PersistentStore("datasets")
CLEANERS = PersistentStore("cleaners")
MV_RESULTS = PersistentStore("mv_results")

def generate_dataset(phase: str = "clean", seed: int = 42, n: int = 100):
    np.random.seed(seed)

    if phase == "clean":
        # Colonnes
        df = pd.DataFrame({
            "x1": np.random.randn(n) * 10,
            "x2": np.random.randn(n) * 5,
            "x3": np.random.randn(n),
            "segment": np.random.choice(["A", "B", "C"], size=n)
        })

        # Cible optionnelle
        if np.random.rand() > 0.5:
            df["target"] = np.random.choice([0, 1], size=n)

        # Introduire NA (~10-20%)
        for col in ["x1", "x2", "x3"]:
            na_indices = np.random.choice(df.index, size=int(n * 0.15), replace=False)
            df.loc[na_indices, col] = np.nan

        # Introduire doublons (~1-5%)
        n_dups = int(n * 0.03)
        dup_indices = np.random.choice(df.index, size=n_dups, replace=False)
        df = pd.concat([df, df.loc[dup_indices]], ignore_index=True)

        # Introduire outliers (~1-3%)
        for col in ["x1", "x2", "x3"]:
            n_out = int(n * 0.02)
            out_indices = np.random.choice(df.index, size=n_out, replace=False)
            df.loc[out_indices, col] = df[col].mean() + 10 * df[col].std()

        # Introduire types cassés dans x2
        n_broken = int(n * 0.02)
        broken_indices = np.random.choice(df.index, size=n_broken, replace=False)
        df["x2"] = df["x2"].astype(object)
        df.loc[broken_indices, "x2"] = "oops"

    elif phase == "mv":
        # Multivarié : 8 variables numériques + 3 clusters + colinéarité
        cluster_centers = np.array([
            [2, 2, 2, 2, 4, 4, 4, 4],
            [-2, -2, -2, -2, -4, -4, -4, -4],
            [5, -5, 5, -5, 10, -10, 10, -10]
        ])
        clusters = np.random.choice([0,1,2], size=n)
        data = []
        for i in range(n):
            center = cluster_centers[clusters[i]]
            # ajouter bruit
            point = center + np.random.randn(8) * 0.5
            data.append(point)
        df = pd.DataFrame(data, columns=[f"x{i+1}" for i in range(8)])
        df["segment"] = clusters.astype(str)

        # Introduire NA faible
        for col in df.columns[:8]:
            na_indices = np.random.choice(df.index, size=int(n * 0.03), replace=False)
            df.loc[na_indices, col] = np.nan

    elif phase == "ml":
        # Supervised ML : x1..x6 + segment + target binaire (déséquilibré 70/30)
        df = pd.DataFrame({f"x{i+1}": np.random.randn(n) for i in range(6)})
        df["segment"] = np.random.choice(["A","B","C"], size=n)
        df["target"] = np.random.choice([0,1], size=n, p=[0.7,0.3])

    else:
        raise NotImplementedError(f"Phase {phase} non supportée")

    # Nettoyer NaN/inf avant retour JSON
    df = df.replace([np.inf, -np.inf], np.nan)

    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df

    return {
        "dataset_id": dataset_id,
        "columns": df.columns.tolist(),
        "data_sample": df.head(20).replace({np.nan: None}).to_dict(orient="records")
    }

def get_dataset(dataset_id: str):
    df = DATASETS.get(dataset_id)
    if df is None:
        raise ValueError(f"Dataset {dataset_id} inconnu")
    return df
