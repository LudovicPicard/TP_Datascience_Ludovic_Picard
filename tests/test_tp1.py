import requests
import pprint

BASE_URL = "http://127.0.0.1:8000"  # l'adresse où tourne ton FastAPI

pp = pprint.PrettyPrinter(indent=2)

# 1️⃣ Générer un dataset "clean"
print("=== Génération du dataset ===")
resp = requests.post(f"{BASE_URL}/dataset/generate", json={
    "phase": "clean",
    "seed": 42,
    "n": 100
})
dataset_info = resp.json()
pp.pprint(dataset_info)

dataset_id = dataset_info["dataset_id"]

# 2️⃣ Fit du cleaner
print("\n=== Fit du cleaner ===")
resp = requests.post(f"{BASE_URL}/clean/fit", json={
    "dataset_id": dataset_id,
    "impute_strategy": "mean",
    "outlier_strategy": "clip",
    "categorical_strategy": "one_hot"
})
fit_info = resp.json()
pp.pprint(fit_info)

cleaner_id = fit_info["cleaner_id"]

# 3️⃣ Transformer le dataset
print("\n=== Transformation du dataset ===")
resp = requests.post(f"{BASE_URL}/clean/transform", json={
    "dataset_id": dataset_id,
    "cleaner_id": cleaner_id
})
transform_info = resp.json()
pp.pprint(transform_info)

# 4️⃣ Récupérer le rapport
print("\n=== Rapport qualité avant transformation ===")
resp = requests.get(f"{BASE_URL}/clean/report/{dataset_id}")
report_info = resp.json()
pp.pprint(report_info)
