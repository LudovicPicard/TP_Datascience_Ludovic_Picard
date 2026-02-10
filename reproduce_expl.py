import requests
import json
import pandas as pd
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

print("=== Generating ML Dataset ===")
resp = requests.post(f"{BASE_URL}/dataset/generate", json={
    "phase": "ml",
    "seed": 42,
    "n": 200
})
dataset_id = resp.json()["dataset_id"]
print(f"Dataset ID: {dataset_id}")

print("\n=== Tuning Model ===")
resp = requests.post(f"{BASE_URL}/ml2/tune", json={
    "dataset_id": dataset_id,
    "model_type": "rf",
    "search": "grid",
    "cv": 3
})
tune_info = resp.json()
best_model_id = tune_info["best_model_id"]
print(f"Model ID: {best_model_id}")

print("\n=== Explaining Instance ===")
# Create a sample instance (mixed types: float and string)
instance = {
    "x1": 0.5,
    "x2": 1.2,
    "x3": -0.8,
    "x4": 0.1,
    "x5": 0.9,
    "x6": -1.5,
    "segment": "A"  # String value
}

print(f"Sending instance: {instance}")
resp = requests.post(f"{BASE_URL}/ml2/explain-instance", json={
    "model_id": best_model_id,
    "data": instance
})

print(f"Status: {resp.status_code}")
try:
    print(f"Response JSON: {json.dumps(resp.json(), indent=2)}")
except:
    print(f"Response Text: {resp.text}")

if resp.status_code == 200:
    explanation = resp.json()
    try:
        df_expl = pd.DataFrame(explanation).sort_values("contribution")
        print("\nSuccess! DataFrame created:")
        print(df_expl)
        
        # Check for non-zero contribution
        max_contrib = df_expl["contribution"].abs().max()
        if max_contrib > 0:
            print(f"\nSUCCESS: Max contribution is {max_contrib} (> 0)")
        else:
            print("\nFAILURE: All contributions are zero!")
            
    except Exception as e:
        print(f"\nPandas Error: {e}")
