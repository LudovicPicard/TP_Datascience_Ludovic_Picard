import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# 1. Generate Dataset
print("=== Generating Dataset ===")
resp = requests.post(f"{BASE_URL}/dataset/generate", json={
    "phase": "clean",
    "seed": 42,
    "n": 200
})
print(f"Status: {resp.status_code}")
if resp.status_code != 200:
    print(resp.text)
    exit()

dataset_info = resp.json()
dataset_id = dataset_info["dataset_id"]
print(f"Dataset ID: {dataset_id}")

# 2. Fit Cleaner
print("\n=== Fitting Cleaner ===")
resp = requests.post(f"{BASE_URL}/clean/fit", json={
    "dataset_id": dataset_id,
    "impute_strategy": "mean",
    "outlier_strategy": "clip",
    "categorical_strategy": "one_hot"
})
print(f"Status: {resp.status_code}")
try:
    print(f"Response JSON: {json.dumps(resp.json(), indent=2)}")
except:
    print(f"Response Text: {resp.text}")

try:
    fit_info = resp.json()
    if "cleaner_id" in fit_info:
        print(f"Success! Cleaner ID: {fit_info['cleaner_id']}")
    else:
        print("Failure! 'cleaner_id' not found in response.")
except Exception as e:
    print(f"Failed to parse JSON: {e}")
