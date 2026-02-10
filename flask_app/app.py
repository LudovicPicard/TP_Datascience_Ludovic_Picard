from flask import Flask, render_template, request, jsonify
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import os

app = Flask(__name__)

# FastAPI base URL (default to localhost for dev, but configurable via ENV)
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dataset/generate", methods=["POST"])
def generate_dataset():
    data = request.json
    print(f"Calling FastAPI: {BASE_URL}/dataset/generate with {data}")
    resp = requests.post(f"{BASE_URL}/dataset/generate", json=data)
    print(f"FastAPI Response ({resp.status_code}): {resp.text}")
    return jsonify(resp.json())

@app.route("/eda/summary", methods=["POST"])
def eda_summary():
    data = request.json
    resp = requests.post(f"{BASE_URL}/eda/summary", json=data)
    return jsonify(resp.json())

@app.route("/eda/plots", methods=["POST"])
def eda_plots():
    data = request.json
    resp = requests.post(f"{BASE_URL}/eda/plots", json=data)
    plots_data = resp.json().get("artifacts", {})
    
    # Generate some Plotly charts from raw data if needed
    # For now, we'll return the raw data and handle plotting in JS or here
    return jsonify(resp.json())

@app.route("/clean/fit", methods=["POST"])
def clean_fit():
    data = request.json
    resp = requests.post(f"{BASE_URL}/clean/fit", json=data)
    return jsonify(resp.json())

@app.route("/clean/transform", methods=["POST"])
def clean_transform():
    data = request.json
    resp = requests.post(f"{BASE_URL}/clean/transform", json=data)
    return jsonify(resp.json())

@app.route("/mv/pca", methods=["POST"])
def mv_pca():
    data = request.json
    resp = requests.post(f"{BASE_URL}/mv/pca/fit_transform", json=data)
    return jsonify(resp.json())

@app.route("/ml/train", methods=["POST"])
def ml_train():
    data = request.json
    resp = requests.post(f"{BASE_URL}/ml/train", json=data)
    return jsonify(resp.json())

@app.route("/ml2/tune", methods=["POST"])
def ml2_tune():
    params = request.json
    # FastAPI expects TuneRequest schema
    payload = {
        "dataset_id": params.get("dataset_id"),
        "model_type": "rf",
        "search": "grid",
        "cv": 3
    }
    resp = requests.post(f"{BASE_URL}/ml2/tune", json=payload)
    return jsonify(resp.json())

@app.route("/ml2/explain", methods=["POST"])
def ml2_explain():
    data = request.json
    resp = requests.post(f"{BASE_URL}/ml2/explain-instance", json=data)
    return jsonify(resp.json())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
