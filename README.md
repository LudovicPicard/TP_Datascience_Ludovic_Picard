# 📊 DataScience Dashboard - Projet Picard

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

Ce projet est une plateforme complète d'analyse de données et de Machine Learning, composée d'un backend performant en **FastAPI** et d'un dashboard interactif moderne en **Flask**.

## 🚀 Fonctionnalités du Dashboard

Le projet est organisé en 5 Travaux Pratiques (TPs) couvrant tout le cycle de vie de la donnée :

- **🏠 Accueil : Génération de Données**
  - Génération de datasets personnalisés selon les phases (Cleaning, MV, ML).
  - Gestion de la persistance des datasets générés.

- **📈 TP1 : Analyse Exploratoire (EDA)**
  - Statistiques descriptives complètes.
  - Visualisation des distributions via des graphiques **Plotly** interactifs.

- **🧹 TP2 : Nettoyage de Données**
  - Pipeline de nettoyage automatisé (Imputation, gestion des outliers, encodage).
  - Rapport de qualité avant/après.

- **🔍 TP3 : Analyse Multivariée**
  - Analyse en Composantes Principales (PCA).
  - Visualisation 2D des clusters et projections.

- **🤖 TP4 : Machine Learning (Base)**
  - Entraînement de modèles Baseline (Régression Logistique).
  - Optimisation d'hyperparamètres (**GridSearch**) sur Random Forest.

- **💡 TP5 : ML Avancé & Interprétabilité**
  - Scores d'importance des variables.
  - Explication locale d'instances spécifiques (SHAP-like contributions).

---

## 🛠️ Installation & Lancement

Le moyen le plus simple de tester le projet est d'utiliser **Docker**.

### Avec Docker Compose (Recommandé)

```bash
# Lancer les services API (8000) et Dashboard (5000)
docker-compose up --build
```

- **Dashboard :** [http://localhost:5000](http://localhost:5000)
- **FastAPI Documentation :** [http://localhost:8000/docs](http://localhost:8000/docs)

### Lancement Manuel

1. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```
2. **Lancer le Backend :**
   ```bash
   uvicorn app.main:app --reload
   ```
3. **Lancer le Dashboard :**
   ```bash
   python flask_app/app.py
   ```

---

## 📁 Structure du Projet

```text
.
├── app/                # Backend FastAPI (Logique métier)
│   ├── routers/        # Endpoints par TP
│   ├── services/       # Calculs et ML
│   └── schemas/        # Modèles Pydantic
├── flask_app/          # Frontend Flask (Dashboard)
│   ├── templates/      # Interface HTML (Glassmorphism)
│   └── app.py          # Serveur Flask / Proxy API
├── Dockerfile.api      # Config Docker Backend
├── Dockerfile.dashboard# Config Docker Frontend
└── docker-compose.yml  # Orchestration des services
```

---

## 🧑‍💻 Auteur

**Ludovic Picard** - *Projet DataScience Sup de Vinci*

---
*Réalisé avec passion et modernité.*
