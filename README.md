# BankChurn - Prediction de churn bancaire

Application MLOps de classification binaire pour predire le desabonnement client (`Exited`) avec:
- un pipeline d'entrainement scikit-learn,
- une API REST FastAPI,
- une interface Streamlit,
- un suivi d'experiences MLflow.

[![UI Render](https://img.shields.io/badge/UI-Render-46E3B7?logo=render&logoColor=white)](https://bankchurn-ui.onrender.com)
[![API Render](https://img.shields.io/badge/API-Render-46E3B7?logo=render&logoColor=white)](https://bankchurn-1-982n.onrender.com)
[![Docs](https://img.shields.io/badge/API-Docs-blue)](https://bankchurn-1-982n.onrender.com/docs)

## Vue d'ensemble

- **Cible**: `Exited` (`0` = fidele, `1` = churn)
- **Donnees**: `data/Churn_Modelling.csv`
- **Modele**: Gradient Boosting + preprocessing + option SMOTE
- **Endpoints API**: `/health`, `/metadata`, `/predict`, `/predict/batch`
- **UI**: `streamlit/streamlit_app.py`

## Acces production (Render)

- UI Streamlit: `https://bankchurn-ui.onrender.com`
- API FastAPI: `https://bankchurn-1-982n.onrender.com`
- API docs: `https://bankchurn-1-982n.onrender.com/docs`
- API health: `https://bankchurn-1-982n.onrender.com/health`

> La section **Demarrage rapide (local)** est reservee a une execution en local.

## Architecture du projet

```text
BankChurn/
├── data/
│   ├── Churn_Modelling.csv
│   └── batch_test_15.csv
├── src/mlops_tp/
│   ├── api.py
│   ├── config.py
│   ├── inference.py
│   ├── schemas.py
│   ├── train.py
│   └── artifacts/
│       ├── model.joblib
│       ├── metrics.json
│       ├── feature_schema.json
│       └── run_info.json
├── streamlit/
│   ├── Dockerfile
│   └── streamlit_app.py
├── tests/
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_training.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Prerequis

- Python 3.10+
- `pip`
- Optionnel: Docker + Docker Compose

## Installation locale

```bash
git clone <url-du-repo>
cd BankChurn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demarrage rapide (local)

1. Entrainer (ou mettre a jour) les artefacts:

```bash
python -m src.mlops_tp.train
```

2. Lancer l'API:

```bash
uvicorn src.mlops_tp.api:app --reload
```

3. Lancer Streamlit:

```bash
streamlit run streamlit/streamlit_app.py
```

Acces local:
- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

## Lancement avec Docker Compose

```bash
docker compose up --build
```

Services exposes:
- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

Arret:

```bash
docker compose down
```

Verification:

```bash
docker compose ps
curl http://localhost:8000/health
```

## Utilisation de l'API

### Endpoints

- `GET /` : message de bienvenue
- `GET /health` : etat API + modele charge
- `GET /metadata` : version, type de tache, features, date d'entrainement
- `POST /predict` : prediction unitaire
- `POST /predict/batch` : prediction batch

### Exemple `POST /predict`

```bash
API_BASE_URL=http://localhost:8000

curl -X POST "$API_BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "CreditScore": 619,
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "Tenure": 2,
      "Balance": 0.0,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 101348.88
    }
  }'
```

Pour la production Render:

```bash
API_BASE_URL=https://bankchurn-1-982n.onrender.com
```

### Exemple `POST /predict/batch`

```bash
curl -X POST "$API_BASE_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88
      }
    ]
  }'
```

## Interface Streamlit

Fichier: `streamlit/streamlit_app.py`

Variable d'environnement supportee:
- `API_URL` (defaut: `https://bankchurn-1-982n.onrender.com`)

Forcer une API locale:

```bash
export API_URL=http://localhost:8000
streamlit run streamlit/streamlit_app.py
```

## Variables d'environnement

### Entrainement / MLflow (`src/mlops_tp/config.py`)

- `MLFLOW_ENABLED` (defaut: `true`)
- `MLFLOW_TRACKING_URI` (defaut: `file:./mlruns`)
- `MLFLOW_EXPERIMENT_NAME` (defaut: `BankChurn`)
- `MLFLOW_RUN_NAME_PREFIX` (defaut: `train`)

Exemple:

```bash
export MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=file:./mlruns
export MLFLOW_EXPERIMENT_NAME=BankChurn
export MLFLOW_RUN_NAME_PREFIX=train
python -m src.mlops_tp.train
```

## Tests

Lancer tous les tests:

```bash
pytest tests/ -v
```

Fichiers principaux:
- `tests/test_training.py`
- `tests/test_inference.py`
- `tests/test_api.py`

## Troubleshooting

- **Erreur `Model not loaded` sur l'API**
  - Verifier que les artefacts existent dans `src/mlops_tp/artifacts/`.
  - Reexecuter: `python -m src.mlops_tp.train`.

- **Streamlit ne joint pas l'API locale**
  - Verifier `API_URL`.
  - En local: `export API_URL=http://localhost:8000` avant `streamlit run ...`.

- **Erreur 422 sur `/predict` ou `/predict/batch`**
  - Verifier les noms de colonnes et les types d'entree.
  - Comparer avec les exemples `curl` et la doc `/docs`.

- **Port deja utilise**
  - API: `8000`, Streamlit: `8501`.
  - Changer les ports ou arreter le process qui bloque.

## Stack technique

- `pandas`, `numpy`
- `scikit-learn`, `imbalanced-learn`
- `FastAPI`, `Pydantic`, `uvicorn`
- `Streamlit`, `plotly`
- `MLflow`
- `pytest`

## Licence

CC0 - Public Domain
