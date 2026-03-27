# 🏦 Bank Churn Prediction — TP MLOps

> Prédiction du désabonnement des clients bancaires avec un pipeline MLOps complet : entraînement, API REST et interface interactive.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?logo=pandas&logoColor=white)

---

## 📋 Contexte métier

Le taux de désabonnement (*churn*) est un indicateur clé pour les banques : acquérir un nouveau client coûte **beaucoup plus cher** que de fidéliser un client existant.

Ce projet analyse les caractéristiques démographiques et les comportements financiers des clients afin de **prédire lesquels risquent de quitter la banque**, permettant de proposer proactivement des services personnalisés pour améliorer la fidélisation.

---

## 🎯 Tâche ML

**Classification binaire** sur la variable `Exited` :

| Valeur | Signification |
|--------|---------------|
| `0`    | Client fidèle ✅ |
| `1`    | Client perdu ❌ |

---

## 📊 Données

**Source** : [Kaggle — Bank Customer Churn Prediction](https://www.kaggle.com/datasets/chetanmittal033/bank-dataset-for-customer-churn-prediction)

| Caractéristique | Détail |
|---|---|
| Fichier | `Churn_Modelling.csv` |
| Lignes | 10 000 |
| Colonnes | 14 |
| Variables numériques | `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` |
| Variables catégorielles | `Geography`, `Gender` |
| Variable cible | `Exited` (0 ou 1) |
| Valeurs manquantes | Aucune |
| Déséquilibre de classe | 80 % fidèles / 20 % perdus |

---

## 🏗️ Architecture du projet

```
BankChurn/
├── data/
│   └── Churn_Modelling.csv          # Dataset brut
├── notebooks/
│   └── EDA_Churn_Modelling.ipynb    # Analyse exploratoire
├── src/mlops_tp/
│   ├── config.py                    # Configuration centralisée (chemins, hyperparamètres)
│   ├── train.py                     # Entraînement + feature engineering + évaluation
│   ├── inference.py                 # Chargement du modèle et prédictions
│   ├── api.py                       # API REST FastAPI
│   ├── schemas.py                   # Schémas Pydantic (validation entrées/sorties)
│   └── artifacts/
│       ├── model.joblib             # Modèle sérialisé
│       ├── metrics.json             # Métriques d'évaluation
│       ├── feature_schema.json      # Schéma des features attendues
│       └── run_info.json            # Informations sur le dernier entraînement
├── tests/
│   ├── test_api.py                  # Tests de l'API
│   ├── test_inference.py            # Tests d'inférence
│   └── test_training.py             # Tests d'entraînement
├── streamlit/
│   └── streamlit_app.py            # Interface utilisateur Streamlit
├── requirements.txt                 # Dépendances Python
└── README.md
```

---

## 🔬 Pipeline ML

### Feature Engineering

6 features supplémentaires sont créées pour enrichir le modèle :

| Feature | Description |
|---------|-------------|
| `BalanceSalaryRatio` | Ratio solde / salaire estimé (capacité d'épargne) |
| `AgeNumProducts` | Âge × nombre de produits (interaction) |
| `CreditScorePerAge` | Score de crédit normalisé par l'âge |
| `HasZeroBalance` | Indicateur binaire de solde nul |
| `TenurePerAge` | Ancienneté relative à l'âge |
| `InactiveWithCard` | Client inactif possédant une carte de crédit |

### Modèle

| Élément | Choix |
|---------|-------|
| Algorithme | **GradientBoostingClassifier** |
| Rééquilibrage | **SMOTE** (ratio 0.8) |
| Prétraitement | `StandardScaler` (numériques) + `OneHotEncoder` (catégorielles) |
| Split | 70 % train / 15 % validation / 15 % test |

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 5 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 4 |
| `subsample` | 0.8 |

---

## 📈 Résultats

### Validation

| Métrique | Score |
|----------|-------|
| **Accuracy** | 86.3 % |
| **F1-Score** (weighted) | 86.2 % |
| **AUC-ROC** | 88.5 % |
| Precision (churn) | 67.4 % |
| Recall (churn) | 64.1 % |

### Test

| Métrique | Score |
|----------|-------|
| **Accuracy** | 84.8 % |
| **F1-Score** (weighted) | 84.9 % |
| **AUC-ROC** | 86.1 % |
| Precision (churn) | 62.2 % |
| Recall (churn) | 64.3 % |

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd BankChurn

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## ▶️ Utilisation

### 1. Entraîner le modèle

```bash
python -m src.mlops_tp.train
```

Les artifacts (modèle, métriques, schéma) seront sauvegardés dans `src/mlops_tp/artifacts/`.

### 2. Lancer l'API FastAPI

```bash
uvicorn src.mlops_tp.api:app --reload
```

- 🌐 API : [http://localhost:8000](http://localhost:8000)
- 📖 Documentation Swagger : [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Lancer l'interface Streamlit

```bash
streamlit run streamlit/streamlit_app.py
```

- 🖥️ Interface : [http://localhost:8501](http://localhost:8501)

> ⚠️ L'API FastAPI doit être lancée **avant** l'application Streamlit.

---

## 🐳 Conteneurisation (Docker)

### Prérequis

- Docker
- Docker Compose (plugin `docker compose`)

### Lancer les services

```bash
docker compose up --build
```

Services disponibles :
- API FastAPI : `http://localhost:8000`
- Streamlit : `http://localhost:8501`

### Vérifier la santé des conteneurs

```bash
docker compose ps
curl http://localhost:8000/health
```

### Arrêter les services

```bash
docker compose down
```

---

## 🔌 Endpoints de l'API

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | Message de bienvenue |
| `GET` | `/health` | État de santé de l'API et du modèle |
| `GET` | `/metadata` | Métadonnées du modèle (version, métriques) |
| `POST` | `/predict` | Prédiction unitaire |
| `POST` | `/predict/batch` | Prédiction par lot |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
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

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Stack technique

| Outil | Rôle |
|-------|------|
| **pandas** | Manipulation des données |
| **scikit-learn** | Pipeline ML, prétraitement, modèle |
| **imbalanced-learn** | SMOTE (rééquilibrage des classes) |
| **FastAPI** | API REST de prédiction |
| **Pydantic** | Validation des données d'entrée/sortie |
| **Streamlit** | Interface utilisateur interactive |
| **Plotly** | Visualisations interactives |
| **pytest** | Tests unitaires et d'intégration |

---

## 📜 Licence

**CC0 : Public Domain** — Libre d'utilisation sans restriction.
# BankChurn
