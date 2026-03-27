from pathlib import Path

# Chemin
ROOT_DIR = Path(__file__).resolve().parent
while not (ROOT_DIR / "README.md").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent

# Fallback robuste (ex: image Docker sans README.md)
if not (ROOT_DIR / "README.md").exists():
    ROOT_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = ROOT_DIR/"src"/"mlops_tp"/"artifacts"   # Chemin vers le dossier artifacts
DATA_DIR = ROOT_DIR / "data"
DATA_FILE = DATA_DIR / "Churn_Modelling.csv"
TARGET_COLUMN = "Exited"  # Nom la colonne à prédire

# Division du dataset
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

# Variables
NUM_VAR = ['CreditScore', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
CAT_VAR = ['Geography', 'Gender']

# Modèle
TASK_TYPE = "classification"
MODEL_TYPE = "GradientBoosting"

# Hyperparamètres GradientBoosting (optimisés)
N_ESTIMATORS = 300
LEARNING_RATE = 0.05
MAX_DEPTH = 5
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 4
SUBSAMPLE = 0.8

# SMOTE — suréchantillonnage de la classe minoritaire
USE_SMOTE = True
SMOTE_RATIO = 0.8  # Ratio cible classe_1/classe_0 — plus élevé pour mieux capter le churn

# Artifacts
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
RUN_INFO_PATH = ARTIFACTS_DIR / "run_info.json"
MODEL_VERSION = "0.2.0"
