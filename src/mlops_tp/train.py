import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.mlops_tp.config import (
    DATA_FILE, TARGET_COLUMN, TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, RANDOM_SEED,
    N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF,
    SUBSAMPLE, USE_SMOTE, SMOTE_RATIO,
    TASK_TYPE, METRICS_PATH, NUM_VAR, CAT_VAR, MODEL_PATH, SCHEMA_PATH, RUN_INFO_PATH
)


#=============================================================================================
# FONCTIONS DE CHARGEMENT DES DONNEES
#=============================================================================================
def load_data() -> pd.DataFrame:
    """Charge les données à partir du fichier CSV."""
    print(f"Data loaded from {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


#=============================================================================================
# FONCTIONS DE PREPARATION DES DONNEES
#=============================================================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features supplémentaires pour améliorer les performances du modèle."""
    df = df.copy()

    # Ratio Balance / Salaire estimé (capacité d'épargne)
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

    # Âge × Nombre de produits (interaction)
    df['AgeNumProducts'] = df['Age'] * df['NumOfProducts']

    # Score de crédit par tranche d'âge
    df['CreditScorePerAge'] = df['CreditScore'] / (df['Age'] + 1)

    # Client avec solde nul (indicateur binaire)
    df['HasZeroBalance'] = (df['Balance'] == 0).astype(int)

    # Ancienneté relative à l'âge
    df['TenurePerAge'] = df['Tenure'] / (df['Age'] + 1)

    # Client inactif avec carte de crédit (combinaison risquée)
    df['InactiveWithCard'] = ((df['IsActiveMember'] == 0) & (df['HasCrCard'] == 1)).astype(int)

    return df


# Features numériques ajoutées par le feature engineering
ENGINEERED_NUM_VAR = [
    'BalanceSalaryRatio', 'AgeNumProducts', 'CreditScorePerAge',
    'HasZeroBalance', 'TenurePerAge', 'InactiveWithCard'
]


def split_data(df: pd.DataFrame):
    """Divise les données en ensembles d'entraînement, de validation et de test."""
    # Supprimer les colonnes non pertinentes pour le modèle
    cols_to_drop = [col for col in ['RowNumber', 'CustomerId', 'Surname'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Feature engineering
    df = feature_engineering(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Première division en train et temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=RANDOM_SEED, stratify=y)

    # Division du temp en validation et test
    test_size_adjusted = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)  # Ajustement de la taille de test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size_adjusted, random_state=RANDOM_SEED, stratify=y_temp)

    print(f"Train : {len(X_train)} samples, ----- Validation : {len(X_val)} samples, -----Test : {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test




#=============================================================================================
# FONCTIONS DE CONSTRUCTION DU PIPELINE
#=============================================================================================
def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Construit un pipeline de prétraitement avec ColumnTransformer + GradientBoosting."""
    all_num_var = NUM_VAR + ENGINEERED_NUM_VAR
    print(f"Numeric features: {all_num_var}")
    print(f"Categorical features: {CAT_VAR}")

    # Sous-pipeline pour les features numériques
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Sous-pipeline pour les features catégorielles
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Assemblage avec ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_num_var),
            ('cat', categorical_transformer, CAT_VAR)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Classifieur GradientBoosting
    classifier = GradientBoostingClassifier(
        random_state=RANDOM_SEED,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        subsample=SUBSAMPLE
    )

    # Pipeline avec SMOTE si activé
    if USE_SMOTE:
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_SEED
            )),
            ('classifier', classifier),
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier),
        ])

    return pipeline

#=============================================================================================
# FONCTIONS D'ENTRAINEMENT ET D'ÉVALUATION
#=============================================================================================
def evaluate_model(model, X_val, y_val, X_test, y_test, threshold=0.5) -> dict:
    """Évalue le modèle sur les ensembles de validation et de test avec seuil de décision optimisé."""
    # Prédictions avec seuil optimisé
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_val_pred = (y_val_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Hyperparamètres à sauvegarder
    hyperparams = {
        "model_type": "GradientBoosting",
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "subsample": SUBSAMPLE,
        "use_smote": USE_SMOTE,
        "smote_ratio": SMOTE_RATIO,
        "optimal_threshold": threshold,
        "random_state": RANDOM_SEED,
    }

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "Hyperparameters": hyperparams,
        "Validation": {
            "accuracy": accuracy_score(y_val, y_val_pred),
            "f1_score": f1_score(y_val, y_val_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_val, y_val_proba),
            "classification_report": classification_report(y_val, y_val_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist()
        },
        "Test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score": f1_score(y_test, y_test_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_test, y_test_proba),
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist()
        },
    }

    # Sauvegarde des métriques dans un fichier JSON
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Accuracy on validation set: {metrics['Validation']['accuracy']:.4f}")
    print(f"F1 Score on validation set: {metrics['Validation']['f1_score']:.4f}")
    print(f"ROC AUC on validation set: {metrics['Validation']['roc_auc']:.4f}" if TASK_TYPE == "classification" else "ROC AUC not applicable for regression")
    print(f"Accuracy on test set: {metrics['Test']['accuracy']:.4f}")
    print(f"F1 Score on test set: {metrics['Test']['f1_score']:.4f}")
    print(f"ROC AUC on test set: {metrics['Test']['roc_auc']:.4f}" if TASK_TYPE == "classification" else "ROC AUC not applicable for regression")

    # Afficher les métriques détaillées de la classe churn
    val_cr = metrics['Validation']['classification_report']
    test_cr = metrics['Test']['classification_report']
    print(f"\nClasse Churn (1) — Validation: Recall={val_cr['1']['recall']:.4f}, F1={val_cr['1']['f1-score']:.4f}")
    print(f"Classe Churn (1) — Test:       Recall={test_cr['1']['recall']:.4f}, F1={test_cr['1']['f1-score']:.4f}")

    return metrics

#=============================================================================================
# FONCTIONS DE SAUVEGARDE DES ARTIFACTS
#=============================================================================================
def save_artifacts(model, metrics, X_train, optimal_threshold=0.5):
    """Sauvegarde le modèle entraîné, les métriques et le schéma des features."""
    # Sauvegarde du modèle
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Sauvegarde du schéma des features
    feature_schema = {
        "numeric_features": NUM_VAR,
        "categorical_features": CAT_VAR,
        "engineered_features": ENGINEERED_NUM_VAR,
        "optimal_threshold": optimal_threshold
    }
    with open(SCHEMA_PATH, 'w') as f:
        json.dump(feature_schema, f, indent=4)
    print(f"Feature schema saved to {SCHEMA_PATH}")

    # Sauvegarde des informations de run
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "model_type": "GradientBoosting",
            "n_estimators": N_ESTIMATORS,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "subsample": SUBSAMPLE,
            "use_smote": USE_SMOTE,
            "smote_ratio": SMOTE_RATIO,
            "optimal_threshold": optimal_threshold,
            "random_state": RANDOM_SEED,
        },
        "metrics_path": str(METRICS_PATH),
        "model_path": str(MODEL_PATH),
        "schema_path": str(SCHEMA_PATH)
    }
    with open(RUN_INFO_PATH, 'w') as f:
        json.dump(run_info, f, indent=4)
    print(f"Run info saved to {RUN_INFO_PATH}")

#=============================================================================================
# FONCTION PRINCIPALE
#=============================================================================================
def find_optimal_threshold(model, X_val, y_val):
    """Trouve le seuil de décision optimal qui maximise le F1-score sur la classe churn."""
    from sklearn.metrics import precision_recall_curve

    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    # Calculer le F1-score pour chaque seuil
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    print(f"   Seuil optimal trouvé : {best_threshold:.4f} (F1-churn: {f1_scores[best_idx]:.4f})")
    return best_threshold


def main():
    # Chargement des données
    print("\nLoading data...")
    df = load_data()

    # Division des données
    print("\nSplitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Construction du pipeline
    print("\nConstruction du pipeline de prétraitement et de modélisation...")
    pipeline = build_pipeline(X_train)

    # Entraînement du modèle
    print("\nTraining the model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed.")

    # Recherche du seuil optimal sur la validation
    print("\n🔍 Recherche du seuil de décision optimal...")
    optimal_threshold = find_optimal_threshold(pipeline, X_val, y_val)

    # Évaluation du modèle avec seuil optimisé
    print("\nEvaluating model...")
    metrics = evaluate_model(pipeline, X_val, y_val, X_test, y_test, threshold=optimal_threshold)

    # Sauvegarde des artifacts
    print("\nSaving artifacts...")
    save_artifacts(pipeline, metrics, X_train, optimal_threshold=optimal_threshold)
    print("\nEntrainement terminé avec succès. Artifacts saved.")

if __name__ == "__main__":
    main()


