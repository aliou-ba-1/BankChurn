import joblib
import pandas as pd

from src.mlops_tp.config import MODEL_PATH, MODEL_VERSION
from src.mlops_tp.train import feature_engineering


# Variable globale pour stocker le modèle chargé
_pipeline = None

#=============================================================================================
# FONCTIONS DE CHARGEMENT DU PIPELINE
#=============================================================================================
def get_pipeline():
    """Charge le pipeline de prétraitement et de prédiction à partir du fichier joblib."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = joblib.load(MODEL_PATH)
            print(f"Pipeline loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e
    return _pipeline

def predict(input_data):
    """Fait une prédiction à partir des données d'entrée en utilisant le pipeline chargé."""
    pipeline = get_pipeline()
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    # Appliquer le feature engineering
    input_data = feature_engineering(input_data)
    try:
        predictions = pipeline.predict(input_data)
        print(f"Predictions made successfully")
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise e

    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba_array = pipeline.predict_proba(input_data)
            proba = {
                "class_0": proba_array[:, 0].tolist(),
                "class_1": proba_array[:, 1].tolist()
            }
            print(f"Probabilities calculated successfully")
        except Exception as e:
            print(f"Error calculating probabilities: {e}")
            raise e
    return {
        "predictions": predictions.tolist(),
        "probabilities": proba,
        "model_version": MODEL_VERSION
    }

