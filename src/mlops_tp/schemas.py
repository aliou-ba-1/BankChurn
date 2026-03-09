from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Optional, List
import json
from src.mlops_tp.config import SCHEMA_PATH

# Cache global du schéma pour éviter les appels répétés
_SCHEMA_CACHE = None

def load_schema() -> Dict[str, Any]:
    """Charge le schéma des features à partir du fichier JSON (avec cache)."""
    global _SCHEMA_CACHE

    if _SCHEMA_CACHE is None:
        try:
            with open(SCHEMA_PATH, 'r') as f:
                _SCHEMA_CACHE = json.load(f)
            print(f"Feature schema loaded successfully from {SCHEMA_PATH}")
        except Exception as e:
            print(f"Error loading feature schema: {e}")
            raise e

    return _SCHEMA_CACHE

class PredictionInput(BaseModel):
    """Schéma de validation pour les données d'entrée de l'API de prédiction."""
    features: Dict[str, Any] = Field(..., description="Dictionnaire des features d'entrée", examples=[{
 'CreditScore': 619,
 'Geography': 'France',
 'Gender': 'Female',
 'Age': 42,
 'Tenure': 2,
 'Balance': 0.0,
 'NumOfProducts': 1,
 'HasCrCard': 1,
 'IsActiveMember': 1,
 'EstimatedSalary': 101348.88}])


# Réponse Health
class HealthResponse(BaseModel):
    """Schéma de validation pour la réponse de l'API de santé."""
    model_config = {"protected_namespaces": ()}
    status: str = "ok"
    model_loaded: bool

# Réponse metadata
class MetadataResponse(BaseModel):
    """Schéma de validation pour la réponse de l'API de métadonnées."""
    model_config = {"protected_namespaces": ()}

    model_version: str
    task_type: str
    features: Dict[str, Any]
    trained_at: Optional[str] = None


# Schémas pour les prédictions batch
class BatchPredictionInput(BaseModel):
    """Schéma de validation pour les données d'entrée de l'API de prédiction batch."""
    instances: List[Dict[str, Any]] = Field(..., description="Liste des observations à prédire", examples=[[{
 'CreditScore': 619,
 'Geography': 'France',
 'Gender': 'Female',
 'Age': 42,
 'Tenure': 2,
 'Balance': 0.0,
 'NumOfProducts': 1,
 'HasCrCard': 1,
 'IsActiveMember': 1,
 'EstimatedSalary': 101348.88}]])

    @model_validator(mode="after")
    def validate_instances(self):
        """Valide que les instances ne sont pas vides et que les features sont cohérentes."""
        if not self.instances:
            raise ValueError("The 'instances' list cannot be empty.")

        try:
            schema = load_schema()
            numeric_features = schema.get("numeric_features", [])
            categorical_features = schema.get("categorical_features", [])

            for idx, instance in enumerate(self.instances):
                for feature in instance.keys():
                    if feature not in numeric_features and feature not in categorical_features:
                        raise ValueError(f"Instance {idx}: Feature '{feature}' is not defined in the schema.")
        except Exception as e:
            # En cas d'erreur, laisser passer pour Swagger
            if "not a valid" in str(e).lower() or "validation error" in str(e).lower():
                pass
            else:
                raise

        return self


class SingleBatchPrediction(BaseModel):
    """Schéma pour une seule prédiction dans un batch."""
    prediction: str = Field(..., description="La classe prédite")
    proba: Optional[Dict[str, float]] = Field(None, description="Probabilités associées à chaque classe")


class SinglePredictionResponse(BaseModel):
    """Schéma de validation pour la réponse de l'API de prédiction unitaire."""
    model_config = {"protected_namespaces": ()}

    prediction: str = Field(..., description="La classe prédite")
    proba: Optional[Dict[str, float]] = Field(None, description="Probabilités associées à chaque classe")
    task: str = "classification"
    model_version: str = Field(..., description="Version du modèle utilisé pour la prédiction")
    latency_ms: float = Field(..., description="Latence de la prédiction en millisecondes")


class BatchPredictionResponse(BaseModel):
    """Schéma de validation pour la réponse de l'API de prédiction batch."""
    model_config = {"protected_namespaces": ()}

    predictions: List[SingleBatchPrediction] = Field(..., description="Liste des prédictions")
    task: str = "classification"
    model_version: str = Field(..., description="Version du modèle utilisé pour la prédiction")
    latency_ms: float = Field(..., description="Latence totale de la prédiction en millisecondes")
    count: int = Field(..., description="Nombre d'observations prédites")

