import time
import json
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.mlops_tp.schemas import (
    MetadataResponse,
    HealthResponse,
    PredictionInput,
    SinglePredictionResponse,
    BatchPredictionInput,
    BatchPredictionResponse,
    SingleBatchPrediction
)
from src.mlops_tp.config import (MODEL_PATH, METRICS_PATH, SCHEMA_PATH, MODEL_VERSION, TASK_TYPE)
from src.mlops_tp.train import feature_engineering

# Stockage global du pipeline et des métriques
ml_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et les métriques au démarrage de l'application."""
    print("Chargement du pipeline et des métriques au démarrage de l'application...")
    try:
        ml_models["pipeline"] = joblib.load(MODEL_PATH)
        with open(SCHEMA_PATH, "r") as f:
            ml_models["schema"] = json.load(f)
        with open(METRICS_PATH, "r") as f:
            ml_models["metrics"] = json.load(f)
        print("✓ Pipeline et métriques chargés avec succès.")
    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        raise
    yield
    ml_models["pipeline"] = None
    print("Pipeline cleared.")


app = FastAPI(
    title="MLOps TP API",
    description="API pour la prédiction de churn avec GradientBoosting + SMOTE",
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json"
)

@app.get('/')
def root():
    """Endpoint racine pour vérifier que l'API est en ligne."""
    return {"message": "Welcome to the MLOps TP API! Visit /docs for API documentation."}

# Gestion globale des erreurs de validation
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Gestionnaire personnalisé pour les erreurs de validation."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "loc": error.get("loc"),
                    "msg": error.get("msg"),
                    "type": error.get("type")
                }
                for error in exc.errors()
            ]
        },
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Endpoint de santé pour vérifier que l'API fonctionne et que le modèle est chargé."""
    return HealthResponse(
        status="ok",
        model_loaded="pipeline" in ml_models and ml_models["pipeline"] is not None
    )

@app.get("/metadata", response_model=MetadataResponse, tags=["Model Metadata"])
async def metadata():
    """Endpoint pour récupérer les métadonnées du modèle."""
    if "pipeline" not in ml_models or ml_models["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return MetadataResponse(
        model_version=MODEL_VERSION,
        task_type=TASK_TYPE,
        features=ml_models["schema"],
        trained_at=ml_models["metrics"].get('timestamp')
    )


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """Endpoint de prédiction unitaire qui reçoit une seule observation et retourne le résultat."""
    if "pipeline" not in ml_models or ml_models["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        t0 = time.perf_counter()
        pipeline = ml_models["pipeline"]

        # Convertir le dictionnaire en DataFrame pour sklearn
        input_df = pd.DataFrame([input_data.features])

        # Appliquer le feature engineering
        input_df = feature_engineering(input_df)

        # Faire la prédiction
        prediction = pipeline.predict(input_df)[0]

        # Calculer les probabilités si disponibles
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba_array = pipeline.predict_proba(input_df)[0]
            proba = {
                "No": float(proba_array[0]),
                "Yes": float(proba_array[1])
            }

        latency_ms = (time.perf_counter() - t0) * 1000

        return SinglePredictionResponse(
            prediction=str(prediction),
            proba=proba,
            task="classification",
            model_version=MODEL_VERSION,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction Batch"])
async def predict_batch(input_data: BatchPredictionInput):
    """Endpoint de prédiction batch qui reçoit plusieurs observations et retourne tous les résultats."""
    if "pipeline" not in ml_models or ml_models["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        t0 = time.perf_counter()
        pipeline = ml_models["pipeline"]

        # Convertir la liste d'instances en DataFrame pour sklearn
        input_df = pd.DataFrame(input_data.instances)

        # Appliquer le feature engineering
        input_df = feature_engineering(input_df)

        # Faire les prédictions
        predictions = pipeline.predict(input_df)

        # Créer la liste des résultats
        results = []
        if hasattr(pipeline, "predict_proba"):
            proba_arrays = pipeline.predict_proba(input_df)
            for pred, proba_array in zip(predictions, proba_arrays):
                proba = {
                    "No": float(proba_array[0]),
                    "Yes": float(proba_array[1])
                }
                results.append(SingleBatchPrediction(
                    prediction=str(pred),
                    proba=proba
                ))
        else:
            for pred in predictions:
                results.append(SingleBatchPrediction(
                    prediction=str(pred),
                    proba=None
                ))

        latency_ms = (time.perf_counter() - t0) * 1000

        return BatchPredictionResponse(
            predictions=results,
            task="classification",
            model_version=MODEL_VERSION,
            latency_ms=latency_ms,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


