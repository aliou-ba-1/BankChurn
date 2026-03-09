import pytest
from fastapi.testclient import TestClient

from src.mlops_tp.api import app


@pytest.fixture
def client():
    """Retourne un client de test FastAPI."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_features():
    """Retourne un exemple de features valides."""
    return {
        'CreditScore': 619,
        'Geography': 'France',
        'Gender': 'Female',
        'Age': 42,
        'Tenure': 2,
        'Balance': 0.0,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 101348.88
    }


class TestRoot:
    def test_root_returns_200(self, client):
        """Vérifie que l'endpoint racine retourne 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_message(self, client):
        """Vérifie que la réponse contient un message."""
        response = client.get("/")
        assert "message" in response.json()


class TestHealth:
    def test_health_returns_200(self, client):
        """Vérifie que l'endpoint health retourne 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client):
        """Vérifie que le statut est 'ok'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_model_loaded(self, client):
        """Vérifie que le modèle est chargé."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True


class TestMetadata:
    def test_metadata_returns_200(self, client):
        """Vérifie que l'endpoint metadata retourne 200."""
        response = client.get("/metadata")
        assert response.status_code == 200

    def test_metadata_has_required_fields(self, client):
        """Vérifie que les métadonnées contiennent les champs requis."""
        response = client.get("/metadata")
        data = response.json()
        assert "model_version" in data
        assert "task_type" in data
        assert "features" in data

    def test_metadata_task_type(self, client):
        """Vérifie que le task_type est 'classification'."""
        response = client.get("/metadata")
        data = response.json()
        assert data["task_type"] == "classification"


class TestPredict:
    def test_predict_returns_200(self, client, sample_features):
        """Vérifie que l'endpoint predict retourne 200."""
        response = client.post("/predict", json={"features": sample_features})
        assert response.status_code == 200

    def test_predict_has_prediction(self, client, sample_features):
        """Vérifie que la réponse contient une prédiction."""
        response = client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["0", "1"]

    def test_predict_has_proba(self, client, sample_features):
        """Vérifie que la réponse contient les probabilités."""
        response = client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert "proba" in data
        if data["proba"] is not None:
            assert "No" in data["proba"]
            assert "Yes" in data["proba"]

    def test_predict_has_latency(self, client, sample_features):
        """Vérifie que la réponse contient la latence."""
        response = client.post("/predict", json={"features": sample_features})
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] > 0

    def test_predict_invalid_input(self, client):
        """Vérifie qu'un input invalide retourne une erreur 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestPredictBatch:
    def test_predict_batch_returns_200(self, client, sample_features):
        """Vérifie que l'endpoint batch retourne 200."""
        response = client.post("/predict/batch", json={"instances": [sample_features]})
        assert response.status_code == 200

    def test_predict_batch_has_predictions(self, client, sample_features):
        """Vérifie que la réponse batch contient les prédictions."""
        response = client.post("/predict/batch", json={"instances": [sample_features, sample_features]})
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_predict_batch_has_count(self, client, sample_features):
        """Vérifie que la réponse contient le nombre de prédictions."""
        response = client.post("/predict/batch", json={"instances": [sample_features]})
        data = response.json()
        assert "count" in data
        assert data["count"] == 1

    def test_predict_batch_empty_instances(self, client):
        """Vérifie qu'une liste vide retourne une erreur."""
        response = client.post("/predict/batch", json={"instances": []})
        assert response.status_code == 422

    def test_predict_batch_has_latency(self, client, sample_features):
        """Vérifie que la réponse batch contient la latence."""
        response = client.post("/predict/batch", json={"instances": [sample_features]})
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] > 0

