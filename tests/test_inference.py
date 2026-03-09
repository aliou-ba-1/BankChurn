import pytest
import pandas as pd

from src.mlops_tp.inference import get_pipeline, predict


@pytest.fixture
def sample_input():
    """Retourne un DataFrame d'exemple pour la prédiction."""
    return pd.DataFrame([{
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
    }])


class TestGetPipeline:
    def test_get_pipeline_loads_model(self):
        """Vérifie que le pipeline est chargé correctement."""
        pipeline = get_pipeline()
        assert pipeline is not None

    def test_get_pipeline_has_predict(self):
        """Vérifie que le pipeline a une méthode predict."""
        pipeline = get_pipeline()
        assert hasattr(pipeline, 'predict')


class TestPredict:
    def test_predict_returns_dict(self, sample_input):
        """Vérifie que predict retourne un dictionnaire."""
        result = predict(sample_input)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, sample_input):
        """Vérifie que le résultat contient les clés attendues."""
        result = predict(sample_input)
        assert "predictions" in result
        assert "probabilities" in result
        assert "model_version" in result

    def test_predict_returns_valid_predictions(self, sample_input):
        """Vérifie que les prédictions sont des valeurs valides (0 ou 1)."""
        result = predict(sample_input)
        for pred in result["predictions"]:
            assert pred in [0, 1]

    def test_predict_probabilities_sum_to_one(self, sample_input):
        """Vérifie que les probabilités somment à 1."""
        result = predict(sample_input)
        if result["probabilities"] is not None:
            proba_0 = result["probabilities"]["class_0"][0]
            proba_1 = result["probabilities"]["class_1"][0]
            assert abs(proba_0 + proba_1 - 1.0) < 1e-6

    def test_predict_with_dict_input(self):
        """Vérifie que predict accepte aussi un dictionnaire."""
        input_data = [{
            'CreditScore': 700,
            'Geography': 'Germany',
            'Gender': 'Male',
            'Age': 35,
            'Tenure': 5,
            'Balance': 50000.0,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 80000.0
        }]
        result = predict(input_data)
        assert "predictions" in result
        assert len(result["predictions"]) == 1

