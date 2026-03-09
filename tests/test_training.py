import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from src.mlops_tp.train import load_data, split_data, build_pipeline, evaluate_model
from src.mlops_tp.config import TARGET_COLUMN, NUM_VAR, CAT_VAR


@pytest.fixture
def raw_data():
    """Charge les données brutes."""
    return load_data()


@pytest.fixture
def split_datasets(raw_data):
    """Retourne les données divisées."""
    return split_data(raw_data)


@pytest.fixture
def trained_pipeline(split_datasets):
    """Retourne un pipeline entraîné."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_datasets
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_val, X_test, y_train, y_val, y_test


class TestLoadData:
    def test_load_data_returns_dataframe(self, raw_data):
        """Vérifie que load_data retourne un DataFrame."""
        assert isinstance(raw_data, pd.DataFrame)

    def test_load_data_not_empty(self, raw_data):
        """Vérifie que le DataFrame n'est pas vide."""
        assert len(raw_data) > 0

    def test_load_data_has_target_column(self, raw_data):
        """Vérifie que la colonne cible est présente."""
        assert TARGET_COLUMN in raw_data.columns

    def test_load_data_has_feature_columns(self, raw_data):
        """Vérifie que les colonnes de features sont présentes."""
        for col in NUM_VAR + CAT_VAR:
            assert col in raw_data.columns


class TestSplitData:
    def test_split_data_returns_six_elements(self, split_datasets):
        """Vérifie que split_data retourne 6 éléments."""
        assert len(split_datasets) == 6

    def test_split_data_removes_non_feature_columns(self, split_datasets):
        """Vérifie que RowNumber, CustomerId, Surname sont supprimées."""
        X_train = split_datasets[0]
        for col in ['RowNumber', 'CustomerId', 'Surname']:
            assert col not in X_train.columns

    def test_split_data_proportions(self, split_datasets):
        """Vérifie les proportions approximatives du split."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_datasets
        total = len(X_train) + len(X_val) + len(X_test)
        assert abs(len(X_train) / total - 0.7) < 0.05
        assert abs(len(X_val) / total - 0.15) < 0.05
        assert abs(len(X_test) / total - 0.15) < 0.05

    def test_split_data_target_not_in_features(self, split_datasets):
        """Vérifie que la colonne cible n'est pas dans X."""
        X_train = split_datasets[0]
        assert TARGET_COLUMN not in X_train.columns


class TestBuildPipeline:
    def test_build_pipeline_returns_pipeline(self, split_datasets):
        """Vérifie que build_pipeline retourne un objet Pipeline."""
        X_train = split_datasets[0]
        pipeline = build_pipeline(X_train)
        assert isinstance(pipeline, (Pipeline, ImbPipeline))

    def test_pipeline_has_preprocessor_and_classifier(self, split_datasets):
        """Vérifie que le pipeline contient un preprocessor et un classifier."""
        X_train = split_datasets[0]
        pipeline = build_pipeline(X_train)
        step_names = [name for name, _ in pipeline.steps]
        assert 'preprocessor' in step_names
        assert 'classifier' in step_names


class TestEvaluateModel:
    def test_evaluate_returns_dict(self, trained_pipeline):
        """Vérifie que evaluate_model retourne un dictionnaire."""
        pipeline, X_train, X_val, X_test, y_train, y_val, y_test = trained_pipeline
        metrics = evaluate_model(pipeline, X_val, y_val, X_test, y_test)
        assert isinstance(metrics, dict)

    def test_evaluate_has_validation_metrics(self, trained_pipeline):
        """Vérifie que les métriques de validation sont présentes."""
        pipeline, X_train, X_val, X_test, y_train, y_val, y_test = trained_pipeline
        metrics = evaluate_model(pipeline, X_val, y_val, X_test, y_test)
        assert "Validation" in metrics
        assert "accuracy" in metrics["Validation"]
        assert "f1_score" in metrics["Validation"]

    def test_evaluate_has_test_metrics(self, trained_pipeline):
        """Vérifie que les métriques de test sont présentes."""
        pipeline, X_train, X_val, X_test, y_train, y_val, y_test = trained_pipeline
        metrics = evaluate_model(pipeline, X_val, y_val, X_test, y_test)
        assert "Test" in metrics
        assert "accuracy" in metrics["Test"]
        assert "f1_score" in metrics["Test"]

    def test_evaluate_metrics_in_valid_range(self, trained_pipeline):
        """Vérifie que les métriques sont dans une plage valide [0, 1]."""
        pipeline, X_train, X_val, X_test, y_train, y_val, y_test = trained_pipeline
        metrics = evaluate_model(pipeline, X_val, y_val, X_test, y_test)
        for split in ["Validation", "Test"]:
            assert 0 <= metrics[split]["accuracy"] <= 1
            assert 0 <= metrics[split]["f1_score"] <= 1
            assert 0 <= metrics[split]["roc_auc"] <= 1

