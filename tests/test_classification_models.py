import numpy as np
import pandas as pd
import pytest

from src.race_prediction_model.classification import ClassificationModels


def _synthetic_dataset(n=60, seed=0):
    rng = np.random.default_rng(seed)
    grid = rng.integers(1, 20, size=n)
    form = rng.normal(size=n)
    # Winner is more likely from the front of the grid - gives the classifier
    # something real to learn without needing actual race data.
    winner = (grid <= 3).astype(int)
    return pd.DataFrame({"grid": grid, "form": form, "winner": winner})


class TestClassificationModelsDeterminism:
    """Regression tests for the random_state fix: GridSearchCV used to be free
    to land on a different 'best' combo across identical runs because the
    underlying estimators had no seed of their own."""

    @pytest.fixture(autouse=True)
    def _isolate_model_dir(self, tmp_path, monkeypatch):
        # fit_model() always writes to ./model/<file_name>.pkl - run from a
        # throwaway directory so tests never touch the real model/ artifacts.
        monkeypatch.chdir(tmp_path)

    def test_same_seed_produces_identical_fitted_params_across_instances(self):
        df = _synthetic_dataset()

        models_a = ClassificationModels(df, "winner", seed=42)
        models_a.fit_model("xgboost", param_grid={"n_estimators": [10, 20]}, cross_validation=3)

        models_b = ClassificationModels(df, "winner", seed=42)
        models_b.fit_model("xgboost", param_grid={"n_estimators": [10, 20]}, cross_validation=3)

        params_a = models_a.results["xgboost"]["best_model"].get_params()
        params_b = models_b.results["xgboost"]["best_model"].get_params()
        assert params_a == params_b

    def test_same_seed_produces_identical_predictions_across_instances(self):
        df = _synthetic_dataset()

        models_a = ClassificationModels(df, "winner", seed=7)
        models_a.fit_model("logistic_regression", param_grid={"C": [1.0]}, cross_validation=3)

        models_b = ClassificationModels(df, "winner", seed=7)
        models_b.fit_model("logistic_regression", param_grid={"C": [1.0]}, cross_validation=3)

        np.testing.assert_array_equal(
            models_a.results["logistic_regression"]["pred_test"],
            models_b.results["logistic_regression"]["pred_test"],
        )

    def test_train_test_split_is_reproducible_for_a_given_seed(self):
        df = _synthetic_dataset()
        models_a = ClassificationModels(df, "winner", seed=42)
        models_b = ClassificationModels(df, "winner", seed=42)

        pd.testing.assert_frame_equal(models_a.X_train, models_b.X_train)
        pd.testing.assert_series_equal(models_a.y_train, models_b.y_train)

    def test_unknown_model_name_raises(self):
        df = _synthetic_dataset()
        models = ClassificationModels(df, "winner")
        with pytest.raises(ValueError):
            models.fit_model("not_a_real_model")
