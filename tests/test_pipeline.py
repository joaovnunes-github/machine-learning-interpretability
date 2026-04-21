import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.pipeline import build_pipeline

from src.data import load_data, split_data

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"


@pytest.fixture
def data():
    X, y = load_data(DATA_PATH)
    return split_data(X, y)


def test_build_pipeline_returns_pipeline(data):
    X_train, X_test, y_train, y_test = data
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    assert isinstance(pipe, Pipeline)


def test_pipeline_has_preprocessor_and_model(data):
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    step_names = [name for name, _ in pipe.steps]
    assert "preprocessor" in step_names
    assert "model" in step_names


def test_pipeline_fit_predict_runs(data):
    X_train, X_test, y_train, y_test = data
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)


def test_pipeline_predictions_are_binary(data):
    X_train, X_test, y_train, y_test = data
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(preds).issubset({0, 1})


def test_pipeline_continuous_columns_are_scaled(data):
    X_train, X_test, y_train, y_test = data
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    pipe.fit(X_train, y_train)

    preprocessor = pipe.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_test)

    # Continuous columns come first in the ColumnTransformer output.
    # After StandardScaler they should have mean ~0 and std ~1 on train data.
    # On test data we just check the values are in a reasonable scaled range (not raw).
    continuous_cols = X_transformed[:, :7]
    assert continuous_cols.max() < 50  # raw creatinine_phosphokinase goes up to 7861
    assert continuous_cols.min() > -50


def test_pipeline_binary_columns_are_unchanged(data):
    X_train, X_test, y_train, y_test = data
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    pipe.fit(X_train, y_train)

    preprocessor = pipe.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_test)

    # Binary columns are passed through — values must still be 0 or 1.
    binary_cols = X_transformed[:, 7:]
    assert set(np.unique(binary_cols)).issubset({0, 1})


def test_pipeline_accepts_any_sklearn_estimator(data):
    from sklearn.naive_bayes import GaussianNB

    X_train, X_test, y_train, y_test = data
    pipe = build_pipeline(GaussianNB())
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)
