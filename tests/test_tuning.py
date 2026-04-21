import pytest

from src.data import load_data, split_data
from src.tuning import tune_decision_tree, tune_knn, tune_naive_bayes

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"


@pytest.fixture(scope="module")
def data():
    X, y = load_data(DATA_PATH)
    return split_data(X, y)


# --- return shape ---


def test_tune_decision_tree_returns_params_and_score(data):
    X_train, _, y_train, _ = data
    params, score = tune_decision_tree(X_train, y_train)
    assert isinstance(params, dict)
    assert isinstance(score, float)


def test_tune_knn_returns_params_and_score(data):
    X_train, _, y_train, _ = data
    params, score = tune_knn(X_train, y_train)
    assert isinstance(params, dict)
    assert isinstance(score, float)


def test_tune_naive_bayes_returns_params_and_score(data):
    X_train, _, y_train, _ = data
    params, score = tune_naive_bayes(X_train, y_train)
    assert isinstance(params, dict)
    assert isinstance(score, float)


# --- best params contain expected keys ---


def test_tune_decision_tree_params_has_max_depth(data):
    X_train, _, y_train, _ = data
    params, _ = tune_decision_tree(X_train, y_train)
    assert "max_depth" in params


def test_tune_decision_tree_params_has_min_samples_leaf(data):
    X_train, _, y_train, _ = data
    params, _ = tune_decision_tree(X_train, y_train)
    assert "min_samples_leaf" in params


def test_tune_knn_params_has_n_neighbors(data):
    X_train, _, y_train, _ = data
    params, _ = tune_knn(X_train, y_train)
    assert "n_neighbors" in params


def test_tune_naive_bayes_params_has_var_smoothing(data):
    X_train, _, y_train, _ = data
    params, _ = tune_naive_bayes(X_train, y_train)
    assert "var_smoothing" in params


# --- scores are valid F1 values ---


def test_tune_decision_tree_score_in_range(data):
    X_train, _, y_train, _ = data
    _, score = tune_decision_tree(X_train, y_train)
    assert 0.0 <= score <= 1.0


def test_tune_knn_score_in_range(data):
    X_train, _, y_train, _ = data
    _, score = tune_knn(X_train, y_train)
    assert 0.0 <= score <= 1.0


def test_tune_naive_bayes_score_in_range(data):
    X_train, _, y_train, _ = data
    _, score = tune_naive_bayes(X_train, y_train)
    assert 0.0 <= score <= 1.0
