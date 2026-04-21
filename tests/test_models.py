import pytest
from sklearn.pipeline import Pipeline
from src.models import train_decision_tree, train_knn, train_naive_bayes

from src.data import load_data, split_data

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"


@pytest.fixture(scope="module")
def data():
    X, y = load_data(DATA_PATH)
    return split_data(X, y)


# --- Decision Tree ---


def test_train_decision_tree_returns_pipeline(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_decision_tree(X_train, y_train)
    assert isinstance(pipe, Pipeline)


def test_train_decision_tree_predict_runs(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_decision_tree(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_decision_tree_predictions_are_binary(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_decision_tree(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(preds).issubset({0, 1})


# --- KNN ---


def test_train_knn_returns_pipeline(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_knn(X_train, y_train)
    assert isinstance(pipe, Pipeline)


def test_train_knn_predict_runs(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_knn(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_knn_predictions_are_binary(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_knn(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(preds).issubset({0, 1})


# --- Naive Bayes ---


def test_train_naive_bayes_returns_pipeline(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_naive_bayes(X_train, y_train)
    assert isinstance(pipe, Pipeline)


def test_train_naive_bayes_predict_runs(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_naive_bayes(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)


def test_train_naive_bayes_predictions_are_binary(data):
    X_train, X_test, y_train, y_test = data
    pipe = train_naive_bayes(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(preds).issubset({0, 1})


# --- All models produce reasonable accuracy ---


def test_all_models_above_chance(data):
    X_train, X_test, y_train, y_test = data
    from sklearn.metrics import accuracy_score

    for train_fn, name in [
        (train_decision_tree, "Decision Tree"),
        (train_knn, "KNN"),
        (train_naive_bayes, "Naive Bayes"),
    ]:
        pipe = train_fn(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        assert acc > 0.5, f"{name} accuracy {acc:.3f} is not above chance"
