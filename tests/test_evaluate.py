import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from src.evaluate import evaluate


@pytest.fixture
def trained_model():
    X_train = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
        ]
    )
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def test_data():
    X_test = np.array([[2, 3], [7, 8]])
    y_test = np.array([0, 1])
    return X_test, y_test


def test_evaluate_returns_dict(trained_model, test_data):
    X_test, y_test = test_data
    result = evaluate(trained_model, X_test, y_test, model_name="Test Model")
    assert isinstance(result, dict)


def test_evaluate_has_accuracy_and_f1(trained_model, test_data):
    X_test, y_test = test_data
    result = evaluate(trained_model, X_test, y_test, model_name="Test Model")
    assert "accuracy" in result
    assert "f1" in result


def test_evaluate_accuracy_in_range(trained_model, test_data):
    X_test, y_test = test_data
    result = evaluate(trained_model, X_test, y_test, model_name="Test Model")
    assert 0.0 <= result["accuracy"] <= 1.0


def test_evaluate_f1_in_range(trained_model, test_data):
    X_test, y_test = test_data
    result = evaluate(trained_model, X_test, y_test, model_name="Test Model")
    assert 0.0 <= result["f1"] <= 1.0


def test_evaluate_perfect_model():
    X_train = np.array([[1, 0], [0, 1]])
    y_train = np.array([0, 1])
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    result = evaluate(model, X_train, y_train, model_name="Perfect Model")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
