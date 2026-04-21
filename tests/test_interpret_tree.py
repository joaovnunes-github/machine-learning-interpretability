import os
import tempfile

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.data import load_data, split_data
from src.interpret_tree import feature_importances, plot_feature_importances, plot_tree
from src.models import train_decision_tree

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"

FEATURE_NAMES = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]
CLASS_NAMES = ["survived", "deceased"]


@pytest.fixture(scope="module")
def dt():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return train_decision_tree(X_train, y_train)


# --- feature_importances ---


def test_feature_importances_returns_dataframe(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    assert isinstance(df, pd.DataFrame)


def test_feature_importances_has_correct_columns(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    assert "feature" in df.columns
    assert "importance" in df.columns


def test_feature_importances_has_all_features(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    assert len(df) == 11


def test_feature_importances_sorted_descending(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    assert list(df["importance"]) == sorted(df["importance"], reverse=True)


def test_feature_importances_sum_to_one(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    assert abs(df["importance"].sum() - 1.0) < 1e-6


def test_feature_importances_all_non_negative(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    assert (df["importance"] >= 0).all()


# --- plot_tree ---


def test_plot_tree_creates_file(dt):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tree.png")
        plot_tree(dt, FEATURE_NAMES, CLASS_NAMES, path)
        assert os.path.isfile(path)


def test_plot_tree_file_is_non_empty(dt):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tree.png")
        plot_tree(dt, FEATURE_NAMES, CLASS_NAMES, path)
        assert os.path.getsize(path) > 0


# --- plot_feature_importances ---


def test_plot_feature_importances_creates_file(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "importances.png")
        plot_feature_importances(df, path)
        assert os.path.isfile(path)


def test_plot_feature_importances_file_is_non_empty(dt):
    df = feature_importances(dt, FEATURE_NAMES)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "importances.png")
        plot_feature_importances(df, path)
        assert os.path.getsize(path) > 0
