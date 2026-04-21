import os
import tempfile

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.interpret_nb import class_statistics, plot_feature_distributions

from src.data import load_data, split_data
from src.models import train_naive_bayes

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"

FEATURE_NAMES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
]
CLASS_NAMES = ["survived", "deceased"]


@pytest.fixture(scope="module")
def pipeline():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return train_naive_bayes(X_train, y_train)


# --- class_statistics ---


def test_class_statistics_returns_dataframe(pipeline):
    df = class_statistics(pipeline, FEATURE_NAMES)
    assert isinstance(df, pd.DataFrame)


def test_class_statistics_has_feature_column(pipeline):
    df = class_statistics(pipeline, FEATURE_NAMES)
    assert "feature" in df.columns


def test_class_statistics_has_mean_columns(pipeline):
    df = class_statistics(pipeline, FEATURE_NAMES)
    assert "mean_survived" in df.columns
    assert "mean_deceased" in df.columns


def test_class_statistics_has_std_columns(pipeline):
    df = class_statistics(pipeline, FEATURE_NAMES)
    assert "std_survived" in df.columns
    assert "std_deceased" in df.columns


def test_class_statistics_has_one_row_per_feature(pipeline):
    df = class_statistics(pipeline, FEATURE_NAMES)
    assert len(df) == len(FEATURE_NAMES)


def test_class_statistics_all_stds_positive(pipeline):
    df = class_statistics(pipeline, FEATURE_NAMES)
    assert (df["std_survived"] > 0).all()
    assert (df["std_deceased"] > 0).all()


def test_class_statistics_features_match(pipeline):
    df = class_statistics(pipeline, FEATURE_NAMES)
    assert list(df["feature"]) == FEATURE_NAMES


# --- plot_feature_distributions ---


def test_plot_feature_distributions_creates_file(pipeline):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nb_distributions.png")
        plot_feature_distributions(pipeline, FEATURE_NAMES, CLASS_NAMES, path)
        assert os.path.isfile(path)


def test_plot_feature_distributions_file_is_non_empty(pipeline):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nb_distributions.png")
        plot_feature_distributions(pipeline, FEATURE_NAMES, CLASS_NAMES, path)
        assert os.path.getsize(path) > 0
