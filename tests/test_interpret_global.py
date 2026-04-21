import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from src.interpret_global import (
    compute_shap_values,
    permutation_importances,
    plot_shap_summary,
)

from src.data import load_data, split_data
from src.models import train_decision_tree, train_knn, train_naive_bayes

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"


@pytest.fixture(scope="module")
def data():
    X, y = load_data(DATA_PATH)
    return split_data(X, y)


@pytest.fixture(scope="module")
def models(data):
    X_train, X_test, y_train, y_test = data
    return {
        "dt": train_decision_tree(X_train, y_train),
        "knn": train_knn(X_train, y_train),
        "nb": train_naive_bayes(X_train, y_train),
    }


@pytest.fixture(scope="module")
def feature_names(data):
    X_train, _, _, _ = data
    return list(X_train.columns)


# ---------------------------------------------------------------------------
# permutation_importances
# ---------------------------------------------------------------------------


def test_perm_importance_returns_dataframe(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    df = permutation_importances(models["dt"], X_test, y_test, feature_names)
    assert isinstance(df, pd.DataFrame)


def test_perm_importance_has_correct_columns(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    df = permutation_importances(models["dt"], X_test, y_test, feature_names)
    assert "feature" in df.columns
    assert "importance_mean" in df.columns
    assert "importance_std" in df.columns


def test_perm_importance_has_one_row_per_feature(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    df = permutation_importances(models["dt"], X_test, y_test, feature_names)
    assert len(df) == len(feature_names)


def test_perm_importance_sorted_descending(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    df = permutation_importances(models["dt"], X_test, y_test, feature_names)
    assert list(df["importance_mean"]) == sorted(df["importance_mean"], reverse=True)


def test_perm_importance_stds_non_negative(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    df = permutation_importances(models["dt"], X_test, y_test, feature_names)
    assert (df["importance_std"] >= 0).all()


def test_perm_importance_works_for_pipeline(data, models, feature_names):
    # KNN and NB are pipelines — permutation importance must work through them
    X_train, X_test, y_train, y_test = data
    for name in ("knn", "nb"):
        df = permutation_importances(models[name], X_test, y_test, feature_names)
        assert len(df) == len(feature_names)


# ---------------------------------------------------------------------------
# compute_shap_values — tree
# ---------------------------------------------------------------------------


def test_shap_tree_returns_array(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    vals = compute_shap_values(models["dt"], X_train, X_test, model_type="tree")
    assert isinstance(vals, np.ndarray)


def test_shap_tree_shape(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    vals = compute_shap_values(models["dt"], X_train, X_test, model_type="tree")
    assert vals.shape == (len(X_test), len(feature_names))


# ---------------------------------------------------------------------------
# compute_shap_values — kernel (use small slice to keep tests fast)
# ---------------------------------------------------------------------------


def test_shap_kernel_returns_array(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    vals = compute_shap_values(
        models["knn"], X_train, X_test.iloc[:5], model_type="kernel"
    )
    assert isinstance(vals, np.ndarray)


def test_shap_kernel_shape(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    vals = compute_shap_values(
        models["knn"], X_train, X_test.iloc[:5], model_type="kernel"
    )
    assert vals.shape == (5, len(feature_names))


def test_shap_kernel_works_for_nb(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    vals = compute_shap_values(
        models["nb"], X_train, X_test.iloc[:5], model_type="kernel"
    )
    assert vals.shape == (5, len(feature_names))


# ---------------------------------------------------------------------------
# plot_shap_summary
# ---------------------------------------------------------------------------


def test_plot_shap_summary_creates_file(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    vals = compute_shap_values(models["dt"], X_train, X_test, model_type="tree")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "shap_summary.png")
        plot_shap_summary(vals, X_test, feature_names, title="Test", path=path)
        assert os.path.isfile(path)


def test_plot_shap_summary_file_is_non_empty(data, models, feature_names):
    X_train, X_test, y_train, y_test = data
    vals = compute_shap_values(models["dt"], X_train, X_test, model_type="tree")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "shap_summary.png")
        plot_shap_summary(vals, X_test, feature_names, title="Test", path=path)
        assert os.path.getsize(path) > 0
