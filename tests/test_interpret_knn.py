import os
import tempfile

import pytest
from src.interpret_knn import explain_instance, plot_explanation

from src.data import load_data, split_data
from src.models import train_knn

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"

CLASS_NAMES = ["survived", "deceased"]


@pytest.fixture(scope="module")
def setup():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = train_knn(X_train, y_train)
    return pipeline, X_train, X_test


# --- explain_instance ---


def test_explain_instance_returns_tuple(setup):
    pipeline, X_train, X_test = setup
    feature_names = list(X_train.columns)
    instance = X_test.iloc[0]
    result = explain_instance(pipeline, X_train, instance, feature_names, CLASS_NAMES)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_explain_instance_explanation_has_local_exp(setup):
    pipeline, X_train, X_test = setup
    feature_names = list(X_train.columns)
    instance = X_test.iloc[0]
    explainer, explanation = explain_instance(
        pipeline, X_train, instance, feature_names, CLASS_NAMES
    )
    assert hasattr(explanation, "local_exp")


def test_explain_instance_explanation_has_as_list(setup):
    pipeline, X_train, X_test = setup
    feature_names = list(X_train.columns)
    instance = X_test.iloc[0]
    explainer, explanation = explain_instance(
        pipeline, X_train, instance, feature_names, CLASS_NAMES
    )
    as_list = explanation.as_list()
    assert isinstance(as_list, list)
    assert len(as_list) > 0


def test_explain_instance_weights_are_floats(setup):
    pipeline, X_train, X_test = setup
    feature_names = list(X_train.columns)
    instance = X_test.iloc[0]
    explainer, explanation = explain_instance(
        pipeline, X_train, instance, feature_names, CLASS_NAMES
    )
    for feature_desc, weight in explanation.as_list():
        assert isinstance(weight, float)


def test_explain_instance_works_for_multiple_instances(setup):
    pipeline, X_train, X_test = setup
    feature_names = list(X_train.columns)
    for i in range(3):
        instance = X_test.iloc[i]
        explainer, explanation = explain_instance(
            pipeline, X_train, instance, feature_names, CLASS_NAMES
        )
        assert explanation.as_list()


# --- plot_explanation ---


def test_plot_explanation_creates_file(setup):
    pipeline, X_train, X_test = setup
    feature_names = list(X_train.columns)
    instance = X_test.iloc[0]
    explainer, explanation = explain_instance(
        pipeline, X_train, instance, feature_names, CLASS_NAMES
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lime_explanation.png")
        plot_explanation(explanation, instance_idx=0, path=path)
        assert os.path.isfile(path)


def test_plot_explanation_file_is_non_empty(setup):
    pipeline, X_train, X_test = setup
    feature_names = list(X_train.columns)
    instance = X_test.iloc[0]
    explainer, explanation = explain_instance(
        pipeline, X_train, instance, feature_names, CLASS_NAMES
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lime_explanation.png")
        plot_explanation(explanation, instance_idx=0, path=path)
        assert os.path.getsize(path) > 0
