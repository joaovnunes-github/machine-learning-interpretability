import pandas as pd
import pytest
from src.data import load_data, split_data

DATA_PATH = "data/heart_failure_clinical_records_dataset.csv"


def test_load_data_shapes():
    X, y = load_data(DATA_PATH)
    assert X.shape == (299, 12)
    assert y.shape == (299,)


def test_load_data_target_not_in_features():
    X, y = load_data(DATA_PATH)
    assert "DEATH_EVENT" not in X.columns


def test_load_data_target_name():
    X, y = load_data(DATA_PATH)
    assert y.name == "DEATH_EVENT"


def test_load_data_no_missing_values():
    X, y = load_data(DATA_PATH)
    assert X.isnull().sum().sum() == 0


def test_load_data_target_distribution():
    X, y = load_data(DATA_PATH)
    counts = y.value_counts()
    assert counts[0] == 203
    assert counts[1] == 96


def test_split_data_sizes():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == 299
    assert len(y_train) + len(y_test) == 299


def test_split_data_default_is_80_20():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 60
    assert len(X_train) == 239


def test_split_data_stratified():
    X, y = load_data(DATA_PATH)
    _, X_test, _, y_test = split_data(X, y)
    # Test set should preserve ~32% class-1 ratio (96/299 ≈ 0.321)
    ratio = y_test.mean()
    assert 0.28 < ratio < 0.36
