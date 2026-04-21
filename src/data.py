import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=["DEATH_EVENT", "time"])
    y = df["DEATH_EVENT"]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
