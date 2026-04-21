from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import build_pipeline


def train_decision_tree(X_train, y_train) -> Pipeline:
    pipe = build_pipeline(
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_knn(X_train, y_train, k: int = 7) -> Pipeline:
    pipe = build_pipeline(KNeighborsClassifier(n_neighbors=k))
    pipe.fit(X_train, y_train)
    return pipe


def train_naive_bayes(X_train, y_train) -> Pipeline:
    pipe = build_pipeline(GaussianNB(var_smoothing=1e-11))
    pipe.fit(X_train, y_train)
    return pipe
