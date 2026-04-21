from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import build_pipeline


def train_decision_tree(X_train, y_train) -> Pipeline:
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    pipe.fit(X_train, y_train)
    return pipe


def train_knn(X_train, y_train, k: int = 5) -> Pipeline:
    pipe = build_pipeline(KNeighborsClassifier(n_neighbors=k))
    pipe.fit(X_train, y_train)
    return pipe


def train_naive_bayes(X_train, y_train) -> Pipeline:
    pipe = build_pipeline(GaussianNB())
    pipe.fit(X_train, y_train)
    return pipe
