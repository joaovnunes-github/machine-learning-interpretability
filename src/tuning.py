import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import build_pipeline

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = "f1_weighted"


def tune_decision_tree(X_train, y_train) -> tuple[dict, float]:
    pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    param_grid = {
        "model__max_depth": [3],
        "model__min_samples_leaf": [10],
    }
    search = GridSearchCV(pipe, param_grid, cv=CV, scoring=SCORING, n_jobs=-1)
    search.fit(X_train, y_train)

    best_params = {
        "max_depth": search.best_params_["model__max_depth"],
        "min_samples_leaf": search.best_params_["model__min_samples_leaf"],
    }
    return best_params, search.best_score_


def tune_knn(X_train, y_train) -> tuple[dict, float]:
    pipe = build_pipeline(KNeighborsClassifier())
    param_grid = {"model__n_neighbors": [3, 5, 7, 9, 11, 15]}
    search = GridSearchCV(pipe, param_grid, cv=CV, scoring=SCORING, n_jobs=-1)
    search.fit(X_train, y_train)
    best_params = {"n_neighbors": search.best_params_["model__n_neighbors"]}
    return best_params, search.best_score_


def tune_naive_bayes(X_train, y_train) -> tuple[dict, float]:
    pipe = build_pipeline(GaussianNB())
    param_grid = {"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}
    search = GridSearchCV(pipe, param_grid, cv=CV, scoring=SCORING, n_jobs=-1)
    search.fit(X_train, y_train)
    best_params = {"var_smoothing": search.best_params_["model__var_smoothing"]}
    return best_params, search.best_score_
