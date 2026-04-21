import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def permutation_importances(
    model, X_test, y_test, feature_names: list[str]
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_weighted",
        n_repeats=30,
        random_state=42,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_shap_values(
    model, X_background, X_test, model_type: str = "tree"
) -> np.ndarray:
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_test)
        # Newer SHAP returns (n_samples, n_features, n_classes); older returns list
        if isinstance(vals, list):
            return np.array(vals[1])
        if isinstance(vals, np.ndarray) and vals.ndim == 3:
            return vals[:, :, 1]
        return np.array(vals)
    else:
        # KernelExplainer: wrap predict_proba in a plain function taking numpy arrays
        # to avoid SHAP trying to set Pipeline.feature_names_in_ (read-only property)
        feature_names = list(X_background.columns)
        background_vals = shap.sample(X_background, 50, random_state=42).values

        def predict_fn(x):
            return model.predict_proba(pd.DataFrame(x, columns=feature_names))

        explainer = shap.KernelExplainer(predict_fn, background_vals)
        X_test_vals = X_test.values if hasattr(X_test, "values") else X_test
        vals = explainer.shap_values(X_test_vals, nsamples=100, silent=True)
        if isinstance(vals, list):
            return np.array(vals[1])
        if isinstance(vals, np.ndarray) and vals.ndim == 3:
            return vals[:, :, 1]
        return np.array(vals)


def plot_shap_summary(
    shap_vals: np.ndarray,
    X_test,
    feature_names: list[str],
    title: str,
    path: str,
) -> None:
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    X_display = (
        X_test
        if isinstance(X_test, pd.DataFrame)
        else pd.DataFrame(X_test, columns=feature_names)
    )

    shap.summary_plot(
        shap_vals,
        X_display,
        feature_names=feature_names,
        show=False,
        plot_type="dot",
    )
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
