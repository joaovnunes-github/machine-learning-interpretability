import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree


def feature_importances(dt, feature_names: list[str]) -> pd.DataFrame:
    importances = dt.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_tree(dt, feature_names: list[str], class_names: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fig, ax = plt.subplots(figsize=(20, 8))
    tree.plot_tree(
        dt,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=11,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importances(importances_df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        importances_df["feature"][::-1],
        importances_df["importance"][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Decision Tree — Feature Importances")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
