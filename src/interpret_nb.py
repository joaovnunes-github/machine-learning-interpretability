import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def class_statistics(pipeline, feature_names: list[str]) -> pd.DataFrame:
    nb = pipeline.named_steps["model"]
    # theta_: shape (n_classes, n_features) — per-class means
    # var_:   shape (n_classes, n_features) — per-class variances
    means = nb.theta_
    stds = np.sqrt(nb.var_)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_survived": means[0],
            "std_survived": stds[0],
            "mean_deceased": means[1],
            "std_deceased": stds[1],
        }
    )


def plot_feature_distributions(
    pipeline, feature_names: list[str], class_names: list[str], path: str
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    nb = pipeline.named_steps["model"]
    means = nb.theta_  # (n_classes, n_features)
    stds = np.sqrt(nb.var_)  # (n_classes, n_features)

    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    colors = ["steelblue", "tomato"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        ax = axes[i]
        for cls in range(len(class_names)):
            mu = means[cls, i]
            sigma = stds[cls, i]
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mu) / sigma) ** 2
            )
            ax.plot(x, pdf, color=colors[cls], label=class_names[cls], linewidth=2)
            ax.axvline(mu, color=colors[cls], linestyle="--", linewidth=1, alpha=0.7)

        ax.set_title(feature, fontsize=10, fontweight="bold")
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)

    # hide unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Naive Bayes — Learned Gaussian Distributions per Class", fontsize=13, y=1.01
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
