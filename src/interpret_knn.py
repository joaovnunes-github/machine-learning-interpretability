import os

import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer


def explain_instance(pipeline, X_train, instance, feature_names, class_names):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    predict_fn = lambda x: pipeline.predict_proba(
        __import__("pandas").DataFrame(x, columns=feature_names)
    )

    explanation = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=predict_fn,
        num_features=len(feature_names),
        num_samples=1000,
    )

    return explainer, explanation


def plot_explanation(explanation, instance_idx, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    features_weights = explanation.as_list()
    features_weights_sorted = sorted(features_weights, key=lambda x: abs(x[1]))

    labels = [fw[0] for fw in features_weights_sorted]
    weights = [fw[1] for fw in features_weights_sorted]
    colors = ["steelblue" if w > 0 else "tomato" for w in weights]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    ax.barh(labels, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight (positive → survived, negative → deceased)")
    ax.set_title(f"LIME Explanation — Instance {instance_idx}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
