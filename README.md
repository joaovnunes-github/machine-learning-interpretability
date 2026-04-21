# Heart Failure Prediction — ML Interpretability

This project predicts patient survival from heart failure clinical records and compares the interpretability of three classical machine learning models: K-Nearest Neighbors (KNN), Naive Bayes, and Decision Tree. The goal is not only to evaluate predictive performance but to understand and explain model decisions using modern interpretability techniques.

## Dataset

**Heart Failure Clinical Records** (UCI / Kaggle)

- ~300 instances, 12 clinical features
- Binary target: `DEATH_EVENT` (0 = survived, 1 = deceased)
- Source: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

## Models

| Model | Type |
|---|---|
| K-Nearest Neighbors | Instance-based |
| Naive Bayes | Probabilistic |
| Decision Tree | Rule-based |

## Interpretability

- **SHAP** — global and local feature attribution
- **LIME** — local surrogate explanations for individual predictions
- **Feature importances** — Decision Tree built-in importances
- **Decision Tree visualization** — full tree plot via `sklearn` / `matplotlib`

## Structure

```
ai-interpretability/
├── data/          # Raw dataset (not committed)
├── src/           # Reusable Python modules
├── notebooks/     # Jupyter notebooks for analysis and visualizations
└── outputs/       # Saved figures and reports (not committed)
```

## Setup

```sh
pip install -r requirements.txt
```

Place the dataset CSV file (`heart_failure_clinical_records_dataset.csv`) in the `data/` directory before running any notebooks.