from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CONTINUOUS_COLS = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
]

BINARY_COLS = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), CONTINUOUS_COLS),
            ("pass", "passthrough", BINARY_COLS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
