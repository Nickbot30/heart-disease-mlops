import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def get_feature_columns(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])

    numeric_cols = [
        "age", "trestbps", "chol", "thalach", "oldpeak", "ca"
    ]

    categorical_cols = [
        "sex", "cp", "fbs", "restecg", "exang", "slope", "thal"
    ]

    # Ensure they exist
    missing = set(numeric_cols + categorical_cols) - set(X.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return numeric_cols, categorical_cols


def build_preprocessor(df: pd.DataFrame, target_col: str):
    df = df.copy()
    numeric_cols, categorical_cols = get_feature_columns(df, target_col)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols
