import pandas as pd
import numpy as np
from pathlib import Path

# Column schema for all processed datasets (numeric)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "class"
]

def load_processed_file(path: str) -> pd.DataFrame:
    """
    Loads a processed UCI heart dataset file.
    Handles missing values encoded as '?' or -9.
    Ensures correct column schema.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?", "-9"]
    )
    return df


def convert_class_to_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts multi-class disease labels (0–4) into binary target:
    0 = healthy
    1 = disease (1,2,3,4)
    """
    df["target"] = df["class"].apply(lambda x: 0 if x == 0 else 1)
    df = df.drop(columns=["class"])
    return df


def combine_all_datasets():
    # Paths to your processed datasets
    cleveland_path = "data/processed/processed.cleveland.data"
    hungarian_path = "data/processed/processed.hungarian.data"
    switzerland_path = "data/processed/processed.switzerland.data"
    va_path = "data/processed/processed.va.data"

    # Load each dataset
    df_clev = load_processed_file(cleveland_path)
    df_hung = load_processed_file(hungarian_path)
    df_swiss = load_processed_file(switzerland_path)
    df_va = load_processed_file(va_path)

    # Convert class → target
    df_clev = convert_class_to_target(df_clev)
    df_hung = convert_class_to_target(df_hung)
    df_swiss = convert_class_to_target(df_swiss)
    df_va = convert_class_to_target(df_va)

    # Combine all datasets
    combined = pd.concat([df_clev, df_hung, df_swiss, df_va], ignore_index=True)

    # Remove rows with all-null values
    combined = combined.dropna(how="all")

    # Save combined dataset
    Path("data/combined").mkdir(parents=True, exist_ok=True)
    combined.to_csv("data/combined/heart_reference.csv", index=False)

    print("Combined dataset created:")
    print(combined.shape)
    print(combined.head())


if __name__ == "__main__":
    combine_all_datasets()
