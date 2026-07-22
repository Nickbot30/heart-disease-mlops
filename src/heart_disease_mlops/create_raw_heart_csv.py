import pandas as pd
from pathlib import Path

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "class"
]

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?"]
    )

def drop_invalid_chol(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["chol"].notna()]
    df = df[df["chol"] > 0]
    return df

def drop_invalid_trestbps(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["trestbps"].notna()]
    df = df[df["trestbps"] > 0]
    return df

def convert_class_to_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = (df["class"] > 0).astype(int)
    return df.drop(columns=["class"])

def create_raw_heart_csv():
    files = [
        "data/processed/processed.cleveland.data",
        "data/processed/processed.hungarian.data",
        "data/processed/processed.switzerland.data",
        "data/processed/processed.va.data",
    ]

    dfs = [load_raw(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    combined = drop_invalid_chol(combined)
    combined = drop_invalid_trestbps(combined)
    combined = convert_class_to_target(combined)

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    combined.to_csv("data/raw/heart.csv", index=False)

    print("Raw unified heart.csv created:")
    print(combined.shape)
    print(combined.head())

if __name__ == "__main__":
    create_raw_heart_csv()












