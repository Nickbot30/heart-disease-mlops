import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    # If raw file has 'num', create binary 'target'
    if "num" in df.columns and "target" not in df.columns:
        df["target"] = (df["num"] > 0).astype(int)

    return df


def train_test_split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
