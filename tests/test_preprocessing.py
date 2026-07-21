import pandas as pd
from heart_disease_mlops.features import build_preprocessor


def sample_df():
    return pd.DataFrame({
        "age": [63, 37, None],
        "sex": [1, 1, 0],
        "cp": [3, 2, 1],
        "trestbps": [145, 130, 120],
        "chol": [233, None, 250],
        "fbs": [1, 0, 0],
        "restecg": [0, 1, 0],
        "thalach": [150, 187, None],
        "exang": [0, 1, 0],
        "oldpeak": [2.3, 3.5, None],
        "slope": [0, 2, 1],
        "ca": [0, 2, None],
        "thal": [3, 7, None],
        "target": [1, 0, 1],
    })


def test_preprocessor_does_not_modify_original_df():
    df = sample_df()
    df_copy = df.copy()
    build_preprocessor(df, "target")
    assert df.equals(df_copy)


def test_missing_values_are_handled():
    df = sample_df()
    preprocessor, _, _ = build_preprocessor(df, "target")
    X = df.drop(columns=["target"])
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(df)


def test_invalid_columns_raise_error():
    df = sample_df().drop(columns=["chol"])
    try:
        build_preprocessor(df, "target")
        assert False, "Expected ValueError for missing columns"
    except ValueError:
        assert True
def test_preprocessor_column_order_consistent():
    df = sample_df()
    preprocessor, num_cols, cat_cols = build_preprocessor(df, "target")

    # Shuffle columns
    shuffled = df.drop(columns=["target"]).sample(frac=1, axis=1)

    transformed_original = preprocessor.fit_transform(df.drop(columns=["target"]))
    transformed_shuffled = preprocessor.transform(shuffled)

    assert transformed_original.shape == transformed_shuffled.shape

def test_preprocessor_excludes_target_column():
    df = sample_df()
    preprocessor, num_cols, cat_cols = build_preprocessor(df, "target")

    assert "target" not in num_cols
    assert "target" not in cat_cols

def test_preprocessor_output_shape():
    df = sample_df()
    preprocessor, _, _ = build_preprocessor(df, "target")

    X = df.drop(columns=["target"])
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(df)
