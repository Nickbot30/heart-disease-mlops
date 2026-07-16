from src.data import load_data


def test_expected_columns_present():
    df = load_data("data/raw/heart.csv")
    expected = {
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    }
    assert expected.issubset(df.columns)


def test_target_is_binary():
    df = load_data("data/raw/heart.csv")
    unique = set(df["target"].unique())
    assert unique.issubset({0, 1})


def test_numeric_ranges_reasonable():
    df = load_data("data/raw/heart.csv")
    assert df["age"].between(20, 100).all()
    assert (df["chol"] > 0).all()
    assert (df["trestbps"] > 0).all()
