from src.data import load_data, train_test_split_data
from src.features import build_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score


def test_model_prediction_shape_and_type():
    df = load_data("data/raw/heart.csv").sample(500, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split_data(
        df, "target", 0.2, 0
    )
    preprocessor, _, _ = build_preprocessor(df, "target")
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_model_meets_minimum_f1_on_sample():
    df = load_data("data/raw/heart.csv").sample(500, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split_data(
        df, "target", 0.2, 1
    )
    preprocessor, _, _ = build_preprocessor(df, "target")
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1 = f1_score(y_test, preds)
    assert f1 > 0.7
