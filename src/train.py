import sys
from pathlib import Path

import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data import load_data, train_test_split_data
from src.features import build_preprocessor


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/train_config.yaml"):
    config = load_config(config_path)

    df = load_data(config["data"]["dvc_path"])
    X_train, X_test, y_train, y_test = train_test_split_data(
        df,
        config["data"]["target"],
        config["data"]["test_size"],
        config["data"]["random_state"],
    )

    preprocessor, _, _ = build_preprocessor(df, config["data"]["target"])

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        min_samples_split=config["model"]["min_samples_split"],
        min_samples_leaf=config["model"]["min_samples_leaf"],
        n_jobs=config["model"]["n_jobs"],
        random_state=config["model"]["random_state"],
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run():
        # log hyperparameters
        for k, v in config["model"].items():
            mlflow.log_param(k, v)

        # data version placeholder (you can inject DVC hash)
        mlflow.log_param("data_version", "dvc_heart_v1")

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Accuracy: {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall: {rec:.3f}")
        print(f"F1: {f1:.3f}")

        if f1 < config["training"]["min_f1"]:
            print("F1 below threshold, failing run.")
            sys.exit(1)


if __name__ == "__main__":
    main()
