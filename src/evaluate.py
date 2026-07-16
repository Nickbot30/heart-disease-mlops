import argparse
import pandas as pd
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    """
    Loads evaluation dataset (production or test).
    Missing values encoded as '?' or -9 are converted to NaN.
    """
    df = pd.read_csv(path, na_values=["?", "-9"])
    return df


def evaluate(model_uri: str, data_path: str, target_col: str = "target"):
    """
    Loads an MLflow model and evaluates it on a dataset.
    Prints metrics and logs them to MLflow.
    """

    print(f"Loading evaluation dataset from: {data_path}")
    df = load_data(data_path)

    if target_col not in df.columns:
        raise ValueError(f"Dataset must contain target column '{target_col}'")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Loading model from MLflow: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    print("Running predictions...")
    preds = model.predict(X)

    # Compute metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Log metrics to MLflow
    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_metric("eval_accuracy", acc)
        mlflow.log_metric("eval_precision", prec)
        mlflow.log_metric("eval_recall", rec)
        mlflow.log_metric("eval_f1", f1)

    print("\nMetrics logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLflow model on dataset")
    parser.add_argument("--model_uri", type=str, required=True,
                        help="MLflow model URI, e.g. 'runs:/<run_id>/model'")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation CSV file")

    args = parser.parse_args()
    evaluate(args.model_uri, args.data_path)
