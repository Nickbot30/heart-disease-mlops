# Heart Disease MLOps Pipeline

This project implements a full MLOps workflow for training, evaluating, and monitoring a heart disease prediction model using DVC, MLflow, GitHub Actions, and Evidently.

## Project Structure

src/
data.py
features.py
train.py
evaluate.py
drift_monitor.py
data/
combined/heart_reference.csv
production/heart_production.csv
configs/
train_config.yaml
reports/
drift_report.html
docs/
drift_analysis.md

## How to Run Training

Training uses MLflow for experiment tracking and DVC for data versioning.

python -m src.train

Metrics and the trained model are logged to the `mlruns/` directory.

## How to Run Evaluation

Evaluation loads a saved MLflow model and computes metrics on a dataset.

python -m src.evaluate \
--model_uri runs:/<run_id>/model \
--data_path data/production/heart_production.csv

## How to Run Drift Monitoring

Drift monitoring compares the reference dataset with the production dataset using Evidently.

python -m src.drift_monitor

The drift report is saved to:

reports/drift_report.html

## CI/CD Workflow

GitHub Actions runs:

1. A **test job** that installs dependencies, pulls DVC data, and runs pytest.
2. A **train job** that runs the training pipeline and uploads MLflow artifacts.

This ensures reproducibility and automated model training on every push or pull request.

## Drift Analysis

A full drift analysis is available here:

`docs/drift_analysis.md`

## Requirements

Install dependencies:

pip install -r requirements.txt

## Notes

- Raw CSV data is tracked with DVC and not committed directly.
- MLflow artifacts are ignored in version control.
