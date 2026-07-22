# Drift Analysis Report

## Overview
A drift report was generated using the Evidently library to compare the current production dataset against the reference dataset created during model training. Several features show statistically significant drift, indicating that the distribution of incoming data has changed.

## Drifted Features
The following features exceeded Evidently’s drift threshold:

- **age** — Production data shows a higher proportion of patients above age 60 compared to the reference dataset.
- **chol** — Cholesterol levels have shifted upward, with more samples above 260 mg/dL.
- **thalach** — Maximum heart rate values are lower on average in production data.
- **oldpeak** — ST depression values show increased variance and a heavier right tail.

## Likely Impact on Model Performance
These changes affect the model in several ways:

- The model may **underestimate risk** for older patients if trained primarily on younger distributions.
- Higher cholesterol and lower thalach values may cause **misclassification**, especially for borderline cases.
- Increased variance in `oldpeak` reduces the model’s ability to generalize, increasing prediction instability.

Overall, the drifted features suggest the model is receiving a population that differs meaningfully from the training data, which can reduce F1 score and calibration reliability.

## Recommended Action
- **Retrain the model** using the latest production dataset to realign feature distributions.
- **Increase monitoring frequency** to weekly instead of monthly.
- **Add alerts** for large shifts in `age`, `chol`, and `thalach`, which appear to be the earliest indicators of drift.
- **Evaluate feature importance** after retraining to confirm whether drifted features are driving prediction changes.

This drift indicates that the model should be refreshed to maintain expected performance.
