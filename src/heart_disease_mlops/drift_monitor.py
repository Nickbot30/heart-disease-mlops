import sys
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from heart_disease_mlops.data import load_data


def main(
    reference_path: str = "data/combined/heart_reference.csv",
    production_path: str = "data/production/heart_production.csv",
    threshold: float = 0.3,
):

    ref = load_data(reference_path)
    prod = load_data(production_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=prod)

    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports/drift_report.html")
    report.save_html(str(report_path))

    summary = report.as_dict()
    drift_share = summary["metrics"][0]["result"]["drift_share"]
    print(f"Drift share: {drift_share:.3f}")

    print("Drifted features:")
    for feat, res in summary["metrics"][0]["result"]["features"].items():
        if res.get("drift_detected"):
            print(f" - {feat}")

    if drift_share > threshold:
        print("Drift above threshold, exiting with code 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
