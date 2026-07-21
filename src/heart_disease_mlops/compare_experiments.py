import mlflow


def main():
    experiment_name = "uci_heart_disease_baseline"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment {experiment_name} not found.")

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

    best = runs.sort_values("metrics.f1", ascending=False).iloc[0]
    print("Best run ID:", best["run_id"])
    print("Best F1:", best["metrics.f1"])
    print("Params:", best.filter(like="params.").to_dict())


if __name__ == "__main__":
    main()
