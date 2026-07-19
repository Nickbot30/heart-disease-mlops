import pandas as pd
from pathlib import Path

def main():
    # Load the reference dataset
    ref = pd.read_csv("data/combined/heart_reference.csv")

    # Create a production dataset by sampling and adding slight drift
    prod = ref.sample(frac=0.6, random_state=42).copy()

    # Add slight drift to simulate real production data
    if "age" in prod.columns:
        prod["age"] = prod["age"] + 2

    if "chol" in prod.columns:
        prod["chol"] = prod["chol"] * 1.05

    # Save production dataset
    Path("data/production").mkdir(exist_ok=True)
    prod.to_csv("data/production/heart_production.csv", index=False)

if __name__ == "__main__":
    main()



