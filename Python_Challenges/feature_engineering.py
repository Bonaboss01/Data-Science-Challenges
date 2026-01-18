import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Feature 1: log salary
    df["log_salary"] = df["salary"].apply(
        lambda x: None if x is None else round(x ** 0.5, 2)
    )

    # Feature 2: age band
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 40, 60, 100],
        labels=["young", "mid", "senior", "retired"]
    )

    return df

if __name__ == "__main__":
    sample = pd.DataFrame({
        "age": [22, 35, 50, 70],
        "salary": [40000, 60000, 80000, 30000]
    })

    engineered = create_features(sample)
    print(engineered)

# Next
import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for demographic and income-related variables.
    """

    df = df.copy()

    # Feature 1: log-transformed salary (robust to skew)
    df["log_salary"] = df["salary"].apply(
        lambda x: np.log(x) if pd.notnull(x) and x > 0 else None
    )

    # Feature 2: income band
    df["income_band"] = pd.cut(
        df["salary"],
        bins=[0, 30000, 60000, 100000, float("inf")],
        labels=["low", "medium", "high", "very_high"]
    )

    # Feature 3: is high earner flag
    df["is_high_earner"] = df["salary"].apply(
        lambda x: 1 if pd.notnull(x) and x >= 60000 else 0
    )

    return df


if __name__ == "__main__":
    sample = pd.DataFrame({
        "age": [22, 35, 50, 70],
        "salary": [40000, 60000, 80000, 30000]
    })

    engineered = create_features(sample)
    print(engineered)

