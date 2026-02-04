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

# Another one

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


# Robust features

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering with robust handling of outliers
    and simple risk-style indicators.
    """

    df = df.copy()

    # Feature 1: capped salary (winsorization)
    salary_cap = df["salary"].quantile(0.95)
    df["salary_capped"] = df["salary"].apply(
        lambda x: min(x, salary_cap) if pd.notnull(x) else None
    )

    # Feature 2: normalized age (0–1 scaling)
    df["age_normalized"] = (
        (df["age"] - df["age"].min()) /
        (df["age"].max() - df["age"].min())
    )

    # Feature 3: potential risk flag (low income & non-working age)
    df["potential_risk_flag"] = df.apply(
        lambda row: 1
        if (
            pd.notnull(row["salary"])
            and row["salary"] < 35000
            and (row["age"] < 21 or row["age"] > 65)
        )
        else 0,
        axis=1
    )

    return df


if __name__ == "__main__":
    sample = pd.DataFrame({
        "age": [22, 35, 50, 70],
        "salary": [40000, 60000, 80000, 30000]
    })

    engineered = create_features(sample)
    print(engineered)
    # Feature Engineering can be used to create new features from existing variables.


