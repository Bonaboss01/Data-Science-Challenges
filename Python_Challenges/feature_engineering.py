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
