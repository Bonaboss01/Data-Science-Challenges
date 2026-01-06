import pandas as pd

def data_quality_summary(df: pd.DataFrame):
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": df.isna().sum().to_dict()
    }

if __name__ == "__main__":
    sample = pd.DataFrame({
        "age": [25, None, 30, 45],
        "salary": [50000, 60000, None, 80000]
    })

    summary = data_quality_summary(sample)
    print(summary)
