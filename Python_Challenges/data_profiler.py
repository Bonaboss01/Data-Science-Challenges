import pandas as pd


def profile_data(file_path, output_file="profile_report.csv"):

    df = pd.read_csv(file_path)

    report = []

    for col in df.columns:
        col_data = df[col]

        report.append({
            "column": col,
            "data_type": col_data.dtype,
            "total_rows": len(col_data),
            "missing_values": col_data.isna().sum(),
            "missing_percent": round(col_data.isna().mean() * 100, 2),
            "unique_values": col_data.nunique(),
            "sample_value": col_data.dropna().iloc[0] if not col_data.dropna().empty else None
        })

    report_df = pd.DataFrame(report)

    report_df.to_csv(output_file, index=False)

    print("Profile report saved to:", output_file)
    print("\nQuick Overview:")
    print(report_df)


if __name__ == "__main__":
    profile_data("data.csv")
