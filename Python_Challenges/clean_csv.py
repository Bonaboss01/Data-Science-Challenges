import pandas as pd

def clean_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop empty rows
    df.dropna(how="all", inplace=True)

    # Trim whitespace from string columns
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].str.strip()

    df.to_csv(output_file, index=False)
    print("Cleaned file saved to:", output_file)


if __name__ == "__main__":
    clean_csv("raw_data.csv", "cleaned_data.csv")
