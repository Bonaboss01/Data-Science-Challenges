import pandas as pd
import os
import logging
from argparse import ArgumentParser

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# Main Cleaning Function
# -----------------------------
def clean_csv(input_file, output_file):

    # Check file exists
    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        return

    logging.info("Reading CSV file...")
    df = pd.read_csv(input_file)

    original_rows = len(df)

    # -----------------------------
    # Standardize column names
    # -----------------------------
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    # -----------------------------
    # Drop completely empty rows
    # -----------------------------
    df.dropna(how="all", inplace=True)

    # -----------------------------
    # Trim whitespace in text fields
    # -----------------------------
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()

    # -----------------------------
    # Remove duplicate rows
    # -----------------------------
    df.drop_duplicates(inplace=True)

    # -----------------------------
    # Convert numeric columns where possible
    # -----------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # -----------------------------
    # Convert date columns (simple heuristic)
    # -----------------------------
    for col in df.columns:
        if "date" in col or "time" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # -----------------------------
    # Optional: Cap extreme outliers (IQR method)
    # -----------------------------
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    # -----------------------------
    # Save cleaned data
    # -----------------------------
    df.to_csv(output_file, index=False)

    cleaned_rows = len(df)

    # -----------------------------
    # Summary
    # -----------------------------
    logging.info("Cleaning completed successfully!")
    logging.info(f"Original rows: {original_rows}")
    logging.info(f"Cleaned rows: {cleaned_rows}")
    logging.info(f"Removed rows: {original_rows - cleaned_rows}")
    logging.info(f"Saved to: {output_file}")


# -----------------------------
# CLI Interface
# -----------------------------
if __name__ == "__main__":

    parser = ArgumentParser(description="Advanced CSV Cleaning Tool")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw CSV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save cleaned CSV"
    )

    args = parser.parse_args()

    clean_csv(args.input, args.output)
