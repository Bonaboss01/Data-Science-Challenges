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



import os
import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    return df


def trim_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        # Convert to string only for trimming; keep NaNs as NaN
        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt basic type inference:
    - numeric coercion for object columns if most values look numeric
    - datetime coercion for columns containing 'date' or 'time'
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            # Try numeric conversion if at least 70% of non-null values convert
            s = df[col].dropna()
            if len(s) > 0:
                numeric_try = pd.to_numeric(s, errors="coerce")
                ratio = numeric_try.notna().mean()
                if ratio >= 0.70:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in col or "time" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def remove_outliers_iqr(df: pd.DataFrame, cap: bool = True) -> pd.DataFrame:
    """
    IQR-based outlier handling.
    If cap=True, clip values to bounds.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if cap:
            df[col] = df[col].clip(lower, upper)

    return df


def generate_quality_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
    def null_percent(df: pd.DataFrame) -> Dict[str, float]:
        return (df.isna().mean() * 100).round(2).to_dict()

    report = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "rows_before": int(len(df_before)),
        "rows_after": int(len(df_after)),
        "rows_removed": int(len(df_before) - len(df_after)),
        "columns": list(df_after.columns),
        "dtypes_after": {c: str(df_after[c].dtype) for c in df_after.columns},
        "null_percent_before": null_percent(df_before),
        "null_percent_after": null_percent(df_after),
        "duplicate_rows_removed": int(df_before.duplicated().sum()),
    }
    return report


def enforce_schema(df: pd.DataFrame, schema: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    schema example:
    {
      "customer_id": "string",
      "revenue": "float",
      "order_date": "datetime"
    }
    """
    if not schema:
        return df

    df = df.copy()

    for col, dtype in schema.items():
        if col not in df.columns:
            logging.warning(f"Schema column missing in data: {col}")
            continue

        dtype_lower = dtype.lower()

        if dtype_lower in ("string", "str", "text"):
            df[col] = df[col].astype("string")
        elif dtype_lower in ("int", "int64", "integer"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype_lower in ("float", "float64", "double", "number"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype_lower in ("datetime", "date", "timestamp"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif dtype_lower in ("bool", "boolean"):
            df[col] = df[col].astype("boolean")
        else:
            logging.warning(f"Unsupported schema type for {col}: {dtype}")

    return df


def clean_csv(input_file: str, output_file: str, report_file: str, schema_file: Optional[str] = None) -> None:
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    logging.info(f"Loading: {input_file}")
    df_before = pd.read_csv(input_file)

    df = df_before.copy()

    # Core cleaning steps
    df = standardize_columns(df)
    df = df.dropna(how="all")
    df = trim_text_fields(df)
    df = df.drop_duplicates()
    df = coerce_types(df)
    df = remove_outliers_iqr(df, cap=True)

    # Optional schema
    schema = None
    if schema_file:
        if os.path.exists(schema_file):
            with open(schema_file, "r") as f:
                schema = json.load(f)
            df = enforce_schema(df, schema)
        else:
            logging.warning(f"Schema file not found: {schema_file}")

    # Save cleaned data
    df.to_csv(output_file, index=False)
    logging.info(f"Saved cleaned CSV: {output_file}")

    # Quality report
    report = generate_quality_report(df_before, df)

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Saved quality report: {report_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="CSV Cleaner with Data Quality Report + Optional Schema")
    parser.add_argument("--input", required=True, help="Raw CSV path")
    parser.add_argument("--output", required=True, help="Cleaned CSV output path")
    parser.add_argument("--report", required=True, help="JSON report output path")
    parser.add_argument("--schema", required=False, help="Optional schema JSON path")

    args = parser.parse_args()

    clean_csv(
        input_file=args.input,
        output_file=args.output,
        report_file=args.report,
        schema_file=args.schema
    )

