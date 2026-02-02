import os
import pandas as pd


def merge_csv_files(folder_path, output_file):

    all_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    if not all_files:
        print("No CSV files found in folder.")
        return

    dataframes = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Basic validation: skip empty files
        if df.empty:
            print(f"Skipping empty file: {file}")
            continue

        df["source_file"] = file  # track where data came from
        dataframes.append(df)

    if not dataframes:
        print("No valid CSV files to merge.")
        return

    merged_df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates
    merged_df.drop_duplicates(inplace=True)

    merged_df.to_csv(output_file, index=False)

    print(f"Merged {len(dataframes)} files into {output_file}")
    print(f"Total rows: {len(merged_df)}")


if __name__ == "__main__":
    merge_csv_files("data_folder", "merged_data.csv")
