import pandas as pd

df = pd.read_csv("data.csv")
print("Rows:", len(df))
print(df.isnull().sum())
# rename columns to snake_case
df.columns = df.columns.str.lower().str.replace(" ", "_")
df.to_csv("cleaned.csv", index=False)
