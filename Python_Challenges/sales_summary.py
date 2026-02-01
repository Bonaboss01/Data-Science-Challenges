import pandas as pd

def sales_summary(file_path):
    df = pd.read_csv(file_path)

    total_sales = df["revenue"].sum()
    avg_sales = df["revenue"].mean()
    top_product = df.groupby("product")["revenue"].sum().idxmax()

    print("Total Sales:", total_sales)
    print("Average Sale:", avg_sales)
    print("Top Product:", top_product)


if __name__ == "__main__":
    sales_summary("sales.csv")
