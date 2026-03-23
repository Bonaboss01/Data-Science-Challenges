"""
bona.py

Small script that prints a personal profile summary.
"""

def profile():
    name = "Bonaventure"
    role = "Data Scientist"
    goal = "Become a Machine Learning Engineer"

    print("Profile Summary")
    print("----------------")
    print(f"Name: {name}")
    print(f"Role: {role}")
    print(f"Goal: {goal}")

if __name__ == "__main__":
    profile()



def weekly_sales_summary(df):
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    
    weekly = (
        df.groupby(["store_id", "product_id", "week"])["units_sold"]
        .sum()
        .reset_index()
    )
    
    return weekly
