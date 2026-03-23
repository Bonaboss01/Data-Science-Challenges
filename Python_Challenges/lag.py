def add_lag_feature(df, group_cols, target_col, lag=7):
    df = df.sort_values("date")
    df[f"{target_col}_lag_{lag}"] = (
        df.groupby(group_cols)[target_col].shift(lag)
    )
    
    return df
