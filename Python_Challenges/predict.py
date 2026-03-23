def sfs_pipeline(df):
    df = feature_engineering(df)
    preds = model.predict(df)
    return preds
df = pd.read_csv("sales_data.csv")


def predict_units(model, features_df, feature_cols):
    X = features_df[feature_cols]
    preds = model.predict(X)
    
    features_df["predicted_units_sold"] = preds
    
    return features_df
