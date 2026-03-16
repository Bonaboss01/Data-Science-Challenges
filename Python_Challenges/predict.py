def sfs_pipeline(df):
    df = feature_engineering(df)
    preds = model.predict(df)
    return preds
