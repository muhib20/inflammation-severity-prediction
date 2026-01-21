import pandas as pd

def prepare_features(df, target_col, group_col):
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col, group_col])
    y = df[target_col]
    groups = df[group_col]
    return X, y, groups
