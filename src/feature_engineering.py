
import pandas as pd
import numpy as np

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()

    features["row_count"] = [len(df)]
    features["null_ratio"] = df.isnull().mean().mean()
    features["unique_ratio"] = df.nunique().mean() / max(len(df), 1)

    numeric = df.select_dtypes(include=np.number)
    features["numeric_mean"] = numeric.mean().mean()
    features["numeric_std"] = numeric.std().mean()
    features["zero_ratio"] = (numeric == 0).mean().mean()

    while features.shape[1] < 12:
        features[f"pad_{features.shape[1]}"] = 0.0

    return features.fillna(0.0)
