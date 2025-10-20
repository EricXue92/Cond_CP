import numpy as np
import pandas as pd
import math, random
import torch
import os

from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed, enforce_determinism=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if enforce_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except Exception as e:
            print(f"[WARNING] Could not enforce deterministic algorithms: {e}")
    return seed

def create_train_calib_test_split(n_samples, y, train_ratio=0.25,
                                  calib_ratio=0.25, random_state=42):
    test_ratio = 1.0 - train_ratio - calib_ratio
    indices = np.arange(n_samples)

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=y,
        random_state=random_state
    )
    calib_size = calib_ratio / (calib_ratio + test_ratio)
    calib_idx, test_idx = train_test_split(
        temp_idx,
        train_size=calib_size,
        stratify=y[temp_idx],
        random_state=random_state
    )
    print(f"[INFO] Split sizes → Train: {len(train_idx)}, Calib: {len(calib_idx)}, Test: {len(test_idx)}")
    return train_idx, calib_idx, test_idx

def categorical_to_numeric(data, col):
    """Encode categorical labels as integers starting from 0."""
    if isinstance(data, pd.DataFrame):
        if col is None:
            raise ValueError("Column name must be provided when metadata is a DataFrame.")
        arr = data[col].copy().to_numpy()
    else:
        arr = np.asarray(data).copy()
    unique_vals = np.unique(arr)
    for i, val in enumerate(unique_vals):
        arr[arr == val] = i
    return arr.astype(int)

def encode_columns(df, cols):
    """Encode multiple DataFrame columns to integers."""
    df_encoded, mappings = df.copy(), {}
    for col in cols:
        uniques = np.unique(df_encoded[col])
        mapping = {val: i for i, val in enumerate(uniques)}
        df_encoded[col] = df_encoded[col].map(mapping)
        mappings[col] = mapping

    return df_encoded, mappings

def split_threshold(scores_cal, alpha):
    """Compute split conformal threshold."""
    scores_cal = np.asarray(scores_cal, dtype=float).ravel()
    n = len(scores_cal)
    q_idx = math.ceil((n+1)*(1-alpha))/n
    return float(np.quantile(scores_cal, q_idx, method="higher"))



