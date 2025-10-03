import numpy as np
import pandas as pd
import math, random
import torch
import os

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

# Data Processing Functions
def create_train_calib_test_split(n_samples, train_ratio=0.25, calib_ratio=0.25):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Calculate split points
    train_end = int(n_samples * train_ratio)
    calib_end = int(n_samples * (train_ratio + calib_ratio))

    return (
        indices[:train_end],  # train
        indices[train_end:calib_end],  # calibration
        indices[calib_end:]  # test
    )

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



