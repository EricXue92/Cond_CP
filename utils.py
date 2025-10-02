import numpy as np
from sklearn.linear_model import  LogisticRegression, LogisticRegressionCV
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

def find_best_regularization(X, y, numCs=20, minC=0.001, maxC=0.1, cv_folds=5, n_jobs=-1):
    x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)

    Cs = np.linspace(minC, maxC, numCs)  # log scale is standard
    model = LogisticRegressionCV(
        Cs=Cs,
        cv=cv_folds,
        solver="lbfgs",
        max_iter=2000,
        scoring="neg_log_loss",
        n_jobs=n_jobs,
        random_state=42
    )
    model.fit(x_np, y_np)
    return model.C_[0]

def computeFeatures(x_train, x_cal, x_test, y_train, best_c):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    model = LogisticRegression(solver="lbfgs", C=best_c, max_iter=5000,
                                    random_state=42, n_jobs=-1 )
    model.fit(x_train, y_train)
    features_cal = model.predict_proba(x_cal)
    features_test = model.predict_proba(x_test)
    print(f"[INFO] Features shape - Calibration: {features_cal.shape}, Test: {features_test.shape}")
    return features_cal, features_test


def split_threshold(scores_cal, alpha):
    """Compute split conformal threshold."""
    scores_cal = np.asarray(scores_cal, dtype=float).ravel()
    n = len(scores_cal)
    q_idx = math.ceil((n+1)*(1-alpha))/n
    return float(np.quantile(scores_cal, q_idx, method="higher"))



