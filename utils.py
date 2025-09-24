import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import math, random # for seed setting

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
        except Exception:
            pass
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

def categorical_to_numeric(data, col="experiment"):
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

def one_hot_encode(labels):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return np.array([]).reshape(0,0)
    n_classes = int(labels.max()) + 1
    return np.eye(n_classes, dtype=float)[labels] # shape (n, K)

def encode_columns(df, cols):
    """Encode multiple DataFrame columns to integers."""
    df_encoded, mappings = df.copy(), {}

    for col in cols:
        uniques = np.unique(df_encoded[col])
        mapping = {val: i for i, val in enumerate(uniques)}
        df_encoded[col] = df_encoded[col].map(mapping)
        mappings[col] = mapping

    return df_encoded, mappings

def find_best_regularization(X, y, c_range=(1e-4, 1e+2), n_values=12, cv_folds=5):

    x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y_np = np.asarray(y)
    cs = np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), n_values)

    est = LogisticRegression(penalty="l2",  solver="saga", max_iter=2000,
        tol=1e-3, random_state=42)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(estimator=est, param_grid={"C": cs}, scoring="neg_log_loss", cv=cv, n_jobs=-1)

    grid_search.fit(x_np, y_np)
    return grid_search.best_params_["C"]

# yTrain is the encoded experiment labels for the training set like [0,1,2,0,1,2,...]
def computeFeatures(x_train, x_cal, x_test, y_train, best_c):

    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    model = LogisticRegression(C=best_c, max_iter=5000)  # 5000
    model.fit(x_train, y_train)

    features_cal = model.predict_proba(x_cal)  # Shape: (n_cal, 14): probabilities for each of 14 classes
    features_test = model.predict_proba(x_test)  # shape: (n_test, 14): probabilities for each of 14 classes
    print(f"Features shape - Calibration: {features_cal.shape}, Test: {features_test.shape}")
    return features_cal, features_test

def split_threshold(scores_cal, alpha):
    """Compute split conformal threshold."""
    scores_cal = np.asarray(scores_cal, dtype=float).ravel()
    n = len(scores_cal)
    q_idx = math.ceil((n+1)*(1-alpha))/n
    return float(np.quantile(scores_cal, q_idx, method="higher"))


