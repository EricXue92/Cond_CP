import numpy as np
from sklearn.linear_model import  LogisticRegression, LogisticRegressionCV

import pandas as pd
import math, random # for seed setting
# from model_builder import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import os
import pickle

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

# # # 5000
# def find_best_regularization(X, y, numCs=20, minC=0.001, maxC=0.1, cv_folds=5):
#
#     x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
#     y_np = np.asarray(y)
#
#     Cvalues = np.linspace(minC, maxC, numCs)
#     folds = KFold(n_splits=cv_folds, shuffle=True)
#     losses = np.zeros(numCs)
#
#     for idx, C in enumerate(Cvalues):
#         model = LogisticRegression(solver="lbfgs", C=C, max_iter=500)
#         for train_idx, test_idx in folds.split(x_np):
#             reg = model.fit(x_np[train_idx, :], y_np[train_idx])
#             predicted_probs = reg.predict_proba(x_np[test_idx, :])
#             # Negative log-likelihood loss
#             fold_loss = -np.mean([ np.log(predicted_probs[j, int(y_np[test_idx][j])]) for j in range(len(test_idx))])
#             losses[idx] += fold_loss
#
#     best_c = Cvalues[np.argmin(losses)]
#     return best_c

def find_best_regularization(X, y, numCs=20, minC=0.001, maxC=0.1, cv_folds=5, n_jobs=-1):
    x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y_np = np.asarray(y)

    Cs = np.linspace(minC, maxC, numCs)  # log scale is standard
    model = LogisticRegressionCV(Cs=Cs, cv=cv_folds, solver="lbfgs", max_iter=5000,
        scoring="neg_log_loss", n_jobs=n_jobs, random_state=42)
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


# def train_model(x_train, y_train, x_val, y_val, C, max_epochs=200, lr=3e-3):
#     in_dim = x_train.shape[1]
#     out_dim = len(torch.unique(y_train))
#
#     model = LogisticRegression(in_dim, out_dim).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0 / C)  # L2 = 1/C
#
#     for epoch in range(max_epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(x_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()
#
#     # Evaluate on validation set (negative log-likelihood)
#     model.eval()
#     with torch.no_grad():
#         val_logits = model(x_val)
#         val_loss = criterion(val_logits, y_val).item()
#
#     return val_loss
#
# def find_best_regularization(X, y, numCs=20, minC=0.001, maxC=0.1, cv_folds=5):
#     X = torch.as_tensor(X, dtype=torch.float32, device=device)
#     y = torch.as_tensor(y, dtype=torch.long, device=device)
#     Cs = np.logspace(np.log10(minC), np.log10(maxC), numCs)  # log scale is standard
#     kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
#     avg_losses = []
#
#     for C in Cs:
#         fold_losses = []
#         for train_idx, val_idx in kf.split(X.cpu()):
#             x_train, x_val = X[train_idx].to(device), X[val_idx].to(device)
#             y_train, y_val = y[train_idx].to(device), y[val_idx].to(device)
#             val_loss = train_model(x_train, y_train, x_val, y_val, C)
#             fold_losses.append(val_loss)
#         avg_losses.append(np.mean(fold_losses))
#
#     best_c = Cs[int(np.argmin(avg_losses))]
#     return best_c
#
#
# def computeFeatures(x_train, x_cal, x_test, y_train, best_c,
#                     dataset_name):
#     x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
#     x_cal   = torch.as_tensor(x_cal,   dtype=torch.float32, device=device)
#     x_test  = torch.as_tensor(x_test,  dtype=torch.float32, device=device)
#     y_train = torch.as_tensor(y_train, dtype=torch.long, device=device)
#
#     in_dim = x_train.shape[1]
#     out_dim = len(torch.unique(y_train))
#
#     model = LogisticRegression(in_dim, out_dim).to(device)
#
#     model_path = f"checkpoints/logreg_model_{dataset_name}.pth"
#     if os.path.exists(model_path):
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint["state_dict"])
#         print(f"[INFO] Loaded model from {model_path}")
#     else:
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1.0 / best_c)
#         # Train final model
#         for epoch in range(200):
#             model.train()
#             optimizer.zero_grad()
#             outputs = model(x_train)
#             loss = criterion(outputs, y_train)
#             loss.backward()
#             optimizer.step()
#         # Save model + meta info
#         torch.save({
#                 "in_dim": in_dim,
#                 "out_dim": out_dim,
#                 "state_dict": model.state_dict()
#             }, model_path)
#         print(f"[INFO] Trained and saved model to {model_path}")
#     # Get predicted probabilities
#     model.eval()
#     with torch.no_grad():
#         features_cal = torch.softmax(model(x_cal), dim=1).cpu().numpy()
#         features_test = torch.softmax(model(x_test), dim=1).cpu().numpy()
#     print(f"[INFO] Features shape - Calibration: {features_cal.shape}, Test: {features_test.shape}")
#     return features_cal, features_test


def split_threshold(scores_cal, alpha):
    """Compute split conformal threshold."""
    scores_cal = np.asarray(scores_cal, dtype=float).ravel()
    n = len(scores_cal)
    q_idx = math.ceil((n+1)*(1-alpha))/n
    return float(np.quantile(scores_cal, q_idx, method="higher"))



