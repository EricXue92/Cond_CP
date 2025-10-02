import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import  LogisticRegression, LogisticRegressionCV


# ==================== Logistic Regression Methods ====================

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


# ==================== MLP Methods ====================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

