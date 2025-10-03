import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import  LogisticRegression, LogisticRegressionCV

from mlp_utils import SimpleMLP, train_with_early_stopping


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



def computeFeatures_mlp(x_train, x_cal, x_test, y_train,
                        hidden_dim=64, dropout=0.2,
                        weight_decay=1e-3, lr=1e-3,
                        batch_size=64, epochs=200, patience=20, val_split=0.2,
                        out_path="checkpoints/simple_mlp.pth"):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    input_dim, num_classes = x_train.shape[1], len(np.unique(y_train))

    model = SimpleMLP(input_dim, hidden_dim, num_classes, dropout).to(device)


    if os.path.exists(out_path):
        print(f"[INFO] Loading existing model from {out_path}")
        model.load_state_dict(torch.load(out_path, map_location=device))
        model.eval()
        train_losses, val_losses = [], []
    else:
        print("[INFO] Training new model...")

        val_size = int(val_split * len(x_train))
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_train[:-val_size]), torch.LongTensor(y_train[:-val_size])),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_train[-val_size:]), torch.LongTensor(y_train[-val_size:])),
            batch_size=batch_size
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer, device, epochs, patience
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        print(f"[INFO] Model saved to {out_path}")
        model.eval()

    # Get predictions
    features_cal = model.predict_proba(x_cal, device)
    features_test = model.predict_proba(x_test, device)

    print(f"[INFO] Features shape - Calibration: {features_cal.shape}, Test: {features_test.shape}")
    return features_cal, features_test


