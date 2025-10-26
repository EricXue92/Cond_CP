import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from utils import set_seed
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve

set_seed(42)

# rxrx1  minC=1e-4, maxC=10
# iwildcam  minC=1e-6, maxC=0.01   # ← Reduce from 10.0 to 1.0 (stronger regularization) for
# fmow minC=1e-4, maxC=0.5


# # Stage 1: Coarse search
# minC = 1e-4, maxC = 10, n_points = 15
# # Stage 2: Fine search around optimum
# minC = best_C / 3, maxC = best_C * 3, n_points = 20

# C = 0.143845

def find_best_regularization(X, y, num_candidates=20, minC= 0.04, maxC = 0.5, cv_folds=5, verbose=True):
    x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)

    if hasattr(y, "detach"):
        y_np = y.detach().cpu().numpy()
    elif hasattr(y, "cpu") and callable(getattr(y, "cpu", None)):
        y_np = y.cpu().numpy()
    else:
        y_np = np.asarray(y)

    unique, counts = np.unique(y_np, return_counts=True)
    ratio = counts.min() / counts.max()
    class_weight = "balanced" if ratio < 0.5 else None

    if verbose and ratio < 0.5:
        print(f"[INFO] Imbalance detected (ratio={ratio:.3f}), using balanced weights")

    candidate_Cs = np.logspace(np.log10(minC), np.log10(maxC), num_candidates)

    model = LogisticRegressionCV(
        Cs=candidate_Cs,
        cv=cv_folds,
        scoring="neg_log_loss",
        solver="lbfgs",
        penalty="l2",
        multi_class="multinomial",
        max_iter=5000,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    model.fit(x_np, y_np)
    best_C = float(np.mean(model.C_))
    train_acc = model.score(x_np, y_np)
    if train_acc > 0.98 and verbose:
        print(f"[WARNING] Training accuracy {train_acc:.4f} suggests overfitting!")
        print(f"[WARNING] Consider using stronger regularization (lower C values)")
    mean_loss = -np.mean([s.mean(axis=0) for s in model.scores_.values()])

    if verbose:
        print(f"[INFO] Probability-calibrated CV complete.")
        print(f"       Best C = {best_C:.6f}")
        print(f"       Mean CV Log-Loss = {mean_loss:.6f} (lower = better)")
    return best_C, candidate_Cs

def computeFeatures_probs(x_train, x_cal, x_test, y_train, C=None,
                          verbose=True, **cv_params):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    if C is None:
        C, _ = find_best_regularization(x_train, y_train, verbose=verbose, **cv_params)
    elif isinstance(C, tuple):
        C = C[0]
    else:
        C = float(C)

    unique, counts = np.unique(y_train, return_counts=True)
    ratio = counts.min() / counts.max()
    class_weight = "balanced" if ratio < 0.5 else None

    if verbose:
        print(f"[INFO] Training calibrated model with C={C:.6f}...")

    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=5000,
        multi_class="multinomial",
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    preds_train = model.predict(x_train)
    probs_train = model.predict_proba(x_train)
    acc = np.mean(preds_train == y_train)
    logloss = log_loss(y_train, probs_train)

    if verbose:
        print(f"[INFO] Training Accuracy: {acc:.4f}")
        print(f"[INFO] Training Log-Loss: {logloss:.6f} (lower = better)")
        print(f"[INFO] Classes: {len(unique)} | Distribution: {dict(zip(unique, counts))}")

    features_cal = model.predict_proba(x_cal)
    features_test = model.predict_proba(x_test)
    return features_cal, features_test

def computeFeatures_indicators(x_train, x_cal, x_test, y_train, best_c=None,
                               save_path=None, dataset_name=None, include_probabilities=False):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    if save_path and os.path.exists(save_path):
        print(f"[INFO] Loading existing logistic model from {save_path}")
        saved = joblib.load(save_path)
        model = saved["model"]
    else:
        if best_c is None:
            best_c = find_best_regularization(x_train, y_train)

        unique, counts = np.unique(y_train, return_counts=True)
        imbalanced = (counts.min() / counts.max()) < 0.3
        model = LogisticRegression(
            C=best_c,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced" if imbalanced else None,
            random_state=42,
            n_jobs=-1,)
        model.fit(x_train, y_train)
        print(
            f"[INFO] Trained logistic model (C={best_c:.3g}) for indicator Φ | Train acc={model.score(x_train, y_train):.3f}")

    pred_cal = model.predict(x_cal)
    pred_test = model.predict(x_test)

    # FIX: Use actual class labels, not range(num_groups)
    unique_classes = np.unique(y_train)  # E.g., [0, 3, 4] for CAMELYON17
    num_groups = len(unique_classes)

    features_cal = np.zeros((len(pred_cal), num_groups + 1))
    features_test = np.zeros((len(pred_test), num_groups + 1))

    # Intercept column
    features_cal[:, 0] = 1.0
    features_test[:, 0] = 1.0

    # Indicator for each actual class
    for i, group_label in enumerate(unique_classes):
        features_cal[:, i + 1] = (pred_cal == group_label).astype(float)
        features_test[:, i + 1] = (pred_test == group_label).astype(float)

    if include_probabilities:
        probs_cal = model.predict_proba(x_cal)
        probs_test = model.predict_proba(x_test)
        features_cal = np.hstack([features_cal, probs_cal])
        features_test = np.hstack([features_test, probs_test])
        print(f"[INFO] Added probability features | Cal: {features_cal.shape}, Test: {features_test.shape}")

    print(f"[INFO] Created indicator Φ(x) | Cal: {features_cal.shape}, Test: {features_test.shape}")
    return features_cal, features_test


def computeFeatures_kernel(x_train, x_cal, x_test, y_train, best_c=None,
                           kernel_gamma=4.0, lambda_reg=0.005, save_path=None, dataset_name=None):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    if save_path and os.path.exists(save_path):
        print(f"[INFO] Loading saved kernel model from {save_path}")
        model = joblib.load(save_path)["model"]
    else:
        if best_c is None:
            best_c = find_best_regularization(x_train, y_train)
        model = LogisticRegression(C=best_c, max_iter=5000, random_state=42)
        model.fit(x_train, y_train)
        print(f"[INFO] Logistic model fitted (C={best_c:.3g}) for kernel Φ")

        # if save_path:
        #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        #     joblib.dump({"model": model, "scaler": scaler}, save_path)

    num_groups = len(np.unique(y_train))
    representatives = []
    for g in range(num_groups):
        samples_g = x_train[y_train == g]
        representatives.append(np.median(samples_g, axis=0))
    representatives = np.stack(representatives)
    print(f"[INFO] Using {num_groups} group medians as kernel centers")

    def gaussian_kernel(x, centers, gamma):
        diffs = x[:, None, :] - centers[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        return np.exp(-gamma * dists)

    kernel_cal = gaussian_kernel(x_cal, representatives, kernel_gamma)
    kernel_test = gaussian_kernel(x_test, representatives, kernel_gamma)

    features_cal = np.hstack([np.ones((len(kernel_cal), 1)), kernel_cal])
    features_test = np.hstack([np.ones((len(kernel_test), 1)), kernel_test])

    print(f"[INFO] Created Gaussian kernel Φ(x) with γ={kernel_gamma:.2f} | Cal: {features_cal.shape}, Test: {features_test.shape}")
    return features_cal, features_test



def plot_calibration_curve(y_true, y_prob, n_bins=10, title="Calibration Curve"):
    prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
    frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins)
    plt.figure(figsize=(5,5))
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.plot([0,1], [0,1], "k--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

