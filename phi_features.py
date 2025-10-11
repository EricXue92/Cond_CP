import numpy as np
import os
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import make_scorer, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, PredefinedSplit

from utils import set_seed
set_seed(42)

#
# def find_best_regularization(X, y, numCs=30, minC=1e-4, maxC=10.0, cv_folds=5):
#
#     x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
#     y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
#
#     Cs = np.logspace(np.log10(minC), np.log10(maxC), numCs)
#     unique, counts = np.unique(y_np, return_counts=True)
#     imbalanced = (counts.min() / counts.max()) < 0.3
#
#     print(f"\n[INFO] Searching {numCs} values from C={minC} to C={maxC}")
#     print(f"[INFO] Classes: {len(unique)}, Samples: {len(y_np)}")
#     if imbalanced:
#         print(f"[INFO] Class imbalance detected → using balanced weights")
#
#     # Drop ultra-rare classes
#     original_size = len(y_np)
#     rare_classes = unique[counts < 2]
#     if len(rare_classes) > 0:
#         mask = ~np.isin(y_np, rare_classes)
#         x_np, y_np = x_np[mask], y_np[mask]
#         unique, counts = np.unique(y_np, return_counts=True)
#         dropped = original_size - len(y_np)
#         print(
#             f"[WARN] Dropped {len(rare_classes)} rare classes ({dropped} samples, {dropped / original_size * 100:.1f}%)")
#         print(f"[INFO] Remaining: {len(unique)} classes, {len(y_np)} samples")
#
#     all_classes = unique
#
#     # Custom scorer - add **kwargs to absorb extra parameters
#     def custom_log_loss(y_true, y_pred, **kwargs):
#         return log_loss(y_true, y_pred, labels=all_classes)
#
#     scorer = make_scorer(custom_log_loss, needs_proba=True, greater_is_better=False)
#
#     # Determine CV strategy
#     min_count = counts.min()
#     adjusted_cv_folds = min(cv_folds, min_count)
#
#     if adjusted_cv_folds < 2:
#         print(f"[INFO] Min class size={min_count} → using train/val split instead of CV")
#         indices = np.arange(len(y_np))
#         train_idx, val_idx = train_test_split(
#             indices, test_size=0.2, stratify=None, random_state=42
#         )
#         test_fold = np.full(len(y_np), -1, dtype=int)
#         test_fold[val_idx] = 0
#         cv_strategy = PredefinedSplit(test_fold)
#     else:
#         if adjusted_cv_folds < cv_folds:
#             print(f"[INFO] Adjusting CV folds from {cv_folds} to {adjusted_cv_folds}")
#         cv_strategy = StratifiedKFold(n_splits=adjusted_cv_folds, shuffle=True, random_state=42)
#
#     try:
#         model = LogisticRegressionCV(
#             Cs=Cs,
#             cv=cv_strategy,
#             solver="lbfgs",
#             max_iter=5000,
#             scoring=scorer,
#             class_weight='balanced' if imbalanced else None,
#             random_state=42,
#             n_jobs=-1
#         )
#         model.fit(x_np, y_np)
#         best_c = model.C_[0]
#         print(f"[INFO] Best C: {best_c:.4f}")
#         return best_c
#
#     except (ValueError, TypeError) as e:
#         print(f"[WARN] LogisticRegressionCV failed: {e}")
#         print("[INFO] Falling back to C=1.0")
#         return 1.0


def find_best_regularization(X, y, numCs=30, minC=1e-4, maxC=10.0, cv_folds=5):
    x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)
    Cs = np.logspace(np.log10(minC), np.log10(maxC), numCs)  # log scale is standard
    unique, counts = np.unique(y_np, return_counts=True)
    imbalanced = (counts.min() / counts.max()) < 0.3
    print(f"\n[INFO] Searching {numCs} values from C={minC} to C={maxC}")
    print(f"[INFO] Classes: {len(unique)}, Samples: {len(y_np)}")
    if imbalanced:
        print(f"[INFO] Class imbalance detected → using balanced weights")
    model = LogisticRegressionCV(
        Cs=Cs,
        cv=cv_folds,
        solver="lbfgs",
        max_iter=5000,
        scoring="neg_log_loss", # Best for probability calibration
        class_weight='balanced' if imbalanced else None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_np, y_np)
    best_c = model.C_[0]
    print(f"[INFO] Best C: {best_c:.4f}")
    return best_c


def computeFeatures_probs(x_train, x_cal, x_test, y_train, best_c=None,
                    save_path=None, dataset_name=None):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    if save_path and os.path.exists(save_path):
        print(f"[INFO] Loading existing model from {save_path}")
        saved_data = joblib.load(save_path)
        model = saved_data['model']

        # Get probability features
        features_cal = model.predict_proba(x_cal)
        features_test = model.predict_proba(x_test)

        print(f"[INFO] Features shape - Cal: {features_cal.shape}, Test: {features_test.shape}")
        return features_cal, features_test

    print("[INFO] Training new model...")

    if best_c is None:
        best_c = find_best_regularization(x_train, y_train)

    unique, counts = np.unique(y_train, return_counts=True)
    imbalanced = (counts.min() / counts.max()) < 0.3

    model = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=5000,
        class_weight='balanced' if imbalanced else None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)
    print("[INFO] Model training complete")

    features_cal = model.predict_proba(x_cal)
    features_test = model.predict_proba(x_test)

    # Diagnostic metrics
    train_acc = model.score(x_train, y_train)
    train_preds = model.predict(x_train)
    cal_preds = model.predict(x_cal)
    test_preds = model.predict(x_test)

    print(f"\n[RESULTS] Train Accuracy: {train_acc:.4f}")
    print(f"[INFO] Prediction diversity:")
    print(f"  Train: {len(np.unique(train_preds))}/{len(unique)} groups predicted")
    print(f"  Cal:   {len(np.unique(cal_preds))}/{len(unique)} groups predicted")
    print(f"  Test:  {len(np.unique(test_preds))}/{len(unique)} groups predicted")

    # Check prediction confidence
    max_probs_cal = features_cal.max(axis=1)
    max_probs_test = features_test.max(axis=1)
    print(f"[INFO] Average max probability:")
    print(f"  Cal:  {max_probs_cal.mean():.3f} (std: {max_probs_cal.std():.3f})")
    print(f"  Test: {max_probs_test.mean():.3f} (std: {max_probs_test.std():.3f})")

    # # Print performance
    # train_acc = model.score(x_train, y_train)
    # print(f"\n[RESULTS] Train Accuracy: {train_acc:.4f}")
    # print(f"[INFO] Features shape - Cal: {features_cal.shape}, Test: {features_test.shape}")

    # Overfitting warning
    if train_acc > 0.99:
        print("[WARN] Perfect training accuracy - groups may be highly separable")
        if max_probs_cal.mean() > 0.95:
            print("[WARN] Very high prediction confidence - model may be overconfident")

    print(f"[INFO] Features shape - Cal: {features_cal.shape}, Test: {features_test.shape}")


    return features_cal, features_test


def computeFeatures_indicators(x_train, x_cal, x_test, y_train, best_c=None, normalize=True,
                               save_path=None, dataset_name=None, include_probabilities=False):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    scaler = None
    if normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_cal = scaler.transform(x_cal)
        x_test = scaler.transform(x_test)
        print("[INFO] Features normalized (StandardScaler)")

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
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        print(
            f"[INFO] Trained logistic model (C={best_c:.3g}) for indicator Φ | Train acc={model.score(x_train, y_train):.3f}")

        # if save_path:
        #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        #     joblib.dump({"model": model, "scaler": scaler, "best_c": best_c}, save_path)

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


def computeFeatures_kernel(x_train, x_cal, x_test, y_train, best_c=None, normalize=True,
                           kernel_gamma=4.0, lambda_reg=0.005, save_path=None, dataset_name=None):
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
    y_train = np.asarray(y_train)

    scaler = None
    if normalize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_cal = scaler.transform(x_cal)
        x_test = scaler.transform(x_test)
        print("[INFO] Features normalized (StandardScaler)")

    if save_path and os.path.exists(save_path):
        print(f"[INFO] Loading saved kernel model from {save_path}")
        model = joblib.load(save_path)["model"]
    else:
        if best_c is None:
            best_c = find_best_regularization(x_train, y_train)
        model = LogisticRegression(C=best_c, max_iter=5000, random_state=42)
        model.fit(x_train, y_train)
        print(f"[INFO] Logistic model fitted (C={best_c:.3g}) for kernel Φ")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "scaler": scaler}, save_path)

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




