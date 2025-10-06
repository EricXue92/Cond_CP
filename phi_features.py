import numpy as np
import os
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from utils import set_seed
set_seed(42)

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

def computeFeatures_probs(x_train, x_cal, x_test, y_train, best_c=None, normalize=True,
                    save_path=None, dataset_name=None):
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
        print("[INFO] Features normalized")

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

    # Print performance
    train_acc = model.score(x_train, y_train)
    print(f"\n[RESULTS] Train Accuracy: {train_acc:.4f}")

    print(f"[INFO] Features shape - Cal: {features_cal.shape}, Test: {features_test.shape}")

    # if save_path:
    #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    #     saved_data = {
    #         'model': model,
    #         'scaler': scaler,
    #         'best_c': best_c,
    #         'num_classes': len(unique),
    #         'dataset_name': dataset_name
    #     }
    #     joblib.dump(saved_data, save_path)
    #     print(f"[INFO] Model saved to {save_path}")
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
        print(f"[INFO] Trained logistic model (C={best_c:.3g}) for indicator Φ | Train acc={model.score(x_train, y_train):.3f}")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "scaler": scaler, "best_c": best_c}, save_path)

    pred_cal = model.predict(x_cal)
    pred_test = model.predict(x_test)
    num_groups = len(np.unique(y_train))

    features_cal = np.zeros((len(pred_cal), num_groups + 1))
    features_test = np.zeros((len(pred_test), num_groups + 1))
    features_cal[:, 0] = 1.0
    features_test[:, 0] = 1.0
    for g in range(num_groups):
        features_cal[:, g + 1] = (pred_cal == g).astype(float)
        features_test[:, g + 1] = (pred_test == g).astype(float)

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




# phi_features.py - Modified version
#
# def computeFeatures_indicators(x_train, x_cal, x_test, y_train,
#                                best_c=None, normalize=True,
#                                save_path=None, dataset_name=None,
#                                include_probabilities=False):
#
#     def to_numpy(x):
#         return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
#
#     x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
#     y_train = np.asarray(y_train)
#
#     # Normalize features
#     scaler = None
#     if normalize:
#         scaler = StandardScaler()
#         x_train = scaler.fit_transform(x_train)
#         x_cal = scaler.transform(x_cal)
#         x_test = scaler.transform(x_test)
#         print("[INFO] Features normalized")
#
#     # Load or train model
#     if save_path and os.path.exists(save_path):
#         print(f"[INFO] Loading existing model from {save_path}")
#         # saved_data = joblib.load(save_path)
#         # model = saved_data['model']
#     else:
#         print("[INFO] Training new model...")
#         if best_c is None:
#             best_c = find_best_regularization(x_train, y_train)
#
#         unique, counts = np.unique(y_train, return_counts=True)
#         imbalanced = (counts.min() / counts.max()) < 0.3
#
#         model = LogisticRegression(
#             C=best_c,
#             solver="lbfgs",
#             max_iter=5000,
#             class_weight='balanced' if imbalanced else None,
#             random_state=42,
#             n_jobs=-1
#         )
#         model.fit(x_train, y_train)
#         #
#         # if save_path:
#         #     Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#         #     joblib.dump({'model': model, 'scaler': scaler, 'best_c': best_c}, save_path)
#
#     # CRITICAL: Get hard predictions for indicators
#     pred_cal = model.predict(x_cal)
#     pred_test = model.predict(x_test)
#
#     # Create indicator matrix: Φ(x) = [1, 1{x ∈ G_1}, 1{x ∈ G_2}, ...]
#     num_groups = len(np.unique(y_train))
#
#     # Intercept + group indicators
#     features_cal = np.zeros((len(pred_cal), num_groups + 1))
#     features_test = np.zeros((len(pred_test), num_groups + 1))
#
#     features_cal[:, 0] = 1  # intercept
#     features_test[:, 0] = 1
#
#     for g in range(num_groups):
#         features_cal[:, g + 1] = (pred_cal == g).astype(float)
#         features_test[:, g + 1] = (pred_test == g).astype(float)
#
#     print(f"[INFO] Created indicator features - Cal: {features_cal.shape}, Test: {features_test.shape}")
#     print(f"[INFO] Feature format: [intercept, 1{{age∈G_0}}, 1{{age∈G_1}}, ...]")
#
#     # Optional: add soft probabilities for richer coverage (Section 3.1)
#     if include_probabilities:
#         probs_cal = model.predict_proba(x_cal)
#         probs_test = model.predict_proba(x_test)
#         features_cal = np.hstack([features_cal, probs_cal])
#         features_test = np.hstack([features_test, probs_test])
#         print(f"[INFO] Added probability features - Cal: {features_cal.shape}, Test: {features_test.shape}")
#
#     # Print group distribution for debugging
#     print("\n[DEBUG] Predicted age group distribution (calibration):")
#     unique_cal, counts_cal = np.unique(pred_cal, return_counts=True)
#     for g, c in zip(unique_cal, counts_cal):
#         print(f"  Group {g}: {c} samples ({100 * c / len(pred_cal):.1f}%)")
#
#     return features_cal, features_test
#
#
# def computeFeatures_kernel(x_train, x_cal, x_test, y_train,
#                            best_c=None, normalize=True,
#                            kernel_gamma=4.0, lambda_reg=0.005,
#                            save_path=None):
#     def to_numpy(x):
#         return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
#
#     x_train, x_cal, x_test = map(to_numpy, [x_train, x_cal, x_test])
#     y_train = np.asarray(y_train)
#
#     # Normalize
#     scaler = StandardScaler() if normalize else None
#     if normalize:
#         x_train = scaler.fit_transform(x_train)
#         x_cal = scaler.transform(x_cal)
#         x_test = scaler.transform(x_test)
#
#     # Train classifier
#     if save_path and os.path.exists(save_path):
#         saved_data = joblib.load(save_path)
#         model = saved_data['model']
#     else:
#         if best_c is None:
#             best_c = find_best_regularization(x_train, y_train)
#         model = LogisticRegression(C=best_c, max_iter=5000, random_state=42)
#         model.fit(x_train, y_train)
#         # if save_path:
#         #     joblib.dump({'model': model, 'scaler': scaler}, save_path)
#
#     # Create kernel features: K(x, x_i) for representative points
#     # Use representative points from each age group
#     num_groups = len(np.unique(y_train))
#     representatives = []
#     for g in range(num_groups):
#         group_samples = x_train[y_train == g]
#         # Use median of each group as representative
#         representatives.append(np.median(group_samples, axis=0))
#     representatives = np.array(representatives)
#
#     # Compute Gaussian kernel features
#     def gaussian_kernel(x, centers, gamma):
#         # K(x, c) = exp(-gamma * ||x - c||^2)
#         n_samples = x.shape[0]
#         n_centers = centers.shape[0]
#         kernel_features = np.zeros((n_samples, n_centers))
#         for i in range(n_centers):
#             diff = x - centers[i]
#             kernel_features[:, i] = np.exp(-gamma * np.sum(diff ** 2, axis=1))
#         return kernel_features
#
#     kernel_cal = gaussian_kernel(x_cal, representatives, kernel_gamma)
#     kernel_test = gaussian_kernel(x_test, representatives, kernel_gamma)
#
#     # Add intercept: Φ(x) = [1, K(x, c_1), K(x, c_2), ...]
#     features_cal = np.hstack([np.ones((len(kernel_cal), 1)), kernel_cal])
#     features_test = np.hstack([np.ones((len(kernel_test), 1)), kernel_test])
#
#     print(f"[INFO] Created kernel features with gamma={kernel_gamma}")
#     print(f"[INFO] Shape - Cal: {features_cal.shape}, Test: {features_test.shape}")
#
#     return features_cal, features_test
#
#
# def computeFeatures_metadata(x_train, x_cal, x_test, y_train,
#                              metadata=None,
#                              calib_indices=None,
#                              test_indices=None,
#                              normalize=False,
#                              save_path=None,
#                              dataset_name=None):
#     if metadata is None or calib_indices is None or test_indices is None:
#         raise ValueError("metadata, calib_indices, and test_indices are required")
#
#     # Convert indices to numpy if needed
#     if hasattr(calib_indices, 'numpy'):
#         calib_indices = calib_indices.numpy()
#     if hasattr(test_indices, 'numpy'):
#         test_indices = test_indices.numpy()
#
#     # Extract actual age groups from metadata
#     age_cal = metadata.loc[calib_indices, "Patient Age"].values
#     age_test = metadata.loc[test_indices, "Patient Age"].values
#
#     # Determine number of groups
#     num_groups = len(np.unique(y_train))
#
#     # Create indicator matrix: [intercept, 1{age∈G_0}, 1{age∈G_1}, ...]
#     features_cal = np.zeros((len(age_cal), num_groups + 1))
#     features_test = np.zeros((len(age_test), num_groups + 1))
#
#     features_cal[:, 0] = 1  # intercept
#     features_test[:, 0] = 1
#
#     for g in range(num_groups):
#         features_cal[:, g + 1] = (age_cal == g).astype(float)
#         features_test[:, g + 1] = (age_test == g).astype(float)
#
#     print(f"\n[INFO] Created metadata-based indicator features:")
#     print(f"  Shape - Cal: {features_cal.shape}, Test: {features_test.shape}")
#     print(f"  Format: [intercept, 1{{age∈G_0}}, 1{{age∈G_1}}, ...]")
#     print(f"  Using ACTUAL age groups from metadata (no learning)")
#
#     # Print group distribution
#     print("\n[DEBUG] True age group distribution (calibration):")
#     unique_cal, counts_cal = np.unique(age_cal, return_counts=True)
#     for g, c in zip(unique_cal, counts_cal):
#         print(f"  Group {g}: {c} samples ({100 * c / len(age_cal):.1f}%)")
#
#     return features_cal, features_test
#
#

