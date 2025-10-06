import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def augment_features(X, metadata):
    """Concatenate DenseNet features with normalized age and one-hot gender."""
    X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)

    age = (metadata['Patient Age'].values / 100.0).reshape(-1, 1)
    gender = pd.get_dummies(metadata['Patient Gender'], drop_first=False).values

    X_aug = np.concatenate([X_np, age, gender], axis=1)
    print(f"[INFO] Augmented features: {X_np.shape[1]} → {X_aug.shape[1]}")
    return X_aug


def find_best_regularization(X_train, y_train, use_balancing):
    """Find optimal C parameter using cross-validation."""
    cv_model = LogisticRegressionCV(
        Cs=np.logspace(-4, 1, 30),
        cv=5,
        max_iter=5000,
        class_weight='balanced' if use_balancing else None,
        random_state=42,
        n_jobs=-1
    )
    cv_model.fit(X_train, y_train)
    return cv_model.C_[0]


def train_predictor(X_train, X_cal, X_test, y_train, y_cal,
                    meta_train, meta_cal, meta_test, target):
    """Train logistic regression to predict age groups or gender."""
    print(f"\n{'=' * 70}\nTraining {target.upper()} predictor\n{'=' * 70}")

    # Augment and normalize features
    X_train_aug = augment_features(X_train, meta_train)
    X_cal_aug = augment_features(X_cal, meta_cal)
    X_test_aug = augment_features(X_test, meta_test)

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_aug)
    X_cal_norm = scaler.transform(X_cal_aug)
    X_test_norm = scaler.transform(X_test_aug)

    # Determine if class balancing is needed
    _, counts = np.unique(y_train, return_counts=True)
    use_balancing = (counts.min() / counts.max()) < 0.3

    # Find best C and train final model
    best_c = find_best_regularization(X_train_norm, y_train, use_balancing)
    print(f"Best C: {best_c:.6f}")

    model = LogisticRegression(
        C=best_c,
        max_iter=5000,
        class_weight='balanced' if use_balancing else None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_norm, y_train)

    # Generate probability features
    phi_cal = model.predict_proba(X_cal_norm)
    phi_test = model.predict_proba(X_test_norm)

    # Report performance
    train_acc = model.score(X_train_norm, y_train)
    cal_acc = accuracy_score(y_cal, model.predict(X_cal_norm))
    print(f"\nTrain Acc: {train_acc:.4f} | Cal Acc: {cal_acc:.4f}")

    return phi_cal, phi_test


def prepare_labels(metadata, train_idx, cal_idx, label_type, age_bins=[0, 40, 65, 100]):
    """Prepare age group or gender labels from metadata."""
    if label_type == 'age':
        # Create categorical bins with string labels for plotting
        age_binned_cat = pd.cut(
            metadata['Patient Age'],
            bins=age_bins,
            right=False,
            include_lowest=True
        )
        # Store string representation for plotting
        metadata['Patient Age_binned'] = age_binned_cat.astype(str)
        # Store numeric codes for training
        metadata['age_group'] = age_binned_cat.cat.codes
        label_col = 'age_group'
        print(f"\nAge group distribution (train):")
    else:  # gender
        metadata['gender_encoded'] = metadata['Patient Gender'].map({'M': 0, 'F': 1})
        label_col = 'gender_encoded'
        print(f"\nGender distribution (train):")

    y_train = metadata.loc[train_idx, label_col].values
    y_cal = metadata.loc[cal_idx, label_col].values

    print(pd.Series(y_train).value_counts().sort_index())

    # Validate that both splits have at least 2 classes
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Training set has only {len(np.unique(y_train))} class(es). Need at least 2.")
    if len(np.unique(y_cal)) < 2:
        raise ValueError(f"Calibration set has only {len(np.unique(y_cal))} class(es). Need at least 2.")

    return y_train, y_cal


def create_phi_features_for_NIH(data, metadata, age_bins=[0, 40, 65, 100]):
    """Create phi features for age groups and gender from NIH dataset."""
    # Extract features and metadata by split
    X_train, X_cal, X_test = (data[s]["features"] for s in ["train", "calib", "test"])
    meta_splits = {i: metadata[metadata["split"] == i].reset_index(drop=True)
                   for i in range(3)}
    meta_train, meta_cal, meta_test = meta_splits[0], meta_splits[1], meta_splits[2]

    results = {}

    # Train age predictor
    print(f"\n{'=' * 70}\nCREATING AGE GROUP PHI FEATURES\n{'=' * 70}")
    y_age_train, y_age_cal = prepare_labels(
        metadata, meta_train.index, meta_cal.index, 'age', age_bins
    )
    results['age'] = train_predictor(
        X_train, X_cal, X_test, y_age_train, y_age_cal,
        meta_train, meta_cal, meta_test, 'age'
    )

    # Train gender predictor
    print(f"\n{'=' * 70}\nCREATING GENDER PHI FEATURES\n{'=' * 70}")
    y_gender_train, y_gender_cal = prepare_labels(
        metadata, meta_train.index, meta_cal.index, 'gender'
    )
    results['gender'] = train_predictor(
        X_train, X_cal, X_test, y_gender_train, y_gender_cal,
        meta_train, meta_cal, meta_test, 'gender'
    )

    results['metadata'] = metadata
    return results