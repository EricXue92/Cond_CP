import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def bin_numeric_variable(values, custom_bins=None, n_bins=5):
    if custom_bins is not None:
        binned = pd.cut(values, bins=custom_bins, right=False, include_lowest=True)
    else:
        binned, bins = pd.qcut(values, q=n_bins, duplicates='drop', retbins=True)
        print(f"[INFO] Automatically binned into {len(bins)-1} quantile bins: {np.round(bins, 1)}")
    return binned.astype(str)

def evaluate_feature_group_correlation(features, metadata, group_col, dataset_name="dataset",
    save_dir="Figures",  custom_bins=None, n_bins=5, sample_tsne=2000,  random_state=42, show_plot=False,):

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[INFO] Evaluating feature–group correlation for '{dataset_name}' — group: '{group_col}'")
    if torch.is_tensor(features):
        X = features.cpu().numpy()
    else:
        X = np.asarray(features)

    X = StandardScaler().fit_transform(X)

    if group_col not in metadata.columns:
        raise ValueError(f"[ERROR] Column '{group_col}' not found in metadata")

    col_values = metadata[group_col].dropna()

    if pd.api.types.is_numeric_dtype(col_values):
        binned_groups = bin_numeric_variable(col_values, custom_bins=custom_bins, n_bins=n_bins)
        group_labels = binned_groups
        print(f"[INFO] Converted numeric '{group_col}' into categorical bins.")
    else:
        group_labels = col_values.astype(str)

    # Filter to non-missing samples
    valid_idx = col_values.index
    X = X[valid_idx]
    group_labels = group_labels.loc[valid_idx]

    # --- encode + train ---
    le = LabelEncoder()
    y = le.fit_transform(group_labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = LogisticRegressionCV(
        max_iter=2000, cv=5, n_jobs=-1, multi_class="multinomial", random_state=random_state
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results = {
        "dataset": dataset_name,
        "group_col": group_col,
        "num_groups": len(le.classes_),
        "group_names": list(le.classes_),
        "accuracy": acc,
    }

    print(f"[RESULT] → {group_col}: {acc*100:.2f}% accuracy across {len(le.classes_)} groups")

    # --- optional AUC for binary groups ---
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print(f"           ROC-AUC: {auc:.3f}")
        results["auc"] = auc

    # --- visualization (PCA + t-SNE) ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab20", s=5)
    plt.title(f"{dataset_name}: PCA by {group_col}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name}_{group_col}_pca.png", dpi=300)
    plt.close()

    if X.shape[0] > sample_tsne:
        idx = np.random.choice(X.shape[0], sample_tsne, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state)
    X_tsne = tsne.fit_transform(X_sub)
    plt.figure(figsize=(7, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub, cmap="tab20", s=5)
    plt.title(f"{dataset_name}: t-SNE by {group_col}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name}_{group_col}_tsne.png", dpi=300)
    plt.close()

    print(f"[INFO] Saved PCA/t-SNE visualizations to {save_dir}/")
    return results

# custom_bins = [0, 18, 30, 50, 70, 100]
# results = evaluate_feature_group_correlation(
#     features, metadata, group_col="Patient Age",
#     dataset_name="NIH",
#     custom_bins=custom_bins,
# )