import os
import torch
import numpy as np
import pandas as pd
from data_config import DATASET_CONFIG
from scipy.special import softmax
from feature_io import load_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else x

def one_hot_fixed(codes: np.ndarray, K: int) -> np.ndarray:
    """One-hot with a fixed number of columns K (even if some classes are missing)."""
    codes = np.asarray(codes, dtype=int).ravel()
    out = np.zeros((len(codes), K), dtype=float)
    if len(codes) > 0:
        out[np.arange(len(codes)), np.clip(codes, 0, K - 1)] = 1.0
    return out

def bin_numeric_to_one_hot(cal_s: pd.Series, tst_s: pd.Series, n_bins: int = 5):
    """Equal-width bins from combined min/max, then fixed-K one-hot."""
    cal_s = pd.to_numeric(cal_s, errors="coerce")
    tst_s = pd.to_numeric(tst_s, errors="coerce")
    combined = pd.concat([cal_s, tst_s], ignore_index=True)

    vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or np.isclose(vmin, vmax):
        K = 1
        zeros_cal = np.zeros(len(cal_s), dtype=int)
        zeros_tst = np.zeros(len(tst_s), dtype=int)
        return one_hot_fixed(zeros_cal, K), one_hot_fixed(zeros_tst, K), (vmin, vmax)

    edges = np.linspace(vmin, vmax, n_bins + 1)  # bins: [e0,e1), [e1,e2), ...
    cal_codes = np.clip(np.digitize(cal_s.values, edges, right=False) - 1, 0, n_bins - 1)
    tst_codes = np.clip(np.digitize(tst_s.values, edges, right=False) - 1, 0, n_bins - 1)
    return one_hot_fixed(cal_codes, n_bins), one_hot_fixed(tst_codes, n_bins), (vmin, vmax)

def categorical_to_one_hot(cal_s: pd.Series, tst_s: pd.Series):
    """Factorize on combined (cal+test) so codes align, then fixed-K one-hot."""
    cal = cal_s.fillna("Unknown").astype(str)
    tst = tst_s.fillna("Unknown").astype(str)
    combined = pd.concat([cal, tst], ignore_index=True)
    codes_all, uniques = pd.factorize(combined, sort=True)
    K = len(uniques)
    cal_codes = codes_all[:len(cal)]
    tst_codes = codes_all[len(cal):]
    return one_hot_fixed(cal_codes, K), one_hot_fixed(tst_codes, K), list(uniques)

def compute_logits(features, model, batch_size=512):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    features = features.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            out.append(model(features[i:i + batch_size]).cpu())
    return torch.cat(out, dim=0)

def get_metadata_splits(metadata: pd.DataFrame, data: dict):
    """Return (train_meta, calib_meta, test_meta) aligned to lengths if no 'split' col."""
    if "split" in metadata.columns:
        train_meta = metadata[metadata["split"] == 0]
        calib_meta = metadata[metadata["split"] == 1]
        test_meta  = metadata[metadata["split"] == 2]
    else:
        n_tr = len(data["train"]["features"])
        n_ca = len(data["calib"]["features"])
        n_te = len(data["test"]["features"])
        train_meta = metadata.iloc[:n_tr]
        calib_meta = metadata.iloc[n_tr:n_tr + n_ca]
        test_meta  = metadata.iloc[n_tr + n_ca:n_tr + n_ca + n_te]
    return train_meta, calib_meta, test_meta



def load_split_dataset(dataset_name):
    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    data = {}
    base_path = cfg["features_base_path"]

    for split in ["train", "calib", "test"]:
        path = os.path.join(base_path, f"{dataset_name}_{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")

        features, logits, labels, _ = load_features(path)
        if hasattr(labels, "long"):
            labels = labels.long()
        elif hasattr(labels, "astype"):
            labels = labels.astype(np.int64)
        data[split] = {"features": features, "logits": logits, "labels": labels}

    if data["calib"]["logits"] is None:
        model = load_classifier(dataset_name).to(device).eval()
        for split in ["train", "calib", "test"]:
            data[split]["logits"] = compute_logits(data[split]["features"], model)
    meta_path = cfg.get("metadata_path")
    metadata = pd.read_csv(meta_path) if meta_path and os.path.exists(meta_path) else None
    return data, metadata


def create_feature_matrix(dataset_name, data, use_groups, use_logits,
                     add_additional_features, custom_bins, n_bins):
    if data is None:
        data = load_split_data(dataset_name)

    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cal, tst, metadata = data["calib"], data["test"], data["metadata"]

    if cal.get("logits") is None or tst.get("logits") is None:
        raise ValueError("[SOFT] logits requested but missing in data.")
    phi_cal = softmax(to_numpy(cal["logits"]), axis=1)
    phi_tst = softmax(to_numpy(tst["logits"]), axis=1)

    return phi_cal, phi_tst

# Convenience wrappers
def create_phi_chestx(use_groups: bool = True, use_logits: bool = False, add_features: bool = True):
    return create_feature_matrix(
        "ChestX",
        use_groups=use_groups,
        use_logits=use_logits,
        add_additional_features=add_features,
        n_bins=10,
    )

def create_phi_padchest(use_groups: bool = True, add_features: bool = True):
    return create_feature_matrix(
        "PadChest",
        use_groups=use_groups,
        add_additional_features=add_features,
        n_bins=10,
    )
