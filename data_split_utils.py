# import os
# import torch
# import numpy as np
# import pandas as pd
# from model_builder import load_classifier
# from feature_io import load_features
# # from utils import categorical_to_numeric, one_hot_encode
# from config import DATASET_CONFIG
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def compute_logits_fn(features, model, batch_size=512):
#     if not isinstance(features, torch.Tensor):
#         features = torch.tensor(features, dtype=torch.float32)
#     logits_list = []
#     with torch.no_grad():
#         for i in range(0, len(features), batch_size):
#             batch = features[i:i + batch_size].to(device)
#             logits_list.append(model(batch).cpu())
#     return torch.cat(logits_list, dim=0)
#
# def to_numpy(x):
#     return x.numpy() if hasattr(x, 'numpy') else x
#
# def load_split_data(dataset_name, compute_logits=True):
#     config = DATASET_CONFIG.get(dataset_name)
#     if not config:
#         raise ValueError(f"Unknown dataset: {dataset_name}")
#     data = {}
#     for split in ['train', 'calib', 'test']:
#         path = os.path.join(config['features_base_path'], f"{dataset_name}_{split}.pt")
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Missing feature file: {path}")
#
#         features, logits, labels = load_features(path)
#
#         if hasattr(labels, 'long'):
#             labels = labels.long()
#         elif hasattr(labels, 'astype'):
#             labels = labels.astype(np.int64)
#
#         data[split] = {'features': features, 'logits': logits,  'labels': labels }
#
#     if compute_logits and data['calib']['logits'] is None:
#         model = load_classifier(dataset_name)
#         for split in data:
#             data[split]['logits'] = compute_logits_fn(data[split]['features'], model)
#
#     metadata_path = config.get('metadata_path')
#     data['metadata'] = pd.read_csv(metadata_path) \
#         if metadata_path and os.path.exists(metadata_path) else None
#     return data
# ####???
# def get_metadata_splits(metadata, data):
#     if 'split' in metadata.columns:
#         train_meta = metadata[metadata['split'] == 0]
#         calib_meta = metadata[metadata['split'] == 1]
#         test_meta = metadata[metadata['split'] == 2]
#     else:
#         train_size = len(data['train']['features'])
#         calib_size = len(data['calib']['features'])
#         test_size = len(data['test']['features'])
#         train_meta = metadata.iloc[:train_size]
#         calib_meta = metadata.iloc[train_size:train_size + calib_size]
#         test_meta = metadata.iloc[train_size + calib_size:train_size + calib_size + test_size]
#     return train_meta, calib_meta, test_meta
#
#
# def create_phi_split(dataset_name, data=None, use_groups=True, use_logits=False,
#                      add_additional_features=False, n_bins=10):
#     """Create phi matrices for calibration and test splits using the correct approach."""
#     if data is None:
#         data = load_split_data(dataset_name)
#
#     config = DATASET_CONFIG.get(dataset_name)
#     if not config:
#         raise ValueError(f"Unknown dataset: {dataset_name}")
#
#     cal = data['calib']
#     tst = data['test']
#     metadata = data['metadata']
#
#     # Return logits if requested and available
#     if use_logits and cal['logits'] is not None:
#         return to_numpy(cal['logits']), to_numpy(tst['logits'])
#
#     # Use groups approach (matches the correct implementation)
#     if use_groups and metadata is not None and config.get('main_group'):
#         # DEBUG: Check for category consistency
#         main_group = config["main_group"]
#         if main_group not in metadata.columns:
#             cols = ", ".join(metadata.columns)
#             raise ValueError(f"Main group '{main_group}' not in metadata. Available: {cols}")
#
#         # Get metadata for each split
#         train_meta, calib_meta, test_meta = get_metadata_splits(metadata, data)
#         main_group = config["main_group"]
#
#         if main_group not in calib_meta.columns or main_group not in test_meta.columns:
#             raise ValueError(f"Main group '{main_group}' not in metadata.")
#
#         if pd.api.types.is_numeric_dtype(calib_meta[main_group]):
#             phi_cal, phi_test, (vmin, vmax) = bin_numeric_to_one_hot(
#                 calib_meta[main_group], test_meta[main_group], n_bins=n_bins
#             )
#             print(f"Using numeric group '{main_group}': {n_bins} bins, range=({vmin:.2f}, {vmax:.2f})")
#         else:
#             phi_cal, phi_test, cats = categorical_to_one_hot(
#                 calib_meta[main_group], test_meta[main_group]
#             )
#             print(f"Using categorical group '{main_group}': {len(cats)} categories")
#
#             # 2b) Optional additional features
#             if add_additional_features:
#                 for feat in (config.get("additional_features") or []):
#                     if feat not in calib_meta.columns or feat not in test_meta.columns:
#                         print(f"[WARN] Skipping missing feature '{feat}'.")
#                         continue
#                     try:
#                         if pd.api.types.is_numeric_dtype(calib_meta[feat]):
#                             f_cal, f_test, _ = bin_numeric_to_one_hot(
#                                 calib_meta[feat], test_meta[feat], n_bins=n_bins
#                             )
#                         else:
#                             f_cal, f_test, _ = categorical_to_one_hot(
#                                 calib_meta[feat], test_meta[feat]
#                             )
#                         # same width guaranteed by fixed-K encoder
#                         phi_cal = np.hstack([phi_cal, f_cal])
#                         phi_test = np.hstack([phi_test, f_test])
#                     except Exception as e:
#                         print(f"[WARN] Feature '{feat}' failed: {e}")
#
#             if phi_cal.shape[1] != phi_test.shape[1]:
#                 raise ValueError(f"Φ width mismatch: cal={phi_cal.shape}, test={phi_test.shape}")
#             print(f"Final Φ shapes: cal={phi_cal.shape}, test={phi_test.shape}")
#             return phi_cal, phi_test
#
#         # 3) Fallback: raw features
#         return to_numpy(cal["features"]), to_numpy(tst["features"])
#
#
# def one_hot_fixed(codes: np.ndarray, K: int) -> np.ndarray:
#     """One-hot with a fixed number of columns K (even if some classes are missing)."""
#     codes = np.asarray(codes, dtype=int).ravel()
#     out = np.zeros((len(codes), K), dtype=float)
#     if len(codes) > 0:
#         out[np.arange(len(codes)), np.clip(codes, 0, K-1)] = 1.0
#     return out
#
# def bin_numeric_to_one_hot(cal_s: pd.Series, tst_s: pd.Series, n_bins: int = 5):
#     """Equal-width bins from combined min/max, then fixed-K one-hot."""
#     combined = pd.concat([cal_s, tst_s], ignore_index=True).astype(float)
#     vmin, vmax = float(np.nanmin(combined)), float(np.nanmax(combined))
#
#     # constant/degenerate => single bin
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
#         K = 1
#         cal_codes = np.zeros(len(cal_s), dtype=int)
#         tst_codes = np.zeros(len(tst_s), dtype=int)
#         return one_hot_fixed(cal_codes, K), one_hot_fixed(tst_codes, K), (vmin, vmax)
#
#     edges = np.linspace(vmin, vmax, n_bins + 1)
#     cal_codes = np.clip(np.digitize(cal_s.astype(float), edges, right=False) - 1, 0, n_bins - 1)
#     tst_codes = np.clip(np.digitize(tst_s.astype(float), edges, right=False) - 1, 0, n_bins - 1)
#     return one_hot_fixed(cal_codes, n_bins), one_hot_fixed(tst_codes, n_bins), (vmin, vmax)
#
# def categorical_to_one_hot(cal_s: pd.Series, tst_s: pd.Series):
#     """Factorize on combined (cal+test) so codes align, then fixed-K one-hot."""
#     cal = cal_s.fillna("Unknown").astype(str)
#     tst = tst_s.fillna("Unknown").astype(str)
#     combined = pd.concat([cal, tst], ignore_index=True)
#     codes_all, uniques = pd.factorize(combined, sort=True)
#     K = len(uniques)
#     cal_codes = codes_all[:len(cal)]
#     tst_codes = codes_all[len(cal):]
#     return one_hot_fixed(cal_codes, K), one_hot_fixed(tst_codes, K), list(uniques)
#
#
# def create_phi_chestx(use_groups=True, use_logits=False, add_features=True):
#     return create_phi_split('ChestX', use_groups=use_groups, use_logits=use_logits,
#                             add_additional_features=False, n_bins=10)
#
# def create_phi_padchest(use_groups=True, add_features=True):
#     return create_phi_split('PadChest', use_groups=use_groups,
#                             add_additional_features=False, n_bins=10)
#

import os
import torch
import numpy as np
import pandas as pd
from model_builder import load_classifier
from feature_io import load_features
from config import DATASET_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- helpers (no leading underscores) --------

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

# -------- public API --------

def load_split_data(dataset_name: str, compute_missing_logits: bool = True) -> dict:
    """Load split features/logits/labels + metadata for a dataset."""
    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = {}
    base = cfg["features_base_path"]
    for split in ["train", "calib", "test"]:
        path = os.path.join(base, f"{dataset_name}_{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")
        features, logits, labels = load_features(path)

        # normalize label dtype
        if hasattr(labels, "long"):
            labels = labels.long()
        elif hasattr(labels, "astype"):
            labels = labels.astype(np.int64)

        data[split] = {"features": features, "logits": logits, "labels": labels}

    # compute logits on-the-fly if absent
    if compute_missing_logits and data["calib"]["logits"] is None:
        model = load_classifier(dataset_name).to(device).eval()
        for split in ["train", "calib", "test"]:
            data[split]["logits"] = compute_logits(data[split]["features"], model)

    # metadata (optional)
    meta_path = cfg.get("metadata_path")
    data["metadata"] = pd.read_csv(meta_path) if meta_path and os.path.exists(meta_path) else None
    return data

def create_phi_split(
    dataset_name: str,
    data: dict | None = None,
    use_groups: bool = True,
    use_logits: bool = False,
    add_additional_features: bool = False,
    custom_bins = None,
    n_bins: int = 10,
):
    """
    Build Φ for calibration/test:
      - if use_logits: Φ := logits
      - elif use_groups: Φ := one-hot of main_group (numeric binned or categorical)
        and optionally append additional metadata features with the same encoding
      - else: Φ := raw features
    """
    if data is None:
        data = load_split_data(dataset_name)

    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cal, tst, metadata = data["calib"], data["test"], data["metadata"]

    # 1) logits path
    if use_logits and cal["logits"] is not None:
        return to_numpy(cal["logits"]), to_numpy(tst["logits"])

    # 2) group-based path
    if use_groups and metadata is not None and cfg.get("main_group"):
        main_group = cfg["main_group"]
        train_meta, calib_meta, test_meta = get_metadata_splits(metadata, data)

        if main_group not in calib_meta.columns or main_group not in test_meta.columns:
            raise ValueError(f"Main group '{main_group}' not found in metadata")

        series_cal = calib_meta[main_group]
        series_tst = test_meta[main_group]

        # Case 1: numeric + custom bins
        if pd.api.types.is_numeric_dtype(series_cal) and custom_bins is not None:
            phi_cal = pd.cut(series_cal, bins=custom_bins, right=False, include_lowest=True).cat.codes
            phi_tst = pd.cut(series_tst, bins=custom_bins, right=False, include_lowest=True).cat.codes

            phi_cal = one_hot_fixed(phi_cal, len(custom_bins) - 1)
            phi_tst = one_hot_fixed(phi_tst, len(custom_bins) - 1)
            print(f"[INFO] Using custom bins for '{main_group}': {custom_bins}")

        # Case 2: numeric + no custom bins
        elif pd.api.types.is_numeric_dtype(series_cal):
            combined = pd.concat([series_cal, series_tst], ignore_index=True)
            binned, bins = pd.qcut(combined, q=n_bins, duplicates="drop", retbins=True)
            codes = binned.cat.codes.to_numpy()
            phi_cal = one_hot_fixed(codes[:len(series_cal)], len(bins) - 1)
            phi_tst = one_hot_fixed(codes[len(series_cal):], len(bins) - 1)
            print(f"[INFO] Using quantile bins for '{main_group}': {bins}")

        # Case 3: categorical
        else:
            phi_cal, phi_tst, cats = categorical_to_one_hot(series_cal, series_tst)
            print(f"[INFO] Using categorical group '{main_group}': {len(cats)} categories")

        return phi_cal, phi_tst

    # Default: raw features
    return to_numpy(cal["features"]), to_numpy(tst["features"])

    #
    #     # Main group encoding
    #     if pd.api.types.is_numeric_dtype(calib_meta[main_group]) and custom_bins is not None:
    #
    #         phi_cal, phi_test, (vmin, vmax) = bin_numeric_to_one_hot(
    #             calib_meta[main_group], test_meta[main_group], n_bins=n_bins
    #         )
    #         print(f"[INFO] Using numeric group '{main_group}': {n_bins} bins, range=({vmin:.2f}, {vmax:.2f})")
    #     else:
    #         phi_cal, phi_test, cats = categorical_to_one_hot(
    #             calib_meta[main_group], test_meta[main_group]
    #         )
    #         print(f"[INFO] Using categorical group '{main_group}': {len(cats)} categories")
    #
    #     # Optional additional metadata features
    #     if add_additional_features:
    #         for feat in (cfg.get("additional_features") or []):
    #             if feat not in calib_meta.columns or feat not in test_meta.columns:
    #                 print(f"[WARN] Skipping missing feature '{feat}'.")
    #                 continue
    #             try:
    #                 if pd.api.types.is_numeric_dtype(calib_meta[feat]):
    #                     f_cal, f_test, _ = bin_numeric_to_one_hot(
    #                         calib_meta[feat], test_meta[feat], n_bins=n_bins
    #                     )
    #                 else:
    #                     f_cal, f_test, _ = categorical_to_one_hot(
    #                         calib_meta[feat], test_meta[feat]
    #                     )
    #                 phi_cal = np.hstack([phi_cal, f_cal])
    #                 phi_test = np.hstack([phi_test, f_test])
    #             except Exception as e:
    #                 print(f"[WARN] Feature '{feat}' failed: {e}")
    #
    #     if phi_cal.shape[1] != phi_test.shape[1]:
    #         raise ValueError(f"Φ width mismatch: cal={phi_cal.shape}, test={phi_test.shape}")
    #
    #     print(f"[INFO] Final Φ shapes: cal={phi_cal.shape}, test={phi_test.shape}")
    #     return phi_cal, phi_test
    #
    # # 3) fallback: raw features
    # print("[INFO] Using raw features for Φ.")
    # return to_numpy(cal["features"]), to_numpy(tst["features"])

# Convenience wrappers
def create_phi_chestx(use_groups: bool = True, use_logits: bool = False, add_features: bool = True):
    return create_phi_split(
        "ChestX",
        use_groups=use_groups,
        use_logits=use_logits,
        add_additional_features=add_features,
        n_bins=10,
    )

def create_phi_padchest(use_groups: bool = True, add_features: bool = True):
    return create_phi_split(
        "PadChest",
        use_groups=use_groups,
        add_additional_features=add_features,
        n_bins=10,
    )
