
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cvxpy")

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils import set_seed
from conformal_scores import compute_conformity_scores
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from conditionalconformal.condconf import CondConf

set_seed(42)

DATASET_CONFIG = {"NIH": {"group_cols": ["age_group", "Patient Gender"] } ,}
AGE_BINS   = [0, 18, 40, 60, 100]
# AGE_LABELS = ["0-18", "18-40", "40-60", "60-80", "80-100"]
AGE_LABELS = ["0-18", "18-40", "40-60", "60-100"]

def load_test_data(checkpoint_dir, dataset="NIH"):
    path = os.path.join(checkpoint_dir, f"test_data_{dataset}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data not found: {path}")
    data = torch.load(path)
    logits,labels,metadata = data["logits"], data["labels"], data["metadata"]
    if isinstance(logits, torch.Tensor): logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    print(f"[INFO] Loaded: logits {logits.shape}, labels {labels.shape}, metadata {len(metadata)} rows")
    return logits, labels, metadata

def prepare_age_groups(metadata, col="Patient Age"):
    md = metadata.copy()
    md[col] = pd.to_numeric(md[col], errors="coerce")
    md["age_group"] = (pd.cut(md[col], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True).astype(str).fillna("Unknown"))
    print("[INFO] Age distribution:")
    counts = md["age_group"].value_counts().reindex(AGE_LABELS, fill_value=0)
    for g, c in counts.items():
        print(f"  {g:>7}: {c:6d} ({100 * c / len(md):5.1f}%)")
    return md

def split_calib_test(logits, metadata, stratify_col="age_group"):
    stratify = pd.factorize(metadata[stratify_col])[0] if stratify_col in metadata.columns else None
    idx = np.arange(len(logits))
    calib_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=42, stratify=stratify)
    print(f"[INFO] Split: calibration={len(calib_idx)}, test={len(test_idx)}")
    return calib_idx, test_idx

   # Filter to samples with at least one positive label
def filter_nonempty(labels, probs, metadata):
    mask = labels.sum(axis=1) > 0
    return labels[mask], probs[mask], metadata.iloc[mask].reset_index(drop=True), mask

def make_aps_score_fn(all_probs):

    def score_fn(x_idx, y_labels):
        x = np.atleast_1d(x_idx).astype(int).flatten()
        y = np.atleast_2d(y_labels)
        if y.shape[0] == 1 and len(x) > 1:
            y = np.repeat(y, len(x), axis=0)
        if y.shape[0] != len(x):
            raise ValueError(f"Shape mismatch: x={len(x)}, y={y.shape}")
        scores = np.zeros(len(x))
        for i, idx in enumerate(x):
            probs = all_probs[idx].ravel()
            positives = np.flatnonzero(y[i] > 0.5)
            if positives.size == 0:
                scores[i] = 0.0
                continue
            order = np.argsort(-probs, kind="stable")
            inv_ranks = np.empty_like(order)
            inv_ranks[order] = np.arange(len(probs))
            K = inv_ranks[positives].max() + 1
            scores[i] = probs[order[:K]].sum()
        return scores

    return score_fn


# def make_aps_score_inverse_fn(all_probs):
#     """
#     Inverse score function: given threshold, return prediction set.
#     Used for predict_naive() to generate actual prediction sets.
#     """
#
#     def score_inv_fn(threshold, x_index):
#         idx = int(x_index)
#         probs = all_probs[idx].ravel()
#         order = np.argsort(-probs, kind="stable")
#         cumsum = np.cumsum(probs[order])
#
#         # Find how many labels needed to reach threshold
#         k = int(np.searchsorted(cumsum, threshold, side="left")) + 1
#         k = min(max(k, 1), len(probs))
#
#         # Create multi-hot prediction set
#         prediction = np.zeros_like(probs, dtype=int)
#         prediction[order[:k]] = 1
#         return prediction
#
#     return score_inv_fn


# =============================================================================
# Group Features (Basis Functions)
# =============================================================================

def composite_keys(metadata, group_cols):
    """Create composite group identifiers like 'age_group=0-18 | Patient Gender=F'."""
    if not group_cols:
        return np.array(["__ALL__"] * len(metadata))

    parts = []
    for col in group_cols:
        if col in metadata.columns:
            parts.append(metadata[col].astype(str).map(lambda v: f"{col}={v}"))
        else:
            parts.append(pd.Series([f"{col}=NA"] * len(metadata), index=metadata.index))

    return pd.concat(parts, axis=1).agg(" | ".join, axis=1).to_numpy()


def make_phi_fn(group_keys_all):
    """Create one-hot encoding basis function for groups."""
    unique = np.unique(group_keys_all)
    key_to_idx = {k: i for i, k in enumerate(unique)}
    identity = np.eye(len(unique))

    def phi_fn(x_indices):
        # Critical: flatten to handle CondConf's reshape(1, -1)
        x = np.atleast_1d(x_indices).astype(int).flatten()
        return identity[[key_to_idx[group_keys_all[i]] for i in x]]

    print(f"[INFO] Groups: {len(unique)} unique combinations")
    return phi_fn

def analyze_group_coverage(coverage_split, coverage_cond, metadata_test, group_cols, alpha):
    """Print coverage analysis by group."""
    print(f"\n{'=' * 70}")
    print(f"Group Coverage Analysis (Target: {1 - alpha:.1%})")
    print(f"{'=' * 70}")

    keys = composite_keys(metadata_test, group_cols)

    for group in sorted(np.unique(keys)):
        mask = keys == group
        n = int(np.sum(mask))

        split_cov = coverage_split[mask].mean()
        cond_cov = coverage_cond[mask].mean()
        gap = cond_cov - (1 - alpha)

        print(f"  {group:<48} | n={n:5d} | split={split_cov:.3f} | "
              f"cond={cond_cov:.3f} | Δ={gap:+.3f}")

    print(f"{'=' * 70}\n")


def save_and_plot(coverage_split, coverage_cond, metadata_test, dataset_name, alpha):
    """Save results and plot miscoverage (ordered Age, Gender panels)."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("Figures", exist_ok=True)

    saved = {}

    # ---- Age group panel (ordered bins) ----
    if "age_group" in metadata_test.columns:
        # Enforce proper categorical order
        order = ["0-18", "18-40", "40-60", "60-100"]
        metadata_test["age_group"] = pd.Categorical(metadata_test["age_group"], categories=order, ordered=True)

        df_age = build_cov_df(
            coverage_split,
            coverage_cond,
            metadata_test["age_group"],
            group_name="Age Group",
        )
        save_csv(df_age, f"{dataset_name}_age_group", "results")
        saved["age"] = f"results/{dataset_name}_age_group.csv"

    # ---- Gender panel ----
    gender_col = "Patient Gender" if "Patient Gender" in metadata_test.columns else (
        "Sex" if "Sex" in metadata_test.columns else None
    )
    if gender_col:
        df_gender = build_cov_df(
            coverage_split,
            coverage_cond,
            metadata_test[gender_col],
            group_name="Patient Gender",
        )
        save_csv(df_gender, f"{dataset_name}_gender", "results")
        saved["gender"] = f"results/{dataset_name}_gender.csv"

    # ---- Plot two panels (gender | age) ----
    if "age" in saved and "gender" in saved:
        plot_miscoverage(
            saved["gender"],  # left: gender
            saved["age"],     # right: ordered age
            alpha,
            "Figures",
            f"{dataset_name}_miscoverage"
        )
    else:
        print("[WARNING] Missing age or gender CSVs for plotting.")

# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Conditional Conformal Prediction with group fairness"
    )
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (1-alpha = target coverage)")
    parser.add_argument("--dataset_name", default="NIH", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--composite_groups", action="store_true",
                        help="Use both age and gender jointly (default: age only)")
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset_name]
    group_cols = config["group_cols"] if args.composite_groups else config["group_cols"][:1]

    print(f"\n{'=' * 70}")
    print(f"Conditional Conformal Prediction: {args.dataset_name}")
    print(f"Target coverage: {1 - args.alpha:.1%} | Groups: {', '.join(group_cols)}")
    print(f"{'=' * 70}\n")

    # =========================================================================
    # 1. Load data
    # =========================================================================
    logits, labels, metadata = load_test_data(args.checkpoint_dir, args.dataset_name)
    metadata = prepare_age_groups(metadata)

    # =========================================================================
    # 2. Split into calibration and test
    # =========================================================================
    calib_idx, test_idx = split_calib_test(logits, metadata, "age_group")
    meta_cal = metadata.iloc[calib_idx].reset_index(drop=True)
    meta_test = metadata.iloc[test_idx].reset_index(drop=True)

    print("[INFO] Computing probabilities...")
    _, _, probs_cal, probs_test = compute_conformity_scores(
        logits[calib_idx], logits[test_idx],
        labels[calib_idx], labels[test_idx]
    )

    y_cal, probs_cal_filtered, meta_cal_filtered, cal_mask = filter_nonempty(
        labels[calib_idx], probs_cal, meta_cal
    )
    y_test, probs_test_filtered, meta_test_filtered, test_mask = filter_nonempty(
        labels[test_idx], probs_test, meta_test
    )

    print(f"[INFO] Filtered non-empty labels:")
    print(f"  Calibration: {len(y_cal)}/{len(cal_mask)} ({100 * len(y_cal) / len(cal_mask):.1f}%)")
    print(f"  Test:        {len(y_test)}/{len(test_mask)} ({100 * len(y_test) / len(test_mask):.1f}%)")


    all_probs = np.vstack([probs_cal_filtered, probs_test_filtered])
    score_fn = make_aps_score_fn(all_probs)
    # score_inv_fn = make_aps_score_inverse_fn(all_probs)

    group_keys_all = np.concatenate([
        composite_keys(meta_cal_filtered, group_cols),
        composite_keys(meta_test_filtered, group_cols)
    ])
    phi_fn = make_phi_fn(group_keys_all)

    n_cal = len(probs_cal_filtered)
    x_cal = np.arange(n_cal)
    x_test = np.arange(n_cal, n_cal + len(probs_test_filtered))

    cal_scores = score_fn(x_cal, y_cal)
    test_scores = score_fn(x_test, y_test)

    q_level = np.ceil((n_cal + 1) * (1 - args.alpha)) / n_cal
    q_split = np.quantile(cal_scores, q_level)
    coverage_split = (test_scores <= q_split).astype(float)

    print(f"[INFO] Split conformal: threshold={q_split:.4f}, "
          f"coverage={coverage_split.mean():.4f} (target={1 - args.alpha:.4f})")

    print("[INFO] Running conditional conformal...")

    condconf = CondConf(score_fn=score_fn,Phi_fn=phi_fn,quantile_fn=None,infinite_params={})
    condconf.setup_problem(x_calib=x_cal, y_calib=y_cal)
    coverage_cond = condconf.verify_coverage(x=x_test, y=y_test, quantile=args.alpha)
    print(f"[INFO] Conditional coverage: {coverage_cond.mean():.4f}")

    # try:
    #     example_idx = x_test[:5]
    #     prediction_sets = condconf.predict_naive(
    #         quantile=args.alpha,
    #         x_array=example_idx,
    #         score_inv_fn=score_inv_fn
    #     )
    #     print(f"[INFO] Generated {len(prediction_sets)} example prediction sets")
    # except Exception as e:
    #     print(f"[WARNING] predict_naive not available: {e}")

    analyze_group_coverage(coverage_split, coverage_cond,meta_test_filtered, group_cols, args.alpha)
    save_and_plot(coverage_split, coverage_cond, meta_test_filtered, args.dataset_name, args.alpha)



if __name__ == "__main__":
    main()