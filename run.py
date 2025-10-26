import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="cvxpy")

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils import set_seed
from conformal_scores import compute_conformity_scores # only for probs
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from conditionalconformal.condconf import CondConf

set_seed(42)

DATASET_CONFIG = {"NIH": {"group_cols": ["age_group", "Patient Gender"] } ,}
# AGE_BINS   = [0, 18, 40, 60, 80,  100]
# AGE_LABELS = ["0-18", "18-40", "40-60", "60-80", "80-100"]

AGE_BINS   = [0, 18, 40, 60, 100]
AGE_LABELS = ["0-18", "18-40", "40-60", "60-100"]

def load_test_data(checkpoint_dir, dataset="NIH"):
    path = os.path.join(checkpoint_dir, f"test_data_{dataset}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data not found: {path}")
    data = torch.load(path)
    logits = data["logits"].cpu().numpy() if isinstance(data["logits"], torch.Tensor) else data["logits"]
    labels = data["labels"].cpu().numpy() if isinstance(data["labels"], torch.Tensor) else data["labels"]
    metadata = data["metadata"]
    print(f"[INFO] Loaded: {logits.shape[0]} samples, {logits.shape[1]} classes")
    return logits, labels, metadata

def prepare_age_groups(metadata, col="Patient Age"):
    metadata= metadata.copy()
    metadata[col] = pd.to_numeric(metadata[col], errors="coerce")
    metadata["age_group"] = (pd.cut(metadata[col], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True).astype(str).fillna("Unknown"))
    print("[INFO] Age distribution:")
    counts = metadata["age_group"].value_counts().reindex(AGE_LABELS, fill_value=0)
    for g, c in counts.items():
        print(f"  {g:>7}: {c:6d} ({100 * c / len(metadata):5.1f}%)")
    return metadata

def split_calib_test(logits, metadata, stratify_col="age_group", test_size=0.5, seed=42):
    stratify = pd.factorize(metadata[stratify_col])[0] if stratify_col in metadata.columns else None
    idx = np.arange(len(logits))
    calib_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=stratify)
    print(f"[INFO] Split: calibration={len(calib_idx)}, test={len(test_idx)}")
    return calib_idx, test_idx

def filter_nonempty(labels, probs, metadata):
    """Keep only samples with at least one positive label."""
    mask = labels.sum(axis=1) > 0
    return labels[mask], probs[mask], metadata.iloc[mask].reset_index(drop=True)

def create_aps_score_fn(all_probs):

    def score_fn(x_idx, y_labels):
        x_idx = np.atleast_1d(x_idx).astype(int).flatten()
        y_labels = np.atleast_2d(y_labels)

        if y_labels.shape[0] == 1 and len(x_idx) > 1:
            y_labels = np.repeat(y_labels, len(x_idx), axis=0)

        scores = np.zeros(len(x_idx))
        for i, idx in enumerate(x_idx):
            probs = all_probs[idx].ravel()
            positives = np.where(y_labels[i] > 0.5)[0]

            if len(positives) == 0: continue # no positive labels

            # Sort by probability (descending)
            order = np.argsort(-probs)

            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(probs))

            # Sum probabilities up to highest-ranked positive
            K = ranks[positives].max() + 1
            scores[i] = probs[order[:K]].sum()
        return scores

    return score_fn

def create_group_keys(metadata, group_cols):
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

def create_phi_fn(group_keys_all):

    unique = np.unique(group_keys_all)
    key_to_idx = {k: i for i, k in enumerate(unique)}
    identity = np.eye(len(unique))
    def phi_fn(x_indices):
        x = np.atleast_1d(x_indices).astype(int).flatten()
        return identity[[key_to_idx[group_keys_all[i]] for i in x]]

    print(f"[INFO] Groups: {len(unique)} unique combinations")
    return phi_fn

# def analyze_group_coverage(coverage_split, coverage_cond, metadata_test, group_cols, alpha):
#     keys = create_group_keys(metadata_test, group_cols)
#     target = 1 - alpha
#     for group in sorted(np.unique(keys)):
#         mask = keys == group
#         n = int(np.sum(mask))
#         split_cov = coverage_split[mask].mean()
#         cond_cov = coverage_cond[mask].mean()
#         gap = cond_cov - target
#         print(f"{group:<50} | {n:5d} | {split_cov:.3f} | {cond_cov:.3f} | {gap:+.3f}")

def analyze_group_coverage(coverage_split, coverage_cond, metadata_test, group_cols, alpha):
    """Print coverage statistics by group with clear column headers."""
    keys = create_group_keys(metadata_test, group_cols)
    target = 1 - alpha

    # Print header
    print(f"\n{'=' * 95}")
    print(f"COVERAGE ANALYSIS BY GROUP (Target: {target:.1%})")
    print(f"{'=' * 95}")
    print(f"{'Group':<50} | {'n':>6} | {'Split':>7} | {'Cond':>7} | {'Gap':>8}")
    print(f"{'-' * 50}-+--------+---------+---------+----------+--------")

    # Print each group
    for group in sorted(np.unique(keys)):
        mask = keys == group
        n = int(np.sum(mask))
        split_cov = coverage_split[mask].mean()
        cond_cov = coverage_cond[mask].mean()
        gap = cond_cov - target

        print(f"{group:<50} | {n:6d} | {split_cov:7.3f} | {cond_cov:7.3f} | {gap:+8.3f} ")

    # Print summary
    print(f"{'-' * 50}-+--------+---------+---------+----------+--------")
    overall_split = coverage_split.mean()
    overall_cond = coverage_cond.mean()
    overall_gap = overall_cond - target
    print(f"{'OVERALL':<50} | {len(keys):6d} | {overall_split:7.3f} | {overall_cond:7.3f} | {overall_gap:+8.3f} |")
    print(f"{'=' * 95}")

    # Print interpretation
    n_below = sum((coverage_cond[keys == g].mean() < target - 0.01) for g in np.unique(keys))
    n_groups = len(np.unique(keys))
    print(f"\nSummary: {n_groups - n_below}/{n_groups} groups meet target coverage (≥{target:.1%})")


def save_and_plot(coverage_split, coverage_cond, metadata, dataset_name, alpha):
    os.makedirs("results", exist_ok=True)
    os.makedirs("Figures", exist_ok=True)
    saved = {}
    if "age_group" in metadata.columns:
        metadata["age_group"] = pd.Categorical(metadata["age_group"], categories=AGE_LABELS, ordered=True)
        df_age = build_cov_df(coverage_split,coverage_cond,metadata["age_group"],group_name="Age Group",)
        save_csv(df_age, f"{dataset_name}_age_group", "results")
        saved["age"] = f"results/{dataset_name}_age_group.csv"

    gender_col = "Patient Gender" if "Patient Gender" in metadata.columns else ("Sex" if "Sex" in metadata.columns else None)
    if gender_col:
        df_gender = build_cov_df(coverage_split,coverage_cond,metadata[gender_col], group_name="Patient Gender",)
        save_csv(df_gender, f"{dataset_name}_gender", "results")
        saved["gender"] = f"results/{dataset_name}_gender.csv"

    if "age" in saved and "gender" in saved:
        plot_miscoverage(saved["gender"],  saved["age"], alpha,"Figures",f"{dataset_name}_miscoverage")


def run_conformal_prediction(logits, labels, metadata, group_cols, alpha, seed=42):
    """Run split and conditional conformal prediction."""

    # Split data
    cal_idx, test_idx = split_calib_test(logits, metadata, seed=seed)
    meta_cal = metadata.iloc[cal_idx].reset_index(drop=True)
    meta_test = metadata.iloc[test_idx].reset_index(drop=True)

    # Compute probabilities
    print("[INFO] Computing probabilities...")
    _, _, probs_cal, probs_test = compute_conformity_scores(
        logits[cal_idx], logits[test_idx], labels[cal_idx], labels[test_idx]
    )

    # Filter non-empty labels
    y_cal, probs_cal, meta_cal = filter_nonempty(labels[cal_idx], probs_cal, meta_cal)
    y_test, probs_test, meta_test = filter_nonempty(labels[test_idx], probs_test, meta_test)

    print(f"[INFO] After filtering: {len(y_cal)} calibration, {len(y_test)} test samples")

    # Setup score and basis functions
    all_probs = np.vstack([probs_cal, probs_test])
    score_fn = create_aps_score_fn(all_probs)

    group_keys = np.concatenate([
        create_group_keys(meta_cal, group_cols),
        create_group_keys(meta_test, group_cols)
    ])
    phi_fn = create_phi_fn(group_keys)

    # Index arrays
    n_cal = len(probs_cal)
    x_cal = np.arange(n_cal)
    x_test = np.arange(n_cal, n_cal + len(probs_test))

    # Compute conformity scores
    cal_scores = score_fn(x_cal, y_cal)
    test_scores = score_fn(x_test, y_test)

    # Split conformal
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_split = np.quantile(cal_scores, q_level)
    coverage_split = (test_scores <= q_split).astype(float)

    print(f"\n[RESULT] Split conformal:")
    print(f"  Threshold: {q_split:.4f}")
    print(f"  Coverage:  {coverage_split.mean():.4f} (target: {1 - alpha:.4f})")

    # Conditional conformal
    print(f"\n[RESULT] Conditional conformal:")
    cond = CondConf(score_fn=score_fn, Phi_fn=phi_fn, quantile_fn=None, infinite_params={})
    cond.setup_problem(x_calib=x_cal, y_calib=y_cal)
    coverage_cond = cond.verify_coverage(x=x_test, y=y_test, quantile=alpha)
    print(f"  Coverage:  {coverage_cond.mean():.4f}")

    return coverage_split, coverage_cond, meta_test

def main():
    parser = argparse.ArgumentParser(
        description="Conditional Conformal Prediction with group fairness"
    )
    parser.add_argument("--alpha", type=float, default=0.1,help="Miscoverage level (1-alpha = target coverage)")
    parser.add_argument("--dataset_name", default="NIH", choices=list(DATASET_CONFIG.keys()))
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--composite_groups", action="store_false", help="Use both age and gender jointly (default: age only)")
    parser.add_argument("--num_splits", type=int, default=5, help="Average marginal coverage over this many random splits")
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset_name]
    group_cols = config["group_cols"] if args.composite_groups else config["group_cols"][:1]

    logits, labels, metadata = load_test_data(args.checkpoint_dir, args.dataset_name)
    metadata = prepare_age_groups(metadata)

    # Run conformal prediction
    coverage_split, coverage_cond, meta_test = run_conformal_prediction(
        logits, labels, metadata, group_cols, args.alpha )

    # Analyze and save results
    analyze_group_coverage(coverage_split, coverage_cond, meta_test, group_cols, args.alpha)
    save_and_plot(coverage_split, coverage_cond, meta_test, args.dataset_name, args.alpha)

    from visualize_coverage import create_all_visualizations
    print("\n[INFO] Creating demographic visualizations...")
    create_all_visualizations(coverage_split, coverage_cond, meta_test,
                              'age_group', 'Patient Gender', 1 - args.alpha, 'Figures')



if __name__ == "__main__":
    main()
