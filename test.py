import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='cvxpy')

import os
import argparse
import pandas as pd
import numpy as np
import torch

from conditionalconformal.condconf import CondConf
from utils import set_seed
from plot_utils import plot_miscoverage
from save_utils import save_csv, build_cov_df
from conformal_scores import compute_conformity_scores
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

DATASET_CONFIG = {
    "NIH": {
        "group_col": "age_group",
        "additional_col": ["Patient Gender"],
        "group_cols": ["age_group", "Patient Gender"],
    },
}

def prepare_age_groups(metadata, age_column='Patient Age'):
    bins = [0, 18, 40, 60, 80, 100]
    labels = ['0-18', '18-40', '40-60', '60-80', "80-100"]

    metadata[age_column] = pd.to_numeric(metadata[age_column], errors='coerce')
    metadata['age_group'] = pd.cut(metadata[age_column],bins=bins,labels=labels,include_lowest=True).astype(str)
    metadata['age_group'] = metadata['age_group'].fillna('Unknown')

    print(f"[INFO] Age group distribution:")
    for group in labels:
        count = int(np.sum(metadata['age_group'] == group))
        pct = 100 * count / len(metadata)
        print(f"{group:10s}: {count:5d} ({pct:5.1f}%)")

    return metadata

def load_test_data(checkpoint_dir, data_name="NIH"):
    test_data_path = os.path.join(checkpoint_dir, f'test_data_{data_name}.pt')
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    data = torch.load(test_data_path)
    logits, labels, metadata = data['logits'], data['labels'], data['metadata']

    print(f"  Logits shape:  {logits.shape}")
    print(f"  Labels shape:  {labels.shape}")
    print(f"  Metadata rows: {len(metadata)}")

    return logits, labels, metadata

def split_calib_test(logits, metadata, stratify_col='age_group'):
    stratify_labels = None
    if stratify_col in metadata.columns:
        stratify_labels, _ = pd.factorize(metadata[stratify_col])

    indices = np.arange(len(logits))
    calib_idx, test_idx = train_test_split(indices,test_size=0.5,random_state=42,stratify=stratify_labels)
    print(f"  Calibration: {len(calib_idx)} samples")
    print(f"  Evaluation:  {len(test_idx)} samples")
    return calib_idx, test_idx


def create_multilabel_aps_score_fn(all_probs):

    def multilabel_aps_score_fn(x_indices, y_labels):
        if isinstance(x_indices, (int, np.integer)):
            x_indices = np.array([x_indices], dtype=int)
        else:
            x_indices = np.asarray(x_indices, dtype=int)

        y_labels = np.atleast_2d(y_labels)
        if y_labels.ndim == 1:
            y_labels = y_labels.reshape(1, -1)

        if y_labels.shape[0] != len(x_indices):
            if y_labels.shape[0] == 1:
                y_labels = np.repeat(y_labels, len(x_indices), axis=0)
            else:
                raise ValueError(
                    f"Shape mismatch: x_indices={len(x_indices)}, "
                    f"y_labels={y_labels.shape}"
                )

        n_samples = len(x_indices)
        scores = np.zeros(n_samples, dtype=np.float64)

        for i, idx in enumerate(x_indices):
            probs = np.asarray(all_probs[idx], dtype=np.float64).flatten()
            y_true = np.asarray(y_labels[i], dtype=np.float64).flatten()
            n_labels = len(probs)

            pos_idx = np.flatnonzero(y_true > 0.5)

            if pos_idx.size == 0:
                scores[i] = 0.0
                continue
            order = np.argsort(-probs, kind="stable")
            inv_rank = np.empty(n_labels, dtype=np.int32)
            inv_rank[order] = np.arange(n_labels)
            K = int(inv_rank[pos_idx].max()) + 1  # Convert to 1-based count
            sorted_probs = probs[order]
            scores[i] = float(np.sum(sorted_probs[:K]))
        return scores
    return multilabel_aps_score_fn

def create_phi(metadata_all, group_cols):
    all_indicators = []
    feature_names = []

    for col in group_cols:
        if col not in metadata_all.columns:
            print(f"[WARNING] Column '{col}' not in metadata, skipping...")
            continue

        print(f"  Processing '{col}'...")
        labels_encoded, unique_vals = pd.factorize(metadata_all[col])
        k = len(unique_vals)
        print(f" Found {k} categories: {list(unique_vals)}")

        onehot = np.eye(k)[labels_encoded]
        all_indicators.append(onehot)
        feature_names.extend([f"{col}_{val}" for val in unique_vals])

    phi_matrix = np.hstack(all_indicators) if all_indicators else np.array([])

    print(f"\n[INFO] Basis function dimension: {phi_matrix.shape[1]}")
    print(f"[INFO] Features: {feature_names}")

    def phi_fn(x_indices):
        """Basis function - returns demographic indicators."""
        if isinstance(x_indices, (int, np.integer)):
            return phi_matrix[x_indices]
        return phi_matrix[x_indices]

    return phi_fn, feature_names


def run_conditional_conformal(
        labels_cal, labels_test, probs_cal, probs_test,
        metadata_cal, metadata_test, group_cols, alpha=0.1
):
    """
    Run conditional conformal prediction with demographic fairness constraints.

    Returns:
        coverage_split: Standard split conformal coverage (binary array)
        coverage_cond: Conditional conformal coverage (binary array)
        condconf: CondConf object
    """
    print(f"\n{'=' * 70}")
    print(f"CONDITIONAL CONFORMAL PREDICTION")
    print(f"Target coverage: {1 - alpha:.1%}")
    print(f"Demographic groups: {group_cols}")
    print(f"{'=' * 70}")

    # Concatenate calibration and test probabilities
    all_probs = np.vstack([probs_cal, probs_test])
    n_cal = len(probs_cal)
    n_test = len(probs_test)

    # Create score function (closure over all_probs)
    score_fn = create_multilabel_aps_score_fn(all_probs)

    # Create demographic basis function
    metadata_all = pd.concat([
        metadata_cal.reset_index(drop=True),
        metadata_test.reset_index(drop=True)
    ])
    phi_fn, _ = create_phi(metadata_all, group_cols)

    # Initialize CondConf
    print(f"\n[INFO] Setting up conditional conformal...")
    condconf = CondConf(
        score_fn=score_fn,
        Phi_fn=phi_fn,
        quantile_fn=None,
        infinite_params={}
    )

    # Calibrate on calibration set
    x_calib = np.arange(n_cal)
    condconf.setup_problem(x_calib, labels_cal)
    print(f"  ✓ Optimization complete")

    # Verify coverage on test set
    print(f"\n[INFO] Computing coverage on test set...")
    x_test = np.arange(n_cal, n_cal + n_test)
    coverage_cond = condconf.verify_coverage(x_test, labels_test, quantile=alpha)

    # Compute standard split conformal for comparison
    print(f"\n[INFO] Computing standard split conformal baseline...")
    cal_scores = score_fn(x_calib, labels_cal)
    test_scores = score_fn(x_test, labels_test)

    # Standard quantile
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_split = np.quantile(cal_scores, q_level)

    print(f"  Split conformal threshold: {q_split:.4f}")
    print(f"  Calibration scores: [{cal_scores.min():.4f}, {cal_scores.max():.4f}]")
    print(f"  Test scores: [{test_scores.min():.4f}, {test_scores.max():.4f}]")

    # Coverage with standard split conformal
    coverage_split = (test_scores <= q_split).astype(float)

    # Overall coverage rates
    print(f"\n[INFO] Overall coverage:")
    print(f"  Split conformal:  {coverage_split.mean():.4f}")
    print(f"  Conditional:      {coverage_cond.mean():.4f}")
    print(f"  Target:           {1 - alpha:.4f}")

    return coverage_split, coverage_cond, condconf


def analyze_group_coverage(coverage_split, coverage_cond, metadata_test,
                           group_cols, alpha):
    """
    Analyze and print coverage for each demographic group.
    """
    print(f"\n{'=' * 70}")
    print(f"COVERAGE ANALYSIS BY DEMOGRAPHIC GROUPS")
    print(f"Target: {1 - alpha:.1%} coverage")
    print(f"{'=' * 70}")

    group_results = {}

    for col in group_cols:
        if col not in metadata_test.columns:
            continue

        print(f"\n[{col.upper()}]")
        groups = metadata_test[col].values
        unique_groups = np.unique(groups)

        col_results = []

        for group in unique_groups:
            mask = groups == group
            n_samples = int(np.sum(mask))

            cov_split = coverage_split[mask].mean()
            cov_cond = coverage_cond[mask].mean()

            meets_target = cov_cond >= (1 - alpha)
            status = "✓" if meets_target else "✗"

            print(f"  {status} {group:<20} (n={n_samples:4d}): "
                  f"Split={cov_split:.4f}, Gibbs={cov_cond:.4f}, "
                  f"Cond={cov_cond - (1 - alpha):+.4f}")

            col_results.append({
                'group': group,
                'n_samples': n_samples,
                'coverage_split': cov_split,
                'coverage_cond': cov_cond,
                "mis_coverage_split": 1 - cov_split,
                'mis_coverage_cond': 1 - cov_cond ,
                'meets_target': meets_target
            })

        group_results[col] = col_results

        # Summary
        cov_values = [r['coverage_cond'] for r in col_results]
        worst = min(cov_values)
        best = max(cov_values)
        fairness_gap = best - worst

        print(f"\n  Summary:")
        print(f"    Worst-case coverage: {worst:.4f}")
        print(f"    Best coverage:       {best:.4f}")
        print(f"    Fairness gap:        {fairness_gap:.4f}")
        print(f"    Groups meeting target: {sum(r['meets_target'] for r in col_results)}/{len(col_results)}")

    return group_results


def save_and_plot(coverage_split, coverage_cond, metadata, test_idx,
                  dataset_name, group_cols, alpha):
    """Save results and generate plots."""
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("Figures", exist_ok=True)

    saved_files = []

    for col in group_cols:
        if col not in metadata.columns:
            continue

        display_name = col.replace('_', ' ').title()
        df_cov = build_cov_df(
            coverage_split,
            coverage_cond,  # Use Gibbs coverage as "conditional"
            metadata[col].iloc[test_idx],
            group_name=display_name
        )

        filename = f"{dataset_name}_{col}_gibbs"
        save_csv(df_cov, filename, "results")
        filepath = f"results/{filename}.csv"
        saved_files.append(filepath)
        print(f"  ✓ Saved: {filepath}")

    # Generate plot
    if len(saved_files) >= 2:
        print(f"\n[INFO] Generating fairness plot...")
        plot_miscoverage(
            main_group=saved_files[0],
            additional_groups=saved_files[1:],
            target_miscoverage=alpha,
            save_dir="Figures",
            save_name=f"{dataset_name}_cond"
        )
        print(f"  ✓ Saved: Figures/{dataset_name}_gibbs_fairness.pdf")


def main():
    parser = argparse.ArgumentParser(description='Conditional Conformal Prediction')
    parser.add_argument("--alpha", type=float, default=0.1,help="Miscoverage level (default: 0.1)")
    parser.add_argument("--dataset_name", default="NIH", choices=["ChestX", "NIH"], help="Dataset to analyze")
    parser.add_argument('--features_path', type=str,help="Custom path to features file")
    parser.add_argument('--checkpoint_dir', type=str,default="checkpoints", help="Custom path to dataset file")
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset_name]

    # Get configuration
    # Step 1: Load data
    logits, labels, metadata  = load_test_data(args.checkpoint_dir, args.dataset_name)

    metadata = prepare_age_groups(metadata)
    print(metadata.columns)

    calib_idx, test_idx = split_calib_test(logits, metadata, 'age_group')


    # Step 2: Compute conformity scores
    print("COMPUTING CONFORMITY SCORES")

    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        logits[calib_idx, :],
        logits[test_idx, :],
        labels[calib_idx],
        labels[test_idx]
    )

    print(f"[INFO] Calibration scores: min={cal_scores.min():.4f}, "
          f"max={cal_scores.max():.4f}, median={np.median(cal_scores):.4f}")
    print(f"[INFO] Test scores: min={test_scores.min():.4f}, "
          f"max={test_scores.max():.4f}, median={np.median(test_scores):.4f}")

    # Step 3: Get metadata for splits
    metadata_cal = metadata.iloc[calib_idx].reset_index(drop=True)
    metadata_test = metadata.iloc[test_idx].reset_index(drop=True)

    group_cols = config["group_cols"]

    # Step 4: Run conditional conformal
    coverage_split, coverage_cond, _ = run_conditional_conformal(
        labels[calib_idx],
        labels[test_idx],
        probs_cal,
        probs_test,
        metadata_cal,
        metadata_test,
        group_cols,
        args.alpha
    )

    # Step 5: Analyze coverage by groups
    analyze_group_coverage(
        coverage_split,
        coverage_cond,
        metadata_test,
        group_cols,
        args.alpha
    )

    # Step 6: Save results and plots
    save_and_plot(
        coverage_split,
        coverage_cond,
        metadata,
        test_idx,
        args.dataset_name,
        group_cols,
        args.alpha
    )


if __name__ == "__main__":
    main()