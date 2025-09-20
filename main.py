import os
from utils import (computeFeatures, find_best_regularization,
                   create_train_calib_test_split, encode_labels, build_cov_df,
                    one_hot_encode,set_seed)
from plot_utils import plot_miscoverage
from save_utils import save_csv
from feature_io import load_features
from conformal_scores import compute_conformity_scores
import pandas as pd
from conditional_coverage import compute_both_coverages, compute_prediction_sets
import numpy as np
import argparse

# Fix seed ONCE at the very beginning
set_seed(42)

# Dataset configurations
DATASET_CONFIG = {
    "rxrx1": {
        "features_path": "data/rxrx1_v1.0/rxrx1_features.pt",
        "metadata_path": "data/rxrx1_v1.0/metadata.csv",
        "filter_key": "dataset",
        "filter_value": "test"
    },
    "ChestX": {
        "features_path": "features/ChestX_test.pt",
        "metadata_path": "data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv",
        "filter_key": None,
        "filter_value": None
    }
}

def load_data(dataset_name, features_path=None):
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    features_file = features_path or config["features_path"]
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file}")
    features, logits, labels = load_features(features_file)
    metadata = pd.read_csv(config["metadata_path"])
    if config["filter_key"]:
        metadata = metadata[metadata[config["filter_key"]] == config["filter_value"]]
    data_length = len(features)
    assert len(metadata) == data_length, "Features and metadata size mismatch"
    assert len(logits) == data_length, "Logits and metadata size mismatch"
    assert len(labels) == data_length, "Labels and metadata size mismatch"

    return features, logits, labels, metadata


def create_phi(features, metadata, train_idx, calib_idx, test_idx,
                         use_groups=True, add_celltype=False):

    experiment = encode_labels(metadata, "experiment")
    exp_train_y = experiment[train_idx].astype(int)
    exp_cal_y = experiment[calib_idx].astype(int)
    exp_test_y = experiment[test_idx].astype(int)

    if use_groups:
        phi_cal = one_hot_encode(exp_cal_y)
        phi_test = one_hot_encode(exp_test_y)

        if add_celltype and "cell_type" in metadata.columns:
            ct_codes = pd.factorize(metadata["cell_type"])[0]
            phi_cal = np.hstack([phi_cal, one_hot_encode(ct_codes[calib_idx])])
            phi_test = np.hstack([phi_test, one_hot_encode(ct_codes[test_idx])])
    else:
        # Use regularized features
        train_feature = features[train_idx, :]
        calib_feature = features[calib_idx, :]
        test_feature = features[test_idx, :]

        best_c = find_best_regularization(train_feature, exp_train_y)
        phi_cal, phi_test = computeFeatures(
            train_feature, calib_feature, test_feature, exp_train_y, best_c
        )

    return phi_cal, phi_test


def run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha=0.1):
    # Validate input dimensions
    assert phi_cal.shape[0] == len(cal_scores), "Φ_cal rows must match cal_scores length"
    assert phi_test.shape[0] == len(test_scores), "Φ_test rows must match test_scores length"
    assert phi_cal.shape[1] == phi_test.shape[1], "Φ dimensions must match between calib and test"

    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha=alpha
    )

    compute_prediction_sets(
        probs_test, q_split, cond_thresholds,
        saved_dir="results", base_name="pred_sets"
    )

    return coverages_split, coverages_cond


def save_results(coverages_split, coverages_cond, metadata, test_idx):
    df_cov_cells = build_cov_df(
        coverages_split, coverages_cond,
        metadata['cell_type'].iloc[test_idx],
        group_name='Cell Type'
    )

    df_cov_experiments = build_cov_df(
        coverages_split, coverages_cond,
        metadata['experiment'].iloc[test_idx],
        group_name='Experiment'
    )

    # Save results
    save_csv(df_cov_cells, "cells", "results")
    save_csv(df_cov_experiments, "experiments", "results")
    plot_miscoverage(save_name="Experiment_Cell_Miscoverage.pdf")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Conformal prediction analysis')

    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (default: 0.1)")
    parser.add_argument("--dataset", default="rxrx1",
                        choices=["rxrx1", "ChestX", "PadChest", "VinDr", "MIMIC"],
                        help="Dataset to analyze")

    method_group = parser.add_mutually_exclusive_group(required=False)
    method_group.add_argument("--use_groups", action="store_true",
                              help="Use group-based design matrix (one-hot experiment)")

    method_group.add_argument("--use_features", action="store_true",
                              help="Use regularized feature-based design matrix")

    # Additional options
    parser.add_argument("--add_celltype", action="store_false",
                        help="Add cell type one-hots to group-based features")
    parser.add_argument('--features_path', type=str,
                        help="Custom path to features file")

    # Legacy paths (for backward compatibility)
    parser.add_argument('--train_path', type=str, default="features/ChestX_train.pt")
    parser.add_argument('--calib_path', type=str, default="features/ChestX_calib.pt")
    parser.add_argument('--test_path', type=str, default="features/ChestX_test.pt")

    args = parser.parse_args()

    if not args.use_groups and not args.use_features:
        args.use_groups = True

    print(f"Running conformal prediction with args: {args}")
    return args

def main():
    args = parse_arguments()
    print(f"Loading {args.dataset} data...")
    features, logits, labels, metadata = load_data(args.dataset, args.features_path)
    print("Creating train/calibration/test splits...")
    train_idx, calib_idx, test_idx = create_train_calib_test_split(len(features))

    print("Computing conformity scores...")
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        logits[calib_idx, :], logits[test_idx, :],
        labels[calib_idx], labels[test_idx]
    )
    print("Creating design matrices...")
    phi_cal, phi_test = create_phi(
        features, metadata, train_idx, calib_idx, test_idx,
        use_groups=args.use_groups,
        add_celltype=args.add_celltype
    )
    print(f"Design matrix shapes: Φ_cal={phi_cal.shape}, Φ_test={phi_test.shape}")

    print("Running conformal analysis...")
    coverages_split, coverages_cond = run_conformal_analysis(
        phi_cal, phi_test, cal_scores, test_scores, probs_test, args.alpha
    )
    print("Saving results...")
    save_results(coverages_split, coverages_cond, metadata, test_idx)
    print("Conformal prediction analysis complete!")

if __name__ == "__main__":
    main()

