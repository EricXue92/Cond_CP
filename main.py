import os
from utils import (computeFeatures, find_best_regularization,
                   create_train_calib_test_split, categorical_to_numeric,
                    one_hot_encode,set_seed)
from plot_utils import plot_miscoverage
from save_utils import save_csv, build_cov_df
from feature_io import load_features
from conformal_scores import compute_conformity_scores
import pandas as pd
from conditional_coverage import compute_both_coverages, compute_prediction_sets
import numpy as np
import argparse
import torch

set_seed(42)

# Dataset configurations
DATASET_CONFIG = {
    "rxrx1": {
        "features_path": "data/rxrx1_v1.0/rxrx1_features.pt",
        "metadata_path": "data/rxrx1_v1.0/metadata.csv",
        "filter_key": "dataset",
        "filter_value": "test",
        "main_group":"experiment",
        "additional_features": ["cell_type"], # metadata columns
        "grouping_columns": ["experiment", "cell_type"]
    },
    "ChestX": {
        "features_path": "features/ChestX_test.pt",
        "metadata_path": "data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv",
        "filter_key": None,
        "filter_value": None,
        "main_group": "age",
        "additional_features": ["age"], # metadata columns
        "grouping_columns": ["sex", "age"]
    }
}

def load_data(dataset_name, features_path=None, split_type="combined"):
    # split_type: 'combined' for single file,
    # 'split' for separate train/calib/test files
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if split_type == "combined":
        features_file = features_path or config["features_path"]
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")

        features, logits, labels = load_features(features_file)
        metadata = pd.read_csv(config["metadata_path"])

        if config["filter_key"]:
            metadata = metadata[metadata[config["filter_key"]] == config["filter_value"]]

        n = len(features)
        assert all(len(x) == n for x in [metadata, labels]), "Size mismatch"
        if logits is not None:
            assert len(logits) == n, "Logits and metadata size mismatch"
        return features, logits, labels, metadata

    elif split_type == "split":
        train_path = os.path.join('features', f"{dataset_name}_train.pt")
        calib_path = os.path.join('features', f"{dataset_name}_calib.pt")
        test_path = os.path.join('features', f"{dataset_name}_test.pt")

        for p in [train_path, calib_path, test_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file not found: {p}")

        train_features, train_logits, train_labels = load_features(train_path)
        calib_features, calib_logits, calib_labels = load_features(calib_path)
        test_features, test_logits, test_labels = load_features(test_path)

        metadata_path = config.get("metadata_path")
        metadata = None
        if metadata_path and os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            if config["filter_key"]:
                metadata = metadata[metadata[config["filter_key"]] == config["filter_value"]]
            total_length = len(train_features) + len(calib_features) + len(test_features)
            assert len(metadata) == total_length, "Metadata size mismatch"

        return {
            "train":{'features': train_features, 'logits': train_logits, 'labels': train_labels},
            "calib":{'features': calib_features, 'logits': calib_logits, 'labels': calib_labels},
            "test":{'features': test_features, 'logits': test_logits, 'labels': test_labels},
            "metadata": metadata
        }

    else:
        raise ValueError(f"Unknown split_type: {split_type}")





def run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha=0.1, dataset_name="rxrx1"):
    # Validate input dimensions
    assert phi_cal.shape[0] == len(cal_scores), "Φ_cal rows must match cal_scores length"
    assert phi_test.shape[0] == len(test_scores), "Φ_test rows must match test_scores length"
    assert phi_cal.shape[1] == phi_test.shape[1], "Φ dimensions must match between calib and test"

    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha=alpha
    )

    compute_prediction_sets(
        probs_test, q_split, cond_thresholds, dataset_name=dataset_name,
        saved_dir="results", base_name="pred_sets"
    )
    return coverages_split, coverages_cond

def save_results(coverages_split, coverages_cond, metadata, test_idx, dataset_name):
    config = DATASET_CONFIG.get(dataset_name, {})
    grouping_columns = config.get("grouping_columns", ["group"])
    available_columns = [col for col in grouping_columns if col in metadata.columns]

    saved_files = []
    for col in available_columns:
        display_name = col.replace('_', ' ').title()
        df_cov = build_cov_df(
            coverages_split, coverages_cond,
            metadata[col].iloc[test_idx],
            group_name=display_name
        )
        filename = f"{dataset_name}_{col}"
        save_csv(df_cov, filename, "results")
        saved_files.append(f"results/{filename}.csv")

    plot_miscoverage(main_group=saved_files[0], additional_group=saved_files[1],
                     target_miscoverage=0.1, save_dir="Figures",
                     save_name=f"{dataset_name}_miscoverage_comparison")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Conformal prediction analysis')
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level")
    parser.add_argument("--dataset", default="rxrx1",choices=list(DATASET_CONFIG.keys()),help="Dataset to analyze")
    method_group = parser.add_mutually_exclusive_group(required=False)
    method_group.add_argument("--use_groups", action="store_true", help="Use group-based design matrix (one-hot experiment)")
    method_group.add_argument("--use_features", action="store_true", help="Use regularized feature-based design matrix")
    parser.add_argument("--add_additional_features", action="store_false",help="Add dataset-specific additional features to design matrix")
    parser.add_argument('--features_path', type=str, help="Custom path to features file")
    args = parser.parse_args()
    if not args.use_groups and not args.use_features:
        args.use_groups = True
    return args

def main():
    args = parse_arguments()
    features, logits, labels, metadata = load_data(args.dataset, args.features_path)
    train_idx, calib_idx, test_idx = create_train_calib_test_split(len(features))
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        logits[calib_idx, :], logits[test_idx, :],
        labels[calib_idx], labels[test_idx]
    )
    phi_cal, phi_test = create_phi(
        features, metadata, train_idx, calib_idx, test_idx,
        dataset_name=args.dataset,
        use_groups=args.use_groups,
        add_additional_features=args.add_additional_features
    )
    print(f"Design matrix shapes: Φ_cal={phi_cal.shape}, Φ_test={phi_test.shape}")
    coverages_split, coverages_cond = run_conformal_analysis(
        phi_cal, phi_test, cal_scores, test_scores, probs_test, args.alpha, args.dataset
    )
    save_results(coverages_split, coverages_cond, metadata, test_idx, args.dataset)

if __name__ == "__main__":
    main()

