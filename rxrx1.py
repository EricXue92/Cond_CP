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
from config import DATASET_CONFIG

set_seed(42)

# Dataset configurations
def load_data(dataset_name, features_path=None):
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    features_file = features_path if features_path is not None else config["features_path"]

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: `{features_file}`")

    # Load features
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
                         dataset_name, use_groups, add_additional_features):
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    main_group = config["main_group"]
    additional_features = config.get("additional_features", []) if add_additional_features else []

    if main_group not in metadata.columns:
        cols = ", ".join(metadata.columns)
        raise ValueError(f"Main group '{main_group}' not in metadata. Available: {cols}")
    # Encode main group "experiment" ( categorical to numeric --> 0,1,2,3,..13 )
    group_encoded = categorical_to_numeric(metadata, main_group)
    y_train_group_encoded = group_encoded[train_idx].astype(int)
    y_cal_group_encoded = group_encoded[calib_idx].astype(int)
    y_test_group_encoded = group_encoded[test_idx].astype(int)

    if use_groups:
        phi_cal = one_hot_encode(y_cal_group_encoded)
        phi_test = one_hot_encode(y_test_group_encoded)
        if add_additional_features:
            for feature in additional_features:
                if feature not in metadata.columns:
                    continue
                try:
                    if metadata[feature].dtype in ['object', 'category']:
                        codes = pd.factorize(metadata[feature])[0]
                        feature_cal = one_hot_encode(codes[calib_idx])
                        feature_test = one_hot_encode(codes[test_idx])
                    else:
                        values = metadata[feature].values
                        bins = pd.qcut(values, q=10, duplicates='drop', labels=False)
                        feature_cal = one_hot_encode(bins[calib_idx])
                        feature_test = one_hot_encode(bins[test_idx])
                    phi_cal = np.hstack([phi_cal, feature_cal])
                    phi_test = np.hstack([phi_test, feature_test])
                except Exception as e:
                    print(f"Error processing feature '{feature}': {e}. Skipping.")
                    continue
    else:
        train_feature = features[train_idx, :]
        calib_feature = features[calib_idx, :]
        test_feature = features[test_idx, :]
        best_c = find_best_regularization(train_feature, y_train_group_encoded)
        phi_cal, phi_test = computeFeatures(
            train_feature, calib_feature, test_feature, y_train_group_encoded, best_c
        )
    return phi_cal, phi_test

def run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha=0.1, dataset_name="rxrx1", analysis_type='groups'):

    assert phi_cal.shape[0] == len(cal_scores), "Φ_cal rows must match cal_scores length"
    assert phi_test.shape[0] == len(test_scores), "Φ_test rows must match test_scores length"
    assert phi_cal.shape[1] == phi_test.shape[1], "Φ dimensions must match between calib and test"

    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha=alpha
    )

    compute_prediction_sets(probs_test, q_split, cond_thresholds, dataset_name,
        saved_dir="results",  base_name=f"{dataset_name}_pred_sets_{analysis_type}",
                            analysis_type=analysis_type
    )
    return coverages_split, coverages_cond

def save_results(coverages_split, coverages_cond, metadata, test_idx, dataset_name, analysis_type='groups'):
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
        filename = f"{dataset_name}_{col}_{analysis_type}"
        save_csv(df_cov, filename, "results")
        saved_files.append(f"results/{filename}.csv")

    plot_miscoverage(main_group=saved_files[0], additional_group=saved_files[1], target_miscoverage=0.1,
                            save_dir="Figures", save_name=f"{dataset_name}_miscoverage_{analysis_type}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Conformal prediction analysis')
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1)")
    parser.add_argument("--dataset", default="rxrx1", choices=list(DATASET_CONFIG.keys()),help="Dataset to analyze")
    parser.add_argument("--use_groups", action="store_true",help="Use group-based design matrix (one-hot experiment)")
    parser.add_argument("--use_features", action="store_true", help="Use regularized feature-based design matrix")

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
    analysis_type = "features" if args.use_features else "groups"

    coverages_split, coverages_cond = run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores, probs_test, args.alpha,
                                                             args.dataset, analysis_type)
    save_results(coverages_split, coverages_cond, metadata, test_idx, dataset_name=args.dataset, analysis_type=analysis_type)

if __name__ == "__main__":
    main()