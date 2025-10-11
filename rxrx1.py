import os
import argparse
import pandas as pd
from phi_features import (computeFeatures_probs, find_best_regularization,
                          computeFeatures_indicators, computeFeatures_kernel)
from utils import create_train_calib_test_split, categorical_to_numeric, set_seed
from plot_utils import plot_miscoverage
from save_utils import save_csv, build_cov_df
from feature_io import load_features
from conformal_scores import compute_conformity_scores
from conditional_coverage import compute_both_coverages, compute_prediction_sets
from data_utils import load_dataset
set_seed(42)

DATASET_CONFIG = {
    "rxrx1": {
        "dataset_name": "rxrx1",
        "features_path": "features/rxrx1_features.pt",
        "metadata_path": "data/rxrx1_v1.0/metadata.csv",
        "filter_key": "dataset",
        "filter_value": "test",
        "group_col": "experiment", #cell_type experiment
        "additional_col": ["cell_type"],  # cell_type
        "group_cols": ["experiment", "cell_type"],
        "features_base_path": "features",
    },

    "iwildcam": {
        "dataset_name": "iwildcam",
        "features_path": "features/iwildcam_features.pt",
        "metadata_path": None,
        "filter_key": None,
        "filter_value": None,
        "group_col": "location_grouped",
        "additional_col": ["time_of_day"],
        "group_cols": ["location_grouped", "time_of_day"],
    },

    "fmow": {
        "dataset_name": "fmow",
        "features_path": "features/fmow_features.pt",
        "metadata_path": None,
        "filter_key": None,
        "filter_value": None,
        "group_col": "region_name",  # Primary: like 'experiment'
        "additional_col": ["year"],  # Secondary: like 'cell_type'
        "group_cols": ["region_name", "year"],
    }

}

def load_features_with_metadata(config, features_path=None):
    features_file = features_path or config["features_path"]
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Features file not found: {features_file} ")

    features, logits, labels = load_features(features_file)
    if config["metadata_path"]:
        metadata = pd.read_csv(config["metadata_path"])
    else:
        _, metadata =  load_dataset(config["dataset_name"])

    if config["filter_key"]:
        metadata = metadata[metadata[config["filter_key"]] == config["filter_value"]]

        # Get group labels for stratification
    group_col = config["group_col"]
    group_labels = categorical_to_numeric(metadata, group_col)

    train_idx, calib_idx, test_idx = create_train_calib_test_split(
        len(features),
        train_ratio = 0.25,
        calib_ratio = 0.25,
        random_state = 42,
        y = group_labels,  # Stratify by groups
    )

    return features, logits, labels, metadata, train_idx, calib_idx, test_idx


def load_dataset_features(dataset_name, features_path=None):
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return load_features_with_metadata(config, features_path)


def create_feature_matrix(features, metadata, train_idx, calib_idx, test_idx,
                         dataset_name, method="probs"):

    config = DATASET_CONFIG.get(dataset_name)
    group_col = config["group_col"]

    if group_col not in metadata.columns:
        raise ValueError(f"Group column '{group_col}' not found in metadata")

    # Encode categorical group to numeric and split data
    group_encoded = categorical_to_numeric(metadata, group_col)
    y_group_train = group_encoded[train_idx].astype(int)

    train_feature = features[train_idx]
    calib_feature = features[calib_idx]
    test_feature = features[test_idx]

    print(f"[INFO] Finding best regularization parameter...")
    best_c = find_best_regularization(train_feature, y_group_train)
    print(f"  Best C: {best_c}")

    print(f"[INFO] Computing phi features using method: {method}")
    if method == "indicators":
        phi_cal, phi_test = computeFeatures_indicators(
            train_feature, calib_feature, test_feature, y_group_train, best_c)
    elif method == "kernel":
        phi_cal, phi_test = computeFeatures_kernel(
            train_feature, calib_feature, test_feature, y_group_train, best_c)
    elif method == "probs":
        phi_cal, phi_test = computeFeatures_probs(
            train_feature, calib_feature, test_feature, y_group_train, best_c)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  phi_cal shape: {phi_cal.shape}")
    print(f"  phi_test shape: {phi_test.shape}")

    return phi_cal, phi_test

def run_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha, dataset_name, group_col):

    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha
    )
    compute_prediction_sets(probs_test, q_split, cond_thresholds,
        "results",  f"{dataset_name}_pred_sets" )

    print(f"[INFO] Coverage statistics:")
    print(f"  Split conformal - Mean: {coverages_split.mean():.4f}, Std: {coverages_split.std():.4f}")
    print(f"  Conditional CP  - Mean: {coverages_cond.mean():.4f}, Std: {coverages_cond.std():.4f}")

    return coverages_split, coverages_cond

def save_and_plot(coverages_split, coverages_cond, metadata, test_idx, dataset_name, alpha):
    config = DATASET_CONFIG.get(dataset_name, {})
    group_cols = config.get("group_cols", [config.get("group_col", "experiment")] )

    available_cols = [col for col in group_cols if col in metadata.columns]
    if not available_cols:
        raise ValueError(f"No valid grouping columns found in metadata for dataset {dataset_name}")

    saved_files = []

    for col in available_cols:
        display_name = col.replace('_', ' ').title()
        df_cov = build_cov_df(
            coverages_split, coverages_cond,
            metadata[col].iloc[test_idx],
            group_name=display_name)
        filename = f"{dataset_name}_{col}"
        save_csv(df_cov, filename, "results")
        saved_files.append(f"results/{filename}.csv")
        print(f"[SAVED] {saved_files[-1]}")

    if len(saved_files) >= 2:
        # Plot miscoverage for both grouping columns
        print(f"[INFO] Generating miscoverage plot...")
        plot_miscoverage(main_group=saved_files[0], additional_group=saved_files[1],
                        target_miscoverage=alpha, save_dir="Figures",
                        save_name=f"{dataset_name}_miscoverage")
        print(f"[SAVED] Figures/{dataset_name}_miscoverage.png")
    elif len(saved_files) == 1:
        print(f"[INFO] Single grouping column - saved to {saved_files[0]}")
    else:
        print("[WARNING] Not enough grouping columns to plot miscoverage.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Conditional Conformal Analysis')

    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1)")
    parser.add_argument("--dataset_name", default="iwildcam", choices=["rxrx1","iwildcam", "fmow"],help="Dataset to analyze")
    parser.add_argument("--group_col", default="location_grouped", choices=["experiment", "location_grouped", "region_name"], help="Group column for analysis")
    parser.add_argument('--features_path', type=str, help="Custom path to features file")
    parser.add_argument("--method", default="probs", choices=["indicators", "kernel", "probs"], help="Method for computing features"
                        )
    args = parser.parse_args()

    if args.group_col is None:
        args.group_col = DATASET_CONFIG[args.dataset_name]["group_col"]

    return args

def main():
    args = parse_arguments()
    features, logits, labels, metadata, train_idx, calib_idx, test_idx = load_dataset_features(
                                        args.dataset_name, args.features_path )

    print(f"\nComputing conformity scores...")

    # Load features and splits
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        logits[calib_idx, :], logits[test_idx, :],
        labels[calib_idx], labels[test_idx]
    )
    print(f"  Calib scores: {cal_scores.shape}")
    print(f"  Test scores: {test_scores.shape}")

    print(f"Creating phi feature matrices (method={args.method})...")
    phi_cal, phi_test = create_feature_matrix(features, metadata, train_idx, calib_idx, test_idx,
                                              args.dataset_name, args.method)

    print(f"Running conditional conformal prediction (α={args.alpha})...")
    coverages_split, coverages_cond = run_analysis(phi_cal, phi_test, cal_scores, test_scores, probs_test,
                                                             args.alpha, args.dataset_name, args.group_col)
    print("Saving results and generating plots...")
    save_and_plot(coverages_split, coverages_cond, metadata, test_idx, args.dataset_name, args.alpha)
    print("Analysis complete!")

if __name__ == "__main__":
    main()

