import os
import argparse
import torch
import pandas as pd
from conformal_scores import compute_conformity_scores
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from conditional_coverage import compute_both_coverages, compute_prediction_sets
from utils import set_seed, categorical_to_numeric
from phi_features import computeFeatures_probs, computeFeatures_indicators, computeFeatures_kernel
from feature_io import load_features

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XRV_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
]

DATASET_CONFIG = {
    "NIH": {
        "features_path": "features",
        "metadata_path": "data/NIH/Data_Entry_2017_clean.csv",
        "main_group_col": "Patient Age",
        "group_cols": ["Patient Age", "Patient Gender"],
        "subset_indices": [XRV_PATHOLOGIES.index(p) for p in XRV_PATHOLOGIES[:14]],
    },
    "CheXpert": {
        "features_path": "features",
        "metadata_path": "data/CheXpert/chexpert_labels.csv",
        "main_group_col": "Sex",
        "group_cols": ["Sex", "Age"],
        "subset_indices": None,
    },
    "PadChest": {
        "features_path": "features",
        "metadata_path": "data/PadChest/PadChest.csv",
        "main_group_col": "Patient Age",
        "group_cols": ["Patient Age", "Sex"],
        "subset_indices": None,
    }
}

def load_split_dataset(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    data = {}

    for split in ["train", "calib", "test"]:
        path = os.path.join(cfg["features_path"], f"{dataset_name}_{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")

        features, logits, labels, indices = load_features(path)

        # Filter to subset of pathologies if needed
        if cfg["subset_indices"] is not None and logits is not None:
            logits = logits[:, cfg["subset_indices"]]
            print(f"[INFO] Filtered logits to {logits.shape[1]} classes")

        data[split] = {
            "features": features,
            "logits": logits,
            "labels": labels,
            "indices": indices
        }

    # Load and align metadata with splits
    metadata = pd.read_csv(cfg["metadata_path"]) if os.path.exists(cfg["metadata_path"]) else None

    if metadata is not None and data["train"]["indices"] is not None:
        metadata["split"] = -1
        for split_id, split_name in enumerate(["train", "calib", "test"]):
            split_indices = data[split_name]["indices"].numpy()
            metadata.loc[split_indices, "split"] = split_id
        print(f"[INFO] Metadata aligned using saved indices")

    return data, metadata


def bin_demographic_groups(metadata, group_col, bins):
    """Bin continuous demographic variables or encode categorical ones."""
    train_meta = metadata[metadata["split"] == 0]

    if pd.api.types.is_numeric_dtype(train_meta[group_col]) and bins:
        binned = pd.cut(metadata[group_col], bins=bins, right=False, include_lowest=True)
        metadata[f"{group_col}_binned"] = binned.astype(str)
        metadata[group_col] = binned.cat.codes.astype(int)
    elif metadata[group_col].dtype == "object":
        metadata[group_col] = categorical_to_numeric(metadata, group_col)

    return metadata


def create_feature_matrix(data, metadata, dataset_name, bins, method="indicators"):
    cfg = DATASET_CONFIG[dataset_name]
    group_col = cfg["main_group_col"]

    if metadata is None or "split" not in metadata.columns:
        raise ValueError("Metadata with 'split' column is required")

    # Extract features
    X_train = data["train"]["features"]
    X_cal = data["calib"]["features"]
    X_test = data["test"]["features"]

    # Save path for model checkpoints
    save_path = f"checkpoints/{dataset_name}_lr.pkl"

    # For other methods, bin demographics first
    metadata = bin_demographic_groups(metadata, group_col, bins)
    train_meta = metadata[metadata["split"] == 0]
    y_group_train = metadata.loc[train_meta.index, group_col].astype(int)

    print(f"[DEBUG] Group counts (train):\n{y_group_train.value_counts()}")

    # Select phi feature generation method
    phi_generators = {
        "indicators": lambda: computeFeatures_indicators(
            X_train, X_cal, X_test, y_group_train,
            save_path=save_path,
            dataset_name=dataset_name,
            include_probabilities=False
        ),
        "kernel": lambda: computeFeatures_kernel(
            X_train, X_cal, X_test, y_group_train,
            save_path=save_path,
            kernel_gamma=4.0
        ),
        "mixed": lambda: computeFeatures_indicators(
            X_train, X_cal, X_test, y_group_train,
            save_path=save_path,
            include_probabilities=True
        ),
        "original": lambda: computeFeatures_probs(
            X_train, X_cal, X_test, y_group_train,
            save_path=save_path,
            dataset_name=dataset_name
        )
    }

    if method not in phi_generators:
        raise ValueError(f"Unknown method: {method}. Choose from {list(phi_generators.keys())}")
    print(f"[INFO] Using '{method}' feature generation method")
    phi_cal, phi_test = phi_generators[method]()
    return phi_cal, phi_test, metadata


def run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha, dataset_name):
    """Compute split and conditional conformal coverages."""
    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha
    )
    compute_prediction_sets(
        probs_test, q_split, cond_thresholds,
        "results", f"{dataset_name}_pred_sets"
    )
    return coverages_split, coverages_cond


def save_results(coverages_split, coverages_cond, metadata, dataset_name, alpha):
    """Save coverage results and generate miscoverage plots."""
    cfg = DATASET_CONFIG[dataset_name]
    saved_paths = []
    test_metadata = metadata[metadata["split"] == 2]

    for col in cfg["group_cols"]:
        if col not in metadata:
            continue

        # Use binned column if available
        subgroup_col = f"{col}_binned" if f"{col}_binned" in metadata else col
        subgroups = test_metadata[subgroup_col]

        df_cov = build_cov_df(coverages_split, coverages_cond, subgroups, group_name=col)
        save_csv(df_cov, f"{dataset_name}_{col}", "results")
        saved_paths.append(f"results/{dataset_name}_{col}.csv")

    # Generate miscoverage plot if multiple group columns available
    if len(saved_paths) >= 2:
        plot_miscoverage(
            saved_paths[0], saved_paths[1], alpha,
            "Figures", f"{dataset_name}_miscoverage"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Conformal prediction with group-conditional coverage'
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Miscoverage level (default: 0.1)"
    )
    parser.add_argument(
        "--dataset_name", default="NIH",
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--method", default="augmented",
        choices=["indicators", "kernel", "original", "mixed", "augmented"],
        help="Feature generation method"
    )
    args = parser.parse_args()

    # Configuration
    age_bins = [0, 18, 40, 60, 80, 100]

    # Pipeline
    print(f"\n{'=' * 70}")
    print(f"Conformal Prediction Analysis: {args.dataset_name}")
    print(f"Method: {args.method} | Alpha: {args.alpha}")
    print(f"{'=' * 70}\n")

    print("[1/5] Loading dataset...")
    data, metadata = load_split_dataset(args.dataset_name)

    print("[2/5] Computing conformity scores...")
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        data['calib']['logits'], data['test']['logits'],
        data['calib']['labels'], data['test']['labels']
    )

    print("[3/5] Creating feature matrices...")
    phi_cal, phi_test, metadata = create_feature_matrix(
        data, metadata, args.dataset_name, age_bins, args.method
    )
    print(f"Phi shapes: Cal {phi_cal.shape}, Test {phi_test.shape}")

    print("[4/5] Running conformal analysis...")
    coverages_split, coverages_cond = run_conformal_analysis(
        phi_cal, phi_test, cal_scores, test_scores,
        probs_test, args.alpha, args.dataset_name
    )

    print("[5/5] Saving results and generating plots...")
    save_results(coverages_split, coverages_cond, metadata, args.dataset_name, args.alpha)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

