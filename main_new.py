import os
import torch
from conformal_scores import compute_conformity_scores
from utils import set_seed, categorical_to_numeric, find_best_regularization, computeFeatures
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from conditional_coverage import compute_both_coverages, compute_prediction_sets
import argparse
import pandas as pd
from feature_io import load_features

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATASET_CONFIG = {
    'NIH': {
        'features_base_path': 'features',
        'metadata_path': 'data/NIH/Data_Entry_2017_clean.csv',
        'main_group_col': 'Patient Age',
        'additional_col': ['Patient Gender'],
        "group_cols": ["Patient Age", "Patient Gender"],
        'num_classes': 14,
    },
}

def load_split_dataset(dataset_name):
    cfg = DATASET_CONFIG.get(dataset_name)
    if not cfg:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = {}
    base_path = cfg["features_base_path"]

    NIH_PATHOLOGIES = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]

    XRV_PATHOLOGIES = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
        'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
    ]

    nih_indices = [XRV_PATHOLOGIES.index(p) for p in NIH_PATHOLOGIES]

    for split in ["train", "calib", "test"]:
        path = os.path.join(base_path, f"{dataset_name}_{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")
        features, logits, labels, indices = load_features(path)

        if dataset_name == "NIH" and logits is not None:
            logits = logits[:, nih_indices]
            print(f"[INFO] Filtered logits from 18 to {len(nih_indices)} classes for NIH")

        data[split] = {
            "features": features,
            "logits": logits,
            "labels": labels,
            "indices": indices
        }
    meta_path = cfg.get("metadata_path")
    metadata = pd.read_csv(meta_path) if meta_path and os.path.exists(meta_path) else None

    # Use indices to filter metadata for each split
    if metadata is not None and data["train"]["indices"] is not None:
        metadata["split"] = -1
        for split_id, split_name in enumerate(["train", "calib", "test"]):
            split_indices = data[split_name]["indices"].numpy()
            metadata.loc[split_indices, "split"] = split_id
        print(f"[INFO] Metadata aligned using saved indices")

    return data, metadata

def create_feature_matrix(data, metadata, dataset_name, custom_bins):
    config = DATASET_CONFIG.get(dataset_name)
    group_col = config["main_group_col"]

    if metadata is None:
        raise ValueError("Metadata is required for group-based features")

    if "split" not in metadata.columns:
        raise ValueError("Metadata must contain a 'split' column for train/calib/test splits")

    train_meta = metadata[metadata["split"] == 0]

    train_feature = data["train"]["features"]
    calib_feature, test_feature = data["calib"]["features"], data["test"]["features"]

    if pd.api.types.is_numeric_dtype(train_meta[group_col]) and custom_bins is not None:
        binned_all = pd.cut(metadata[group_col], bins=custom_bins, right=False, include_lowest=True)
        metadata[f"{group_col}_binned"] = binned_all.astype(str)
        metadata[group_col] = binned_all.cat.codes.astype(int)
        y_group_train = metadata.loc[train_meta.index, group_col].astype(int)

    elif isinstance(train_meta[group_col].dtype, pd.CategoricalDtype) or train_meta[group_col].dtype == "object":
        metadata[group_col] = categorical_to_numeric(metadata, group_col)
        y_group_train = metadata.loc[train_meta.index, group_col].astype(int)
    else:
        raise ValueError(f"Unsupported data type for group column '{group_col}'")

    print("[DEBUG] Age bin counts in train split:")
    print(y_group_train.value_counts())

    best_c = find_best_regularization(train_feature, y_group_train)

    phi_cal, phi_test = computeFeatures(
        train_feature, calib_feature, test_feature, y_group_train, best_c
    )
    return phi_cal, phi_test, metadata

def run_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha, dataset_name):
    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha )
    compute_prediction_sets(probs_test, q_split, cond_thresholds,"results",  f"{dataset_name}_pred_sets" )
    return coverages_split, coverages_cond

def save_and_plot(coverages_split, coverages_cond, metadata, dataset_name, alpha):
    config = DATASET_CONFIG.get(dataset_name, {})
    group_cols = config.get("group_cols", [])
    available_cols = [col for col in group_cols if col in metadata.columns]
    if not available_cols:
        raise ValueError(f"No valid grouping columns found in metadata for dataset {dataset_name}")

    saved_files = []
    for col in available_cols:
        display_name = col.replace('_', ' ').title()
        subgroup_series = metadata[metadata["split"] == 2][col]
        if f"{col}_binned" in metadata.columns:
            subgroup_series = metadata.loc[metadata["split"] == 2, f"{col}_binned"]
        df_cov = build_cov_df(
            coverages_split, coverages_cond,
            subgroup_series,
            group_name=display_name)
        filename = f"{dataset_name}_{col}"
        save_csv(df_cov, filename, "results")

        saved_files.append(f"results/{filename}.csv")

    if len(saved_files) == 2:
        plot_miscoverage(saved_files[0], saved_files[1],alpha,"Figures", f"{dataset_name}_miscoverage")
    else:
        print("[WARNING] Not enough grouping columns to plot miscoverage.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Conformal prediction analysis')
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1)")
    parser.add_argument("--dataset_name", default="NIH", choices=list(DATASET_CONFIG.keys()),help="Dataset to analyze")
    parser.add_argument("--group_col", default="Patient Age", help="Group column for analysis")
    parser.add_argument('--features_path', type=str, help="Custom path to features file")

    args = parser.parse_args()
    return args

def main():
    custom_bins = [0, 18, 40, 60, 80, 100]
    args = parse_arguments()
    data, metadata = load_split_dataset(args.dataset_name)
    print("Computing conformity scores...")
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        data['calib']['logits'], data['test']['logits'],
        data['calib']['labels'], data['test']['labels'],
    )
    print("Creating feature matrices...")
    phi_cal, phi_test, encoded_metadata = create_feature_matrix(data, metadata, args.dataset_name, custom_bins)

    print("Running conformal analysis...")
    coverages_split, coverages_cond = run_analysis(phi_cal, phi_test, cal_scores, test_scores, probs_test,
                                                             args.alpha, args.dataset_name)
    print("Saving results and generating plots...")
    save_and_plot(coverages_split, coverages_cond, encoded_metadata, args.dataset_name, args.alpha)
    print("Analysis complete!")

if __name__ == "__main__":
    main()










