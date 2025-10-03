import os
import argparse
import torch
import pandas as pd
from conformal_scores import compute_conformity_scores
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from conditional_coverage import compute_both_coverages, compute_prediction_sets
from utils import set_seed, categorical_to_numeric
from phi_features import computeFeatures, find_best_regularization
from feature_io import load_features

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XRV_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
]

# Dataset configs
DATASET_CONFIG = {
    "NIH": {
        "features_path": "features",
        "metadata_path": "data/NIH/Data_Entry_2017_clean.csv",
        "main_group_col": "Patient Age",
        "group_cols": ["Patient Age", "Patient Gender"],
        "subset_indices": [XRV_PATHOLOGIES.index(p) for p in XRV_PATHOLOGIES[:14]],  # keep 14 only
    },
    "CheXpert": {
        "features_path": "features",
        "metadata_path": "data/CheXpert/chexpert_labels.csv",
        "main_group_col": "Sex",
        "group_cols": ["Sex", "Age"],
        "subset_indices": None,  # keep all 18
    },
    "PadChest": {
        "features_path": "features",
        "metadata_path": "data/PadChest/PadChest.csv",
        "main_group_col": "Patient Age",
        "group_cols": ["Patient Age", "Sex"],
        "subset_indices": None,
    }
}

# DATASET_CONFIG = {
#     'NIH': {
#         'features_path': 'features',
#         'metadata_path': 'data/NIH/Data_Entry_2017_clean.csv',
#         'main_group_col': 'Patient Age',
#         'additional_col': ['Patient Gender'],
#         "group_cols": ["Patient Age", "Patient Gender"],
#         'num_classes': 14,
#     },
# }

def load_split_dataset(dataset_name):
    cfg = DATASET_CONFIG.get(dataset_name)
    data = {}
    base_path = cfg["features_path"]

    # NIH_PATHOLOGIES = [
    #     'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    #     'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    #     'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    # ]
    #
    # XRV_PATHOLOGIES = [
    #     'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    #     'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    #     'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
    #     'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum'
    # ]
    #
    # nih_indices = [XRV_PATHOLOGIES.index(p) for p in NIH_PATHOLOGIES]

    for split in ["train", "calib", "test"]:
        path = os.path.join(base_path, f"{dataset_name}_{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")
        features, logits, labels, indices = load_features(path)

        if cfg.get("subset_indices") and logits is not None:
            logits = logits[:, cfg["subset_indices"]]
            print(f"[INFO] {dataset_name}: logits filtered to {logits.shape[1]} classes")

        data[split] = dict(features=features, logits=logits, labels=labels, indices=indices)

        # if dataset_name == "NIH" and logits is not None:
        #     logits = logits[:, nih_indices]
        #     print(f"[INFO] Filtered logits from 18 to {len(nih_indices)} classes for NIH")

        # data[split] = {
        #     "features": features,
        #     "logits": logits,
        #     "labels": labels,
        #     "indices": indices
        # }

    metadata = pd.read_csv(cfg["metadata_path"]) if os.path.exists(cfg["metadata_path"]) else None

    if metadata is not None and data["train"]["indices"] is not None:
        metadata["split"] = -1
        for i, split in enumerate(["train", "calib", "test"]):
            metadata.loc[data[split]["indices"].numpy(), "split"] = i

            # split_indices = data[split_name]["indices"].numpy()
            # metadata.loc[split_indices, "split"] = split_id
        print(f"[INFO] Metadata aligned using saved indices")

    return data, metadata

def create_feature_matrix(data, metadata, dataset_name, bins):
    config = DATASET_CONFIG.get(dataset_name)
    group_col = config["main_group_col"]

    if metadata is None or "split" not in metadata.columns:
        raise ValueError("Metadata with 'split' column is required")

    train_meta = metadata[metadata["split"] == 0]
    X_train, X_cal, X_test = data["train"]["features"], data["calib"]["features"], data["test"]["features"]


    if pd.api.types.is_numeric_dtype(train_meta[group_col]) and bins:
        binned = pd.cut(metadata[group_col], bins=bins, right=False, include_lowest=True)
        metadata[f"{group_col}_binned"] = binned.astype(str)
        metadata[group_col] = binned.cat.codes.astype(int)
    elif metadata[group_col].dtype == "object":
        metadata[group_col] = categorical_to_numeric(metadata, group_col)

    y_group_train = metadata.loc[train_meta.index, group_col].astype(int)
    print("[DEBUG] Group counts (train):\n", y_group_train.value_counts())

    best_c = find_best_regularization(X_train, y_group_train)
    phi_cal, phi_test = computeFeatures(X_train, X_cal, X_test, y_group_train, best_c)
    return phi_cal, phi_test, metadata


def run_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha, dataset_name):
    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha )
    compute_prediction_sets(probs_test, q_split, cond_thresholds,"results",  f"{dataset_name}_pred_sets" )
    return coverages_split, coverages_cond

def save_and_plot(cover_split, cover_cond, metadata, dataset_name, alpha):
    cfg = DATASET_CONFIG[dataset_name]
    saved = []

    for col in cfg["group_cols"]:
        if col not in metadata:
            continue
        subgroup = metadata.loc[metadata["split"] == 2, f"{col}_binned"] \
            if f"{col}_binned" in metadata else metadata.loc[metadata["split"] == 2, col]
        df_cov = build_cov_df(cover_split, cover_cond, subgroup, group_name=col)
        save_csv(df_cov, f"{dataset_name}_{col}", "results")
        saved.append(f"results/{dataset_name}_{col}.csv")

    if len(saved) >= 2:
        plot_miscoverage(saved[0], saved[1], alpha, "Figures", f"{dataset_name}_miscoverage")


def main():
    parser = argparse.ArgumentParser(description='Conformal prediction analysis')
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1)")
    parser.add_argument("--dataset_name", default="NIH", choices=list(DATASET_CONFIG.keys()),help="Dataset to analyze")
    parser.add_argument("--group_col", default="Patient Age", help="Group column for analysis")
    args = parser.parse_args()

    bins = [0, 18, 10, 30, 20, 40, 60, 0, 0, 100]
    data, metadata = load_split_dataset(args.dataset_name)

    print("Computing conformity scores...")
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        data['calib']['logits'], data['test']['logits'],
        data['calib']['labels'], data['test']['labels'],
    )
    print("Creating feature matrices...")
    phi_cal, phi_test, encoded_metadata = create_feature_matrix(data, metadata, args.dataset_name, bins)
    print(phi_cal.shape, phi_test.shape)
    print("Example row:", phi_cal[0])

    print("Running conformal analysis...")
    coverages_split, coverages_cond = run_analysis(phi_cal, phi_test, cal_scores, test_scores, probs_test,
                                                             args.alpha, args.dataset_name)
    print("Saving results and generating plots...")
    save_and_plot(coverages_split, coverages_cond, encoded_metadata, args.dataset_name, args.alpha)
    print("Analysis complete!")

if __name__ == "__main__":
    main()










