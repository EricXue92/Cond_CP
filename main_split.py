import torch
from conformal_scores import compute_conformity_scores
from utils import set_seed
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from data_split_utils import load_split_data, create_phi_split
from conditional_coverage import run_conformal_analysis
import argparse
from config import DATASET_CONFIG
import pandas as pd

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_results(coverages_split, coverages_cond, metadata, test_labels, dataset_name,
                 analysis_type, n_bins=5):
    config = DATASET_CONFIG.get(dataset_name, {})
    grouping_columns = config.get("grouping_columns", [])

    available_cols = [col for col in grouping_columns if col in metadata.columns]
    saved_files = []

    if "split" in metadata.columns:
        test_mask = (metadata["split"] == 2)
    else:
        n_test = len(test_labels)
        test_mask = metadata.index >= (len(metadata) - n_test)

    for col in available_cols:
        display_name = col.replace("_", " ").title()
        values = metadata.loc[test_mask, col]

        if pd.api.types.is_numeric_dtype(values):
            binned, bins = pd.qcut(values, q=n_bins, duplicates="drop", retbins=True)
            group_values = binned.astype(str).to_numpy()  # labels like "(20.0, 40.0]"
            print(f"[INFO] Binned column '{col}' into {len(bins) - 1} bins: {bins}")
        else:
            group_values = values.to_numpy()

        df_cov = build_cov_df(
            coverages_split, coverages_cond, group_values, group_name=display_name
        )

        filename = f"{dataset_name}_{col}_{analysis_type}"
        save_csv(df_cov, filename, "results")
        saved_files.append(f"results/{filename}.csv")

    # for col in available_columns:
    #     display_name = col.replace('_', ' ').title()
    #     if "split" in metadata.columns:
    #         test_metadata = metadata[metadata["split"] == 2]
    #         group_values = test_metadata[col]
    #     else:
    #         test_size = len(test_labels)
    #         group_values = metadata[col].iloc[-test_size:]
    #
    #     df_cov = build_cov_df(coverages_split, coverages_cond, group_values, group_name=display_name)
    #
    #     filename = f"{dataset_name}_{col}_{analysis_type}"
    #     save_csv(df_cov, filename, "results")
    #     saved_files.append(f"results/{filename}.csv")

    plot_miscoverage(main_group=saved_files[0], additional_group=saved_files[1], target_miscoverage=0.1,
                            save_dir="Figures", save_name=f"{dataset_name}_miscoverage_{analysis_type}")

def run_conditional_analysis(dataset_name, use_groups, use_logits, add_features, alpha=0.1):

    analysis_type = "groups" if use_groups else "logits"
    data = load_split_data(dataset_name, compute_missing_logits=True)

    calib_logits, calib_labels = data['calib']['logits'], data['calib']['labels']
    test_logits, test_labels = data['test']['logits'] , data['test']['labels']

    metadata = data['metadata']

    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        calib_logits, test_logits, calib_labels, test_labels
    )


    phi_cal, phi_test = create_phi_split(
        dataset_name=dataset_name,
        data=data,
        use_groups=use_groups,
        use_logits=use_logits,
        add_additional_features=add_features
    )

    coverages_split, coverages_cond = run_conformal_analysis(
        phi_cal, phi_test, cal_scores, test_scores, probs_test,
        alpha=alpha, dataset_name=dataset_name, analysis_type=analysis_type)

    save_results(coverages_split, coverages_cond, metadata, test_labels, dataset_name, analysis_type)
    return coverages_split, coverages_cond

def parse_arguments():
    parser = argparse.ArgumentParser(description='Conformal prediction analysis for split datasets')
    parser.add_argument("--dataset", default="ChestX", choices=["ChestX", "PadChest", "VinDr", "MIMIC"],help="Split dataset to analyze")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1)")
    parser.add_argument("--use_groups", action="store_false", help="Use group-based design matrix")
    parser.add_argument("--use_logits", action="store_true", help="Use classifier logits as design matrix")

    parser.add_argument("--add_features", action="store_false", help="Add additional metadata features to design matrix")
    args = parser.parse_args()

    if sum([args.use_groups, args.use_logits]) != 1:
        parser.error("Please specify exactly one of --use_groups or --use_logits.")
    return args

def main():
    args = parse_arguments()
    run_conditional_analysis(dataset_name=args.dataset, use_groups=args.use_groups,
                            use_logits=args.use_logits, add_features=args.add_features, alpha=args.alpha)

if __name__ == "__main__":
    main()

