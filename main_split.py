import torch
from conformal_scores import compute_conformity_scores
from utils import expand_phi
from utils import set_seed
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from data_split_utils import load_split_data, create_phi_split
from conditional_coverage import run_conformal_analysis
import argparse
from config import DATASET_CONFIG

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_results(coverages_split, coverages_cond, metadata, test_labels, dataset_name):
    config = DATASET_CONFIG.get(dataset_name, {})
    grouping_columns = config.get("grouping_columns", ["group"])
    available_columns = [col for col in grouping_columns if col in metadata.columns]
    saved_files = []

    for col in available_columns:
        display_name = col.replace('_', ' ').title()
        if 'split' in metadata.columns:
            test_metadata = metadata[metadata['split'] == 2]
            group_values = test_metadata[col]
        else:
            test_size = len(test_labels)
            group_values = metadata[col].iloc[-test_size:]

        df_cov = build_cov_df(
            coverages_split, coverages_cond,
            group_values,
            group_name=display_name
        )
        filename = f"{dataset_name}_{col}"
        save_csv(df_cov, filename, "results")
        saved_files.append(f"results/{filename}.csv")

    if len(saved_files) >= 2:
        plot_miscoverage(
            main_group=saved_files[0],
            additional_group=saved_files[1],
            target_miscoverage=0.1,
            save_dir="Figures",
            save_name=f"{dataset_name}_miscoverage_comparison"
        )

def run_conditional_analysis(dataset_name, use_groups, add_features, alpha=0.1):
    data = load_split_data(dataset_name, compute_logits=True)
    calib_logits, calib_labels = data['calib']['logits'], data['calib']['labels']
    test_logits, test_labels = data['test']['logits'] , data['test']['labels']
    metadata = data['metadata']

    ####### ??
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(
        calib_logits, test_logits, calib_labels, test_labels
    )

    phi_cal, phi_test = create_phi_split(dataset_name=dataset_name, use_groups=use_groups, add_features=add_features)
    # phi_cal = expand_phi(phi_cal, calib_labels)
    # phi_test = expand_phi(phi_test, test_labels)

    print("phi_cal:", phi_cal.shape, " cal_scores:", len(cal_scores))
    print("phi_test:", phi_test.shape, " test_scores:", len(test_scores))

    assert phi_cal.shape[0] == len(cal_scores), f"Mismatch: phi_cal={phi_cal.shape[0]} vs cal_scores={len(cal_scores)}"
    assert phi_test.shape[0] == len(test_scores), f"Mismatch: phi_test={phi_test.shape[0]} vs test_scores={len(test_scores)}"


    coverages_split, coverages_cond = run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha=alpha, dataset_name=dataset_name)

    if metadata is not None:
        save_results(coverages_split, coverages_cond, metadata, test_labels, dataset_name)
        print(f"Results saved to results/ directory")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Conformal prediction analysis for split datasets')
    parser.add_argument("--dataset", default="ChestX", choices=["ChestX", "PadChest", "VinDr", "MIMIC"],help="Split dataset to analyze")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1)")
    method_group = parser.add_mutually_exclusive_group(required=False)
    method_group.add_argument("--use_groups", action="store_false", help="Use group-based design matrix")
    method_group.add_argument("--use_logits", action="store_true", help="Use classifier logits as design matrix")
    method_group.add_argument("--use_features", action="store_false", help="Use raw features as design matrix")
    parser.add_argument("--add_features", action="store_false", help="Add additional metadata features to design matrix")
    args = parser.parse_args()
    if not any([args.use_groups, args.use_logits, args.use_features]):
        args.use_groups = True
    return args

def main():
    args = parse_arguments()
    run_conditional_analysis(dataset_name=args.dataset, use_groups=args.use_groups,
                             add_features=args.add_features, alpha=args.alpha)

if __name__ == "__main__":
    main()

