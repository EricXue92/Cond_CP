import os
from save_utils import save_csv, build_cov_df
from plot_utils import plot_miscoverage
from conditional_coverage import compute_both_coverages, compute_prediction_sets
from config import DATASET_CONFIG


def run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha=0.1, dataset_name="dataset"):
    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha=alpha
    )

    compute_prediction_sets(
        probs_test, q_split, cond_thresholds,
        dataset_name=dataset_name, saved_dir="results", base_name="pred_sets"
    )
    return coverages_split, coverages_cond


def save_results(coverages_split, coverages_cond, metadata, test_idx, dataset_name):
    config = DATASET_CONFIG.get(dataset_name, {})
    grouping_columns = config.get("grouping_columns", [])
    available = [c for c in grouping_columns if c in metadata.columns]

    saved_files = []
    for col in available:
        df_cov = build_cov_df(
            coverages_split, coverages_cond,
            metadata[col].iloc[test_idx],
            group_name=col.title().replace("_", " ")
        )
        filename = f"{dataset_name}_{col}"
        save_csv(df_cov, filename, "results")
        saved_files.append(f"results/{filename}.csv")

    if len(saved_files) >= 2:
        plot_miscoverage(
            main_group=saved_files[0], additional_group=saved_files[1],
            target_miscoverage=0.1, save_dir="Figures",
            save_name=f"{dataset_name}_miscoverage_comparison"
        )