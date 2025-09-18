from tornado.http1connection import parse_hex_int
import torch

from utils import (computeFeatures, find_best_regularization,
                   create_train_calib_test_split, encode_labels, build_cov_df,
                   plot_miscoverage, save_csv, one_hot_encode,set_seed
                   )
from extract_features import load_features
from conformal_scores import compute_conformity_scores
import os
import pandas as pd
from conditional_coverage import compute_both_coverages, compute_prediction_sets
import numpy as np
import argparse

set_seed(42)

def main(args):
    # Load features
    filepath = 'data/rxrx1_v1.0/rxrx1_features.pt'  # labels: sirna
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Features file not found: {filepath}")
    features, logits, y= load_features(filepath)

    # Load metadata
    metadata = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
    metadata = metadata[metadata['dataset'] == 'test']

    assert len(metadata) == len(features), "Features and metadata size mismatch"
    assert len(metadata) == len(logits), "metadata and logits size mismatch"
    assert len(metadata) == len(y), "metadata and y size mismatch"

    train_idx, calib_idx, test_idx = create_train_calib_test_split(len(features))

    # cal_scores: (n_cal,), test_scores: (n_test,)
    cal_scores, test_scores, probs_cal, probs_test = compute_conformity_scores(logits[calib_idx, :], logits[test_idx, :],
                                                        y[calib_idx], y[test_idx])
    # build new labels for logistic regression [0,1,2..,13]
    experiment = encode_labels(metadata, "experiment")
    exp_train_y = experiment[train_idx].astype(int)
    exp_cal_y = experiment[calib_idx].astype(int)
    exp_test_y = experiment[test_idx].astype(int)

    # Deign matrices Φ_cal, Φ_test
    if not args.group_flag:
        train_feature = features[train_idx, :]
        calib_feature = features[calib_idx, :]
        test_feature = features[test_idx, :]

        best_c = find_best_regularization(train_feature, exp_train_y)  
        phi_cal, phi_test = computeFeatures(train_feature,
                                           calib_feature, test_feature,
                                           exp_train_y, best_c)
    elif args.group_flag:
        # HARD: Φ = one-hot(experiment)
        phi_cal = one_hot_encode(exp_cal_y)
        phi_test = one_hot_encode(exp_test_y)

        if args.add_celltype:
            ct_codes = pd.factorize(metadata["cell_type"])[0] # encode cell types as 0,1,2,...
            phi_cal = np.hstack((phi_cal, one_hot_encode(ct_codes[calib_idx])))
            phi_test = np.hstack([phi_test, one_hot_encode(ct_codes[test_idx])])

    else:
        raise ValueError("group_flag must be either True or False")

    assert phi_cal.shape[0] == len(cal_scores), "Φ_cal rows must match cal_scores length"
    assert phi_test.shape[0] == len(test_scores), "Φ_test rows must match test_scores length"
    assert phi_cal.shape[1] == phi_test.shape[1], "Φ dims must match between calib and test"

    # --- Compute coverages with your finite-basis engine ---
    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha=args.alpha
    )

    compute_prediction_sets(probs_test, q_split, cond_thresholds, saved_dir="results", base_name="pred_sets")


    df_cov_cells = build_cov_df(coverages_split, coverages_cond, metadata['cell_type'].iloc[test_idx], group_name='Cell Type')
    df_cov_experiments = build_cov_df(coverages_split, coverages_cond, metadata['experiment'].iloc[test_idx],
                                    group_name='Experiment')

    save_csv(df_cov_cells, "cells","results")
    save_csv(df_cov_experiments, "experiments","results")
    plot_miscoverage(save_name="Experiment_Cell_Miscoverage.pdf")

    return

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level")
    parser.add_argument("--group_flag", action="store_true", help="Use group_flag")
    parser.add_argument("--soft_flag", action="store_true", help="Use soft_flag")
    parser.add_argument("--add_celltype", action="store_true",
                        help="Add cell type one-hots to features")
    args = parser.parse_args()
    if sum([args.soft_flag, args.group_flag]) != 1:
        parser.error("Exactly one of group or soft must be set.")
    print("runing with args:", args)
    return args

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)