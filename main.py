from utils import (computeFeatures, find_best_regularization,
                   create_train_calib_test_split, encode_labels, build_cov_df,
                   plot_miscoverage, encode_columns, save_or_append_csv, tune_logreg_c)
from extract_features import load_rxrx_features
from conformal_scores import compute_conformity_scores
import os
import pandas as pd
from conditional_coverage import compute_both_coverages, split_threshold, prediction_sets_form_probs, conditional_thresholds
import numpy as np


def main():
    filepath = 'data/rxrx1_v1.0/rxrx1_features.pt'  # labels: sirna
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Features file not found: {filepath}")
    else:
        features, logits, y= load_rxrx_features(filepath)
        print(f"Feature loaded from: {filepath}")

    metadata = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
    metadata = metadata[metadata['dataset'] == 'test']

    train_idx, calib_idx, test_idx = create_train_calib_test_split(len(features))

    # cal_scores: (n_cal,), test_scores: (n_test,)
    cal_scores, test_scores = compute_conformity_scores(logits[calib_idx, :], logits[test_idx, :],
                                                        y[calib_idx], y[test_idx])



    # build new labels for logistic regression [0,1,2..,13]
    experiment = encode_labels(metadata, "experiment")
    print(f"Created new labels for LogisticRegression: {experiment}")

    # c_values, losses = find_best_regularization(features[train_idx,:],
    #                                            experiment[train_idx], c_range=(0.001, 0.1), n_values=20, cv_folds=5)

    best_c = tune_logreg_c(features[train_idx, :], experiment[train_idx])

    # final_features_cal: (, 14) , final_features_test : (, 14)

    # group indicators but smooth: instead of hard one-hots by experiment, you get soft membership,
    # which is stabler and lets thresholds adapt when a point is ambiguous between experiments.

    final_features_cal, final_features_test = computeFeatures(features[train_idx, :],
                                                          features[calib_idx, :], features[test_idx, :],
                                                          experiment[train_idx], best_c)

    #
    exp_train = experiment[train_idx].astype(int)
    exp_cal = experiment[calib_idx].astype(int)
    exp_test = experiment[test_idx].astype(int)

    K = int( max(exp_cal.max(), exp_test.max()) )+ 1
    PhiCal_hard = np.eye(K)[exp_cal]
    PhiTest_hard = np.eye(K)[exp_test]

    # add cell_type one-hots to Φ(x)
    labels_ct = pd.factorize(metadata["cell_type"])[0] # encode cell types as 0,1,2,...
    print(f"Cell type labels (encoded): {labels_ct}")
    onehot_ct_cal = np.eye(labels_ct.max()+1)[labels_ct[calib_idx]]
    oenhot_ct_test = np.eye(labels_ct.max()+1)[labels_ct[test_idx]]
    # Concatenate with existing features Φ from experiment probabilities
    phi_cal = np.hstack([final_features_cal, onehot_ct_cal])
    phi_test = np.hstack([final_features_test, oenhot_ct_test])






    coverages_split, coverages_cond = compute_both_coverages(final_features_cal, cal_scores, final_features_test, test_scores,
                                                     alpha=0.1)

    metadata, mappings = encode_columns(metadata, ["cell_type", "experiment"])

    # Encode your metadata columns earlier if needed, then:
    df_cov_cells = build_cov_df(coverages_split, coverages_cond, metadata['cell_type'].iloc[test_idx], group_name='Cell Type')
    df_cov_experiments = build_cov_df(coverages_split, coverages_cond, metadata['experiment'].iloc[test_idx],
                                    group_name='Experiment')

    save_or_append_csv(df_cov_cells, "cells.csv")
    save_or_append_csv(df_cov_experiments, "experiments.csv")

    plot_miscoverage(save_name="Experiment_Cell_Miscoverage.pdf")

if __name__ == "__main__":
    main()