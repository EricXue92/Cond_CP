import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
from conditionalconformal.condconf import setup_cvx_problem_calib
from utils import split_threshold
import os


def compute_prediction_sets(probs_test, q_split, cond_thresholds,
                            dataset_name, saved_dir, base_name, analysis_type):

    probs_test = np.array(probs_test, dtype=np.float32)
    n_test, K = probs_test.shape

    if np.ndim(q_split) > 0 and len(q_split) > 1:
        raise ValueError(f"q_split should be scalar for single-label datasets, got shape {q_split.shape}")

    q_split = float(np.squeeze(q_split))

    if np.ndim(cond_thresholds) == 0:
        cond_thresholds = np.full(n_test, cond_thresholds, dtype=np.float32)
    else:
        cond_thresholds = np.array(cond_thresholds, dtype=np.float32)
        if len(cond_thresholds) != n_test:
            raise ValueError(f"cond_thresholds length {len(cond_thresholds)} != n_test {n_test}")

    split_sets, split_sizes, cond_sets, cond_sizes  = [],[],[],[]
    for i in range(n_test):
        split_set = np.where(probs_test[i] >= (1 - q_split))[0]
        split_sets.append(split_set.tolist())
        split_sizes.append(len(split_set))

        cond_set = np.where(probs_test[i] >= (1.0 - cond_thresholds[i]))[0]
        cond_sets.append(cond_set.tolist())
        cond_sizes.append(len(cond_set))

    split_thresholds = np.full(n_test, round(q_split, 4), dtype=np.float32)
    print("Lens:", n_test,
          len(split_sets), len(split_sizes),
          len(cond_sets), len(cond_sizes),
          len(split_thresholds), len(cond_thresholds))

    df = pd.DataFrame({
        "Index": range(n_test),
        "Split_Set": [str(s) for s in split_sets],
        "Split_Size": split_sizes,
        "Split_Threshold": round(q_split, 4),
        "Cond_Set": [str(s) for s in cond_sets],
        "Cond_Size": cond_sizes,
        "Cond_Threshold": np.round(cond_thresholds, 4)
    })

    # Save with timestamp
    os.makedirs(saved_dir, exist_ok=True)
    filename = f"{base_name}_{dataset_name}_{analysis_type}.csv"
    filepath = os.path.join(saved_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[INFO] Saved: {filepath}")
    return

# Xcal and XTest are the Phi(x) features for calibration set and test set respectively
def compute_both_coverages(x_cal, scores_cal, x_test, scores_test, alpha):

    x_cal, x_test = np.asarray(x_cal, dtype=float), np.asarray(x_test, dtype=float)
    scores_cal, scores_test = np.asarray(scores_cal, dtype=float).ravel(), np.asarray(scores_test, dtype=float).ravel()

    cond_thresholds = np.zeros(len(x_test))
    # split coverage
    q_split = split_threshold(scores_cal, alpha)
    coveragesSplit = scores_test <= q_split

    #conditional coverage (finite-basis; no RKHS)
    coveragesCond = np.zeros(len(x_test), dtype=bool)

    # Add the test point’s score onto the calibration scores
    for i in tqdm(range(len(x_test))):
        prob = setup_cvx_problem_calib(quantile=1-alpha,
                                       x_calib=None,  #  (unused in finite-basis path)
                                       scores_calib=np.concatenate( (scores_cal, np.array([scores_test[i]])) ), # scores_calib = [scoresCal, scoresTest[i]]
                                       phi_calib=np.vstack( (x_cal, x_test[i,:]) ), # phi_calib = [XCal; XTest[i,:]]
                                       infinite_params={} )
        try:
            if "MOSEK" in cp.installed_solvers():
                prob.solve(solver="MOSEK")
            else:
                prob.solve()
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise RuntimeError(prob.status)
            # Dual of (eta.T @ Phi == 0) is beta
            beta = np.asarray(prob.constraints[2].dual_value).ravel() # shape (d, )
            t_i = float(x_test[i,:] @ beta)   # per-x threshold t(x)= Phi(x_i)^T beta
            cond_thresholds[i] = t_i
            coveragesCond[i] = scores_test[i] <= t_i
        except Exception as e:
            coveragesCond[i] = False
    return coveragesSplit, coveragesCond, q_split, cond_thresholds


def run_conformal_analysis(phi_cal, phi_test, cal_scores, test_scores,
                           probs_test, alpha, dataset_name, analysis_type):

    assert phi_cal.shape[0] == len(cal_scores), "Φ_cal rows must match cal_scores length"
    assert phi_test.shape[0] == len(test_scores), "Φ_test rows must match test_scores length"
    assert phi_cal.shape[1] == phi_test.shape[1], "Φ dimensions must match between calib and test"

    coverages_split, coverages_cond, q_split, cond_thresholds = compute_both_coverages(
        phi_cal, cal_scores, phi_test, test_scores, alpha
    )

    compute_prediction_sets(
        probs_test, q_split, cond_thresholds, dataset_name,
        "results",  f"{dataset_name}_pred_sets", analysis_type
    )
    return coverages_split, coverages_cond

