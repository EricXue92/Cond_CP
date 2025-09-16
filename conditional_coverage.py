import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
from conditionalconformal.condconf import setup_cvx_problem_calib
from utils import split_threshold
import os
from datetime import datetime

def compute_prediction_sets(probs_test, q_split, cond_thresholds,
                            saved_dir="results", base_name="pred_sets"):

    probs_test = np.array(probs_test, dtype=np.float32)
    n_test, K = probs_test.shape

    split_sets, split_sizes, cond_sets, cond_sizes  = [],[],[],[]
    for i in range(n_test):
        split_set = np.where(probs_test[i] >= (1 - q_split))[0]
        split_sets.append(split_set.tolist())
        split_sizes.append(len(split_set))

        cond_set = np.where(probs_test[i] >= (1.0 - cond_thresholds[i]))[0]
        cond_sets.append(cond_set.tolist())
        cond_sizes.append(len(cond_set))

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.csv"
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
            # fail safe: mark as not covered (or choose a policy)
            coveragesCond[i] = False
    return coveragesSplit, coveragesCond, q_split, cond_thresholds

