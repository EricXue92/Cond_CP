import cvxpy as cp
import numpy as np
from tqdm import tqdm
from conditionalconformal.condconf import setup_cvx_problem_calib
from utils import split_threshold

import warnings
warnings.filterwarnings("ignore",
                       message="You didn't specify the order of the vec expression")


# phi_cal: (n_cal, d) calibration features; scores_cal: (n_cal,) calibration scores ;
# phi_test: (n_test, d) test features
def conditional_thresholds(phi_cal, scores_cal, phi_test, alpha):
    """
    Solve the finite-basis LP once to get beta (dual of eta^T Phi == 0),
    then t_i = Phi(x_i)^T @ beta for every test point.
    """
    phi_cal, phi_test = np.asarray(phi_cal, dtype=float), np.asarray(phi_test, dtype=float)
    scores_cal = np.asarray(scores_cal, dtype=float).ravel()
    # LP over calibration ONLY (no test rows needed in this finite-basis path)
    prob = setup_cvx_problem_calib(quantile=1.0-alpha,
                                   x_calib=None,
                                   scores_calib=scores_cal,
                                   phi_calib=phi_cal,
                                   infinite_params={} )
    if "MOSEK" in cp.installed_solvers():
        prob.solve(solver="MOSEK")
    else:
        prob.solve(verbose=False)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"LP status: {prob.status}")
    # equality constraint is the 3rd one in setup_cvx_problem_calib (index 2)
    beta = np.asarray(prob.constraints[2].dual_value, dtype=float).ravel()# shape (d, )
    return phi_test @ beta     # shape (n_test, )

# Prediction sets from probabilities
def prediction_sets_form_probs(probs, thresholds):
    """
        probs: (n_test, K) class probabilities
        thresholds: scalar or array (n_test,) of score thresholds t.
                    Label k is included iff p_k >= 1 - t.
        Returns: (list_of_arrays prediction_sets, point_preds)
        """
    probs = np.asarray(probs, dtype=float)
    n_test, K = probs.shape
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    if thresholds.size == 1:
        thresholds = np.full(n_test, float(thresholds[0]))
    thresholds = np.clip(thresholds, 0, 1)
    pred_sets = [np.where(probs[i] >= 1.0 - thresholds[i])[0] for i in range(n_test) ]
    point_preds = np.argmax(probs, axis=1)
    return pred_sets, point_preds

# End-to-end (classification)
def classification_prediction_sets(phi_cal, y_cal, probs_cal, phi_test, probs_test, alpha=0.1, exact=True):
    """
        x_cal, x_test are Phi(x) features (finite basis).
         probs_cal/probs_test are class probabilities.
    """
    phi_cal, phi_test = np.asarray(phi_cal, dtype=float), np.asarray(phi_test, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)

    probs_cal = np.asarray(probs_cal, dtype=float)
    probs_test = np.asarray(probs_test, dtype=float)

    # calibration scores: 1 - p_{y_i}
    scores_cal = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]

    # split threshold (same for everyone)
    t_split = split_threshold(scores_cal, alpha)

    # conditional thresholds (per x via Phi(x)^T beta)
    t_cond = conditional_thresholds(phi_cal, scores_cal, phi_test, alpha)

    # prediction sets + point labels
    split_sets, split_labels = prediction_sets_form_probs(probs_test, t_split)
    cond_sets,  cond_labels  = prediction_sets_form_probs(probs_test, t_cond)

    return {
        'split': {'sets': split_sets, 'labels': split_labels},
        'cond':  {'sets': cond_sets,  'labels': cond_labels},
        'thresholds': {'split': t_split, 'cond': t_cond}
    }



# Xcal and XTest are the Phi(x) features for calibration set and test set respectively
def compute_both_coverages(x_cal, scores_cal, x_test, scores_test, alpha):

    x_cal, x_test = np.asarray(x_cal, dtype=float), np.asarray(x_test, dtype=float)
    scores_cal, scores_test = np.asarray(scores_cal, dtype=float).ravel(), np.asarray(scores_test, dtype=float).ravel()

    # split coverage
    # q_idx = math.ceil((1 - alpha) * (len(scores_cal) + 1)) / len(scores_cal)
    # qSplit = np.quantile(scores_cal, q_idx)
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
            coveragesCond[i] = scores_test[i] <= t_i
        except Exception as e:
            # fail safe: mark as not covered (or choose a policy)
            coveragesCond[i] = False
    return coveragesSplit, coveragesCond
