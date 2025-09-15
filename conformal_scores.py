import numpy as np
from wandb.integration.torch.wandb_torch import torch
import torch
from temperatureScaling import temperature_scaling
import torch.nn.functional as F

def compute_conformity_scores(x_calib, x_test, y_calib, y_test, device=None):

    """Compute conformity scores for calibration and test sets."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_calib = torch.as_tensor(x_calib, dtype=torch.float32).to(device)
    x_test = torch.as_tensor(x_test, dtype=torch.float32).to(device)
    y_calib = torch.as_tensor(y_calib, dtype=torch.long).to(device)
    y_test = torch.as_tensor(y_test, dtype=torch.long).to(device)

    T = temperature_scaling(x_calib, y_calib)
    if isinstance(T, torch.Tensor):
        T = float(T.detach().cpu().numpy())
    T = max(T, 1e-6)  # Ensure T is positive

    probs_calib = F.softmax(x_calib / T, dim=-1)
    probs_test = F.softmax(x_test / T, dim=-1)

    # ----- vectorized "sum of probs > p_true" conformity score
    p_true_cal = probs_calib.gather(1, y_calib.view(-1,1)).squeeze(1)        # (N_cal,)
    p_true_tst = probs_test.gather(1,  y_test.view(-1,1)).squeeze(1)         # (N_tst,)

    scores_calib = (probs_calib * (probs_calib > p_true_cal.unsqueeze(1))).sum(dim=1)
    scores_test  = (probs_test  * (probs_test  > p_true_tst.unsqueeze(1))).sum(dim=1)

    return ( scores_calib.cpu().numpy(), scores_test.cpu().numpy(),
             probs_calib.detach().cpu().numpy(), probs_test.detach().cpu().numpy() )