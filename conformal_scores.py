import numpy as np
import torch
from temperatureScaling import temperature_scaling
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def compute_conformity_scores(x_calib, x_test, y_calib, y_test, device=None):
#
#     """Compute conformity scores for calibration and test sets."""
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     x_calib = torch.as_tensor(x_calib, dtype=torch.float32).to(device)
#     x_test = torch.as_tensor(x_test, dtype=torch.float32).to(device)
#
#     y_calib = torch.as_tensor(y_calib, dtype=torch.long).to(device)
#     y_test = torch.as_tensor(y_test, dtype=torch.long).to(device)
#
#     T = temperature_scaling(x_calib, y_calib)
#     if isinstance(T, torch.Tensor):
#         T = float(T.detach().cpu().numpy())
#     T = max(T, 1e-6)  # Ensure T is positive
#
#     probs_calib = F.softmax(x_calib / T, dim=-1)
#     probs_test = F.softmax(x_test / T, dim=-1)
#
#     # ----- vectorized "sum of probs > p_true" conformity score
#     p_true_cal = probs_calib.gather(1, y_calib.view(-1,1)).squeeze(1)        # (N_cal,)
#     p_true_tst = probs_test.gather(1,  y_test.view(-1,1)).squeeze(1)         # (N_tst,)
#
#     scores_calib = (probs_calib * (probs_calib > p_true_cal.unsqueeze(1))).sum(dim=1)
#     scores_test  = (probs_test  * (probs_test  > p_true_tst.unsqueeze(1))).sum(dim=1)
#
#     return ( scores_calib.cpu().numpy(), scores_test.cpu().numpy(),
#              probs_calib.detach().cpu().numpy(), probs_test.detach().cpu().numpy() )

#
# def compute_conformity_scores(x_calib, x_test, y_calib, y_test, device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Convert to tensors
#     x_calib = torch.as_tensor(x_calib, dtype=torch.float32).to(device)
#     x_test = torch.as_tensor(x_test, dtype=torch.float32).to(device)
#     y_calib = torch.as_tensor(y_calib, dtype=torch.float32).to(device)  # (N, C)
#     y_test = torch.as_tensor(y_test, dtype=torch.float32).to(device)
#
#     # Multi-label → sigmoid probabilities
#     probs_calib = torch.sigmoid(x_calib)
#     probs_test = torch.sigmoid(x_test)
#
#     # Only keep probabilities for true labels (per-class conformity)
#     # Flatten into (num_true_labels,) for calibration and test
#     p_true_cal = probs_calib * y_calib
#     p_true_tst = probs_test * y_test
#
#     # Gather non-zero entries (true labels)
#     scores_calib = (1 - p_true_cal[y_calib == 1]).detach().cpu().numpy()
#     scores_test = (1 - p_true_tst[y_test == 1]).detach().cpu().numpy()
#
#     return (
#         scores_calib,
#         scores_test,
#         probs_calib.detach().cpu().numpy(),
#         probs_test.detach().cpu().numpy()
#     )

# import torch
# import torch.nn.functional as F
#
# def compute_conformity_scores(x_calib, x_test, y_calib, y_test):
#     """Compute conformity scores (one per sample) for multi-label or single-label."""
#     # Ensure tensors
#     x_calib = torch.as_tensor(x_calib, dtype=torch.float32, device=device)
#     x_test  = torch.as_tensor(x_test,  dtype=torch.float32, device=device)
#
#     y_calib = torch.as_tensor(y_calib, device=device)
#     y_test  = torch.as_tensor(y_test,  device=device)
#
#     # Temperature scaling
#     T = temperature_scaling(x_calib, y_calib)
#     if isinstance(T, torch.Tensor):
#         T = float(T.detach().cpu().numpy())
#     T = max(T, 1e-6)
#
#     # Probabilities
#     probs_calib = F.softmax(x_calib / T, dim=-1)
#     probs_test  = F.softmax(x_test / T,  dim=-1)
#
#     # # ---- One conformity score per sample ----
#     def expand_scores(probs, y):
#         scores, idx_map = [], []
#         y = y.float()
#         # case 1: single-label (1D or shape (N,))
#         if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
#             y = y.view(-1).long()
#             p_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
#             s = (probs * (probs > p_true.unsqueeze(1))).sum(dim=1)
#             for i, yi in enumerate(y.tolist()):
#                 scores.append(s[i].item())
#                 idx_map.append((i, yi))
#
#         # take the minimum conformity score across all true labels (most conservative).
#         else:
#             for i in range(y.shape[0]):
#                 positives = (y[i] > 0).nonzero(as_tuple=True)[0]
#                 if len(positives) == 0:
#                     continue
#                 s_list = []
#                 for cls in positives:
#                     p_true = probs[i, cls]
#                     s = (probs[i] * (probs[i] > p_true)).sum()
#                     s_list.append(s.item())
#                 scores.append(min(s_list))
#                 idx_map.append((i, positives.tolist()))
#         return scores, idx_map
#
#     scores_calib, idx_calib = expand_scores(probs_calib, y_calib)
#     scores_test,  idx_test  = expand_scores(probs_test,  y_test)
#
#     return (
#         np.array(scores_calib),
#         np.array(scores_test),
#         probs_calib.detach().cpu().numpy(),
#         probs_test.detach().cpu().numpy(),
#     )

import torch
import torch.nn.functional as F
import numpy as np

def compute_conformity_scores(x_calib, x_test, y_calib, y_test, device=None):
    """Compute conformity scores (one per sample) for multi-label or single-label."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure tensors
    x_calib = torch.as_tensor(x_calib, dtype=torch.float32, device=device)
    x_test  = torch.as_tensor(x_test,  dtype=torch.float32, device=device)
    y_calib = torch.as_tensor(y_calib, device=device)
    y_test  = torch.as_tensor(y_test,  device=device)

    # Temperature scaling
    T = temperature_scaling(x_calib, y_calib)
    if isinstance(T, torch.Tensor):
        T = float(T.detach().cpu().numpy())
    T = max(T, 1e-6)

    # Probabilities
    probs_calib = F.softmax(x_calib / T, dim=-1)
    probs_test  = F.softmax(x_test / T,  dim=-1)

    # ---- One conformity score per sample ----
    def per_sample_scores(probs, y):
        scores = []
        y = y.float()
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            # Single-label
            y = y.view(-1).long()
            p_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
            s = (probs * (probs > p_true.unsqueeze(1))).sum(dim=1)
            scores = s.tolist()
        else:
            # Multi-label: min over positives
            for i in range(y.shape[0]):
                positives = (y[i] > 0).nonzero(as_tuple=True)[0]
                if len(positives) == 0:
                    scores.append(0.0)  # or np.nan if you want to skip
                    continue
                s_list = []
                for cls in positives:
                    p_true = probs[i, cls]
                    s = (probs[i] * (probs[i] > p_true)).sum()
                    s_list.append(s.item())
                scores.append(min(s_list))
        return np.array(scores)

    scores_calib = per_sample_scores(probs_calib, y_calib)
    scores_test  = per_sample_scores(probs_test,  y_test)

    return (
        scores_calib,
        scores_test,
        probs_calib.detach().cpu().numpy(),
        probs_test.detach().cpu().numpy(),
    )
