import numpy as np
import torch
from temperatureScaling import temperature_scaling
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_conformity_scores(x_calib, x_test, y_calib, y_test):
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
    # Sum of probs greater than p_true
    # low score means "easier" sample, smell set
    # high score means "harder" sample, larger set
    # For multi-label, take the min over all true labels (most conservative)

    def per_sample_scores(probs, y, agg="min"):
        scores = []
        y = y.float()
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y = y.view(-1).long()
            p_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
            s = (probs * (probs > p_true.unsqueeze(1))).sum(dim=1)
            scores = s.tolist()
        else:
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
                if agg == "min":
                    scores.append(min(s_list)) # coverage as long as any positive is well-ranked.
                elif agg == "mean":
                    scores.append(float(np.mean(s_list))) # coverage only if all positives are well-ranked.
                elif agg == "max":
                    scores.append(max(s_list)) #  coverage only if all positives are well-ranked
                else:
                    raise ValueError(f"Unknown agg method: {agg}")
        return np.array(scores)

    scores_calib = per_sample_scores(probs_calib, y_calib)
    scores_test  = per_sample_scores(probs_test,  y_test)

    return (
        scores_calib,
        scores_test,
        probs_calib.detach().cpu().numpy(),
        probs_test.detach().cpu().numpy(),
    )
