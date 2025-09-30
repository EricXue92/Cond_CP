import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CalibrationDataset(Dataset):
    def __init__(self, logits, labels):
        self.logits = torch.as_tensor(logits, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, i):
        return self.logits[i], self.labels[i]

# 500,
def temperature_scaling(logits, labels, max_iters=5, lr=0.01, tolerance=1e-4, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(CalibrationDataset(logits, labels), batch_size=batch_size, shuffle=True)

    multilabel_flag = labels.dim() > 1 and labels.shape[1] > 1
    criterion = nn.BCEWithLogitsLoss() if multilabel_flag else nn.CrossEntropyLoss()

    log_T = nn.Parameter(torch.zeros(1, device=device))  # T = exp(log_T), ensures T > 0
    opt = optim.AdamW([log_T], lr=lr)

    for i in range(max_iters):
        T_old = float(torch.exp(log_T).item())
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            T = torch.exp(log_T)
            scaled_logits = x / T
            if multilabel_flag:
                loss = criterion(scaled_logits, y.float())
            else:
                loss = criterion(scaled_logits, y.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
        T_new = float(torch.exp(log_T).item())
        # origin: [0.9360]
        if abs(T_new - T_old) < tolerance:
            print(f"Converged after {i+1} iterations, T = {T_new:.4f}")
            break
    return float(torch.exp(log_T).item())