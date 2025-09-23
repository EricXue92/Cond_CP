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

def temperature_scaling(logits, labels, max_iters=1000, lr=0.1, tolerance=1e-4, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        CalibrationDataset(logits, labels),
        batch_size=batch_size, shuffle=True,
        pin_memory=False
    )

    if labels.dim() > 1 and labels.shape[1]>1:
        criterion = nn.BCEWithLogitsLoss()
        is_multilabel = True
        print(f"Using BCEWithLogitsLoss for multi-label classification ({labels.shape[1]} classes)")
    else:
        criterion = nn.CrossEntropyLoss()
        is_multilabel = False
        print(f"Using CrossEntropyLoss for single-label classification")

    T = nn.Parameter( torch.tensor([1.3], dtype=torch.float32, device=device))
    opt = optim.SGD([T], lr=lr)

    for _ in range(max_iters):
        T_old = float(T.item())
        for batch_logits, batch_labels in loader:
            # Move each minibatch to SAME device as T
            batch_logits = batch_logits.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            scaled_logits = batch_logits / T
            if is_multilabel:
                batch_labels = batch_labels.float()
                loss = criterion(scaled_logits, batch_labels)
            else:
                batch_labels = batch_labels.long()
                loss = criterion(scaled_logits, batch_labels)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if abs(T_old  - T.item()) < tolerance:
            print(f"Converged after {_+1} iterations, T = {T.item():.4f}")
            break
    print(f"Reached max iterations ({max_iters}), T = {T.item():.4f}")
    return T