
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x

    def predict_probs(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            x = x.to(device or next(self.parameters()).device)
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

def compute_metrics(probs, targets):
    """Compute AUROC and F1 scores."""
    auroc = roc_auc_score(targets, probs, average="macro")
    f1 = f1_score(targets, (probs > 0.5).astype(int), average="macro")
    return auroc, f1

def train_step(model, dataloader, loss_fn, optimizer):
    """Execute one training epoch."""
    model.train()
    total_loss = 0
    all_probs, all_targets = [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_probs.append(torch.sigmoid(y_pred).detach().cpu())
        all_targets.append(y.detach().cpu())
    avg_loss = total_loss / len(dataloader)
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    auroc, f1 = compute_metrics(all_probs, all_targets)
    print(f"Train Loss: {avg_loss:.4f} | AUROC: {auroc:.4f} | F1: {f1:.4f}")
    return avg_loss, auroc, f1


def eval_step(model, dataloader, loss_fn, split_name="Val"):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_probs, all_targets = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y.float())
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(y_pred).cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / len(dataloader)
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    auroc, f1 = compute_metrics(all_probs, all_targets)

    print(f"{split_name} Loss: {avg_loss:.4f} | AUROC: {auroc:.4f} | F1: {f1:.4f}")
    return avg_loss, auroc, f1

def train_with_early_stopping(model, train_loader, val_loader, loss_fn, optimizer,
                               epochs=100, patience=10):
    """Train model with early stopping"""
    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    history = {"train_loss": [], "train_auroc": [], "train_f1": [],
               "val_loss": [], "val_auroc": [], "val_f1": []}


    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_auroc, train_f1 = train_step(model, train_loader, loss_fn, optimizer)
        val_loss, val_auroc, val_f1 = val_loss = eval_step(model, val_loader, loss_fn, split_name="Val")
          # record history
        history["train_loss"].append(train_loss)
        history["train_auroc"].append(train_auroc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_f1"].append(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_loss
