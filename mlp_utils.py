import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.network(x)

    def predict_probs(self, x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            logits = self(x.to(device))
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

def compute_metrics(probs, targets):
    preds = probs.argmax(axis=1)
    acc = (preds == targets).mean()
    return acc

def run_epoch(model, dataloader, loss_fn, optimizer=None, split_name="Train"):
    """Run one epoch of training or evaluation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0
    all_probs, all_targets = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu())
            all_targets.append(y.detach().cpu())

    avg_loss = total_loss / len(dataloader)
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    acc = compute_metrics(all_probs, all_targets)

    print(f"{split_name} Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    return avg_loss, acc

def train_with_early_stopping(model, train_loader, val_loader, loss_fn, optimizer,
                               epochs=100, patience=10):
    """Train model with early stopping"""
    best_val_loss, best_state, patience_counter = float("inf"), None, 0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss,  train_acc = run_epoch(model, train_loader, loss_fn, optimizer, split_name="Train")
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, split_name="Val")

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
