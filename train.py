import os
import torch
import argparse
import numpy as np
from model_builder import SimpleMLP
from data_utils import load_dataloaders
from model_setup import save_checkpoint, load_best_model
from plot_utils import plot_loss_curves
from sklearn.metrics import roc_auc_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_CONFIG = {
    "ChestX": 15,
    "PadChest": 19,
    "VinDr": 28
}

def compute_pos_weight(loader, num_classes):
    """Compute positive class weights for handling class imbalance."""
    pos_counts = torch.zeros(num_classes)
    total_counts = 0
    for _, y in loader:
        pos_counts += y.sum(0)
        total_counts += y.size(0)
    neg_counts = total_counts - pos_counts
    return neg_counts / (pos_counts + 1e-6)


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


def get_scheduler(optimizer, warmup_epochs, total_epochs):
    """Create learning rate scheduler with warmup and cosine annealing."""

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, args):
    """Train model with early stopping and return best model path."""
    os.makedirs(args.ckpt_dir, exist_ok=True)

    learning_curve = {
        "train_loss": [], "train_auroc": [], "train_f1": [],
        "val_loss": [], "val_auroc": [], "val_f1": []
    }

    best_val_auroc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.ckpt_dir, f"best_model_{args.dataset}.pth")
    last_checkpoint_path = os.path.join(args.ckpt_dir, f"last_checkpoint_{args.dataset}.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch + 1}/{args.epochs}\n{'-' * 40}")

        train_loss, train_auroc, train_f1 = train_step(model, train_loader, loss_fn, optimizer)
        val_loss, val_auroc, val_f1 = eval_step(model, val_loader, loss_fn, split_name="Val")

        scheduler.step()

        # Update learning curves
        learning_curve["train_loss"].append(train_loss)
        learning_curve["train_auroc"].append(train_auroc)
        learning_curve["train_f1"].append(train_f1)
        learning_curve["val_loss"].append(val_loss)
        learning_curve["val_auroc"].append(val_auroc)
        learning_curve["val_f1"].append(val_f1)

        save_checkpoint(model, optimizer, epoch + 1, val_loss, val_auroc, last_checkpoint_path)

        # Save best model and check early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            save_checkpoint(model, optimizer, epoch + 1, val_loss, val_auroc, best_model_path)
            print(f"New best model saved! Val AUROC: {val_auroc:.4f}, F1: {val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    plot_loss_curves(learning_curve)
    return best_model_path


def run_inference(model_path, test_loader):
    """Load best model and evaluate on test set."""
    print("\nLoading best model for inference...")
    model = load_best_model(model_path, device)
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            all_probs.append(torch.sigmoid(y_pred).cpu())
            all_targets.append(y.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    auroc, f1 = compute_metrics(all_probs, all_targets)

    print(f"Test AUROC: {auroc:.4f} | Test F1: {f1:.4f}")
    return all_probs, all_targets


def setup_model_and_training(args, train_loader):
    """Initialize model, optimizer, loss function, and scheduler."""
    if args.dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported: {list(DATASET_CONFIG.keys())}")

    model = load_classifier(args.dataset, overwriting=args.overwrite)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    num_classes = DATASET_CONFIG[args.dataset]
    pos_weight = compute_pos_weight(train_loader, num_classes).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = get_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs)

    return model, optimizer, loss_fn, scheduler


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train classifier on extracted features')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

    # Data paths
    parser.add_argument('--train_path', type=str, default="features/ChestX_train.pt")
    parser.add_argument('--calib_path', type=str, default="features/ChestX_calib.pt")
    parser.add_argument('--test_path', type=str, default="features/ChestX_test.pt")

    # Dataset and checkpoint
    parser.add_argument('--dataset', default="ChestX", choices=["ChestX", "PadChest", "VinDr"])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--overwrite', action='store_false', help='Overwrite existing model and retrain')

    return parser.parse_args()


def main():
    args = parse_arguments()
    print(f"Training {args.dataset} | Epochs: {args.epochs} | Batch size: {args.batch_size}")

    train_loader, calib_loader, test_loader = load_dataloaders(
        train_path=args.train_path,
        calib_path=args.calib_path,
        test_path=args.test_path,
        batch_size=args.batch_size
    )

    model, optimizer, loss_fn, scheduler = setup_model_and_training(args, train_loader)
    best_model_path = os.path.join(args.ckpt_dir, f"best_model_{args.dataset}.pth")

    if os.path.exists(best_model_path) and not args.overwrite:
        print(f"Best model already exists at {best_model_path}. Skipping training.")
    else:
        print(f"Training model for {args.dataset}...")
        best_model_path = train_model(
            model, train_loader, calib_loader, optimizer, scheduler, loss_fn, args
        )

    run_inference(best_model_path, test_loader)


if __name__ == "__main__":
    main()


