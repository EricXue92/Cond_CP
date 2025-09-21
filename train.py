import os
import torch
import argparse
import numpy as np
from model_builder import LinearClassifier
from data_utils import load_dataloaders
from model_setup import save_checkpoint, load_best_model
from plot_utils import plot_loss_curves

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
DATASET_CONFIG = {
    "ChestX": 15,
    "PadChest": 19,
    "VinDr": 28
}

def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss, correct_predictions, total_samples = 0, 0, 0

    for idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(y_pred) > 0.5
        correct_predictions += (preds == y.bool()).float().sum().item()
        total_samples += y.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def eval_step(model, dataloader, loss_fn, split_name="Test"):
    model.eval()
    total_loss, correct_predictions, total_samples = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y.float())

            total_loss += loss.item()
            preds = torch.sigmoid(y_pred) > 0.5
            correct_predictions += (preds == y.bool()).float().sum().item()
            total_samples += y.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    print(f"{split_name} Loss : {avg_loss:.4f} | {split_name} Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# return best model path and last checkpoint path
def train_model(model, train_loader, val_loader, optimizer, loss_fn, args):
    os.makedirs(args.ckpt_dir, exist_ok=True)  # make sure checkpoint dir exists

    learning_curve = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc, best_epoch = 0.0, 0

    best_model_path = os.path.join(args.ckpt_dir, f"best_model_{args.dataset}.pth")
    last_checkpoint_path = os.path.join(args.ckpt_dir, f"last_checkpoint_{args.dataset}.pth")

    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch + 1}/{args.epochs}\n {'-' * 40}")

        # Training step
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = eval_step(model, val_loader, loss_fn, split_name="Val")

        learning_curve["train_loss"].append(train_loss)
        learning_curve["train_acc"].append(train_acc)
        # Validation step
        learning_curve["val_loss"].append(val_loss)
        learning_curve["val_acc"].append(val_acc)

        # Save "last checkpoint" (always overwrite)
        save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, last_checkpoint_path)

        # Save "best model" if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, best_model_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print(f"\n{'=' * 50}")
    print(f"Best model: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%")

    plot_loss_curves(learning_curve)
    return best_model_path, last_checkpoint_path

def run_inference(model_path, test_loader):
    print(f"\n{'=' * 50}")
    print("\nLoading best model for inference...")
    model = load_best_model(model_path, device)
    model.eval()

    predictions, targets, probabilities = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            preds = (torch.sigmoid(y_pred) > 0.5).cpu().numpy()
            probs = torch.sigmoid(y_pred).cpu().numpy()

            predictions.extend(preds)
            targets.extend(y.cpu().numpy())
            probabilities.extend(probs)

    predictions = np.array(predictions)
    targets = np.array(targets)
    probabilities = np.array(probabilities)

    total_labels = targets.size
    correct_labels = np.equal(predictions, targets).sum()
    accuracy = 100.0 * correct_labels / total_labels

    print(f"Multi-label accuracy: {accuracy:.2f}% "
          f"({correct_labels}/{total_labels} labels correct)")
    return predictions, targets, probabilities

def setup_model_and_training(args):
    if args.dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets: {list(DATASET_CONFIG.keys())}")

    num_classes = DATASET_CONFIG[args.dataset]

    model = LinearClassifier(
        in_dim=768,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    return model, optimizer, loss_fn

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train classifier on extracted features')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    # Data paths
    parser.add_argument('--train_path', type=str, default="features/ChestX_train.pt")
    parser.add_argument('--calib_path', type=str, default="features/ChestX_calib.pt")
    parser.add_argument('--test_path', type=str, default="features/ChestX_test.pt")
    # Dataset and checkpoint directory
    parser.add_argument("--dataset", default="ChestX", choices=["ChestX", "PadChest", "VinDr", "MIMIC"])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--overwrite', action='store_false', help='Overwrite existing model and retrain')
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(f"Training {args.dataset} with {args.epochs} epochs, batch size {args.batch_size}")
   # Load data
    train_loader, calib_loader, test_loader = load_dataloaders(
        train_path=args.train_path,
        calib_path=args.calib_path,
        test_path=args.test_path,
        batch_size=args.batch_size
    )
    # Setup model and training components
    model, optimizer, loss_fn = setup_model_and_training(args)

    # Check if model already exists
    best_model_path = os.path.join(args.ckpt_dir, f"best_model_{args.dataset}.pth")

    if os.path.exists(best_model_path):
        if args.overwrite:
            print("Best model already exists but --overwrite is set. Retraining...")
            best_model_path, last_checkpoint_path = train_model(
                model, train_loader, calib_loader, optimizer, loss_fn, args
            )
        else:
            print("Best model already exists. Skipping training.")
    else:
        print(f"Training model for {args.dataset}...")
        best_model_path, last_checkpoint_path = train_model(
            model, train_loader, calib_loader, optimizer, loss_fn, args
        )

    # Run inference on test data
    run_inference(best_model_path, test_loader)

if __name__ == "__main__":
    main()




