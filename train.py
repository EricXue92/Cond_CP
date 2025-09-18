import torch
from model_builder import LinearClassifier
from datasets import load_dataloaders
import argparse
import os
from model_setup import save_checkpoint, load_best_model
from utils import plot_loss_curves

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, calib_loader, test_loader = load_dataloaders(
            train_path=args.train_path,
            calib_path=args.calib_path,
            test_path=args.test_path,
            batch_size=args.batch_size)
    # Build model
    if args.dataset == "ChestX":
        num_classes = 15
    elif args.dataset == "PadChest":
        num_classes = 19
    elif args.dataset == "VinDr":
        num_classes = 28
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    model = LinearClassifier(in_dim=768, num_classes=num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = os.path.join(args.ckpt_dir, f"best_model_{args.dataset}.pth")
    last_checkpoint_path = os.path.join(args.ckpt_dir, f"last_checkpoint_{args.dataset}.pth")

    # --- Training ---
    if os.path.exists(best_model_path):
        print("Best model already exists. Skipping training.")
    else:
        print(f"Training model for {args.dataset}...")
        learning_curve, best_model_path, last_checkpoint_path = train_model(
            model, train_loader, calib_loader, optimizer, loss_fn, device,
            epochs=args.epochs, ckpt_dir=args.ckpt_dir
        )
    # -- Inference on test data --
    test_predictions, test_targets, test_probs = inference(best_model_path, test_loader, device)
    print("Training complete!")
    print(f"Saved best model to {best_model_path}")
    print(f"Saved last checkpoint to {last_checkpoint_path}")
    print(f"Test predictions shape: {len(test_predictions)}, Test targets shape: {len(test_targets)}")


def train_step(model, train_loader, loss_fn, optimizer, device):
    train_loss, train_acc = 0, 0
    total_samples = 0
    model = model.to(device)
    model.train()

    for idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        if y.ndim > 1:
            y = torch.argmax(y, dim=1)

        y_pred = model(X)
        loss = loss_fn(y_pred, y.float())
        train_loss += loss.item()
        preds = torch.sigmoid(y_pred) > 0.5
        train_acc += (preds == y.bool()).float().mean().item()
        total_samples += y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100 * train_acc  / total_samples
    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def test_step(model, data_loader, loss_fn, device):
    test_loss, test_acc = 0.0, 0.0
    total_samples = 0
    model = model.to(device)
    model.eval()

    prob_list, target_list = [], []

    with torch.no_grad():
        for X, y in data_loader:
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            prob_list.append(y_pred.detach().cpu())
            target_list.append(y.detach().cpu())

            preds = torch.sigmoid(y_pred) > 0.5
            test_acc += (preds == y.bool()).float().mean().item()
            total_samples += y.size(0)

    avg_loss = test_loss / len(data_loader)
    accuracy = 100 * test_acc / total_samples
    print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")
    # # Concatenate predictions and labels
    # prob = torch.cat(prob_list, dim=0)
    # target = torch.cat(target_list, dim=0)
    return avg_loss, accuracy

def train_model(model, train_loader, calib_loader, optimizer, loss_fn, device,
                epochs=10, ckpt_dir="checkpoints", dataset="ChestX"):
    os.makedirs(ckpt_dir, exist_ok=True)  # make sure checkpoint dir exists
    learning_curve = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_val_acc, best_epoch = 0.0, 0
    best_model_path = os.path.join(ckpt_dir, f"best_model_{dataset}.pth")
    last_checkpoint_path = os.path.join(ckpt_dir, f"last_checkpoint_{dataset}.pth")

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}/{args.epochs}\n {'-' * 40}")

        # Training step
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        learning_curve["train_loss"].append(train_loss)
        learning_curve["train_acc"].append(train_acc)

        # Validation step
        val_loss, val_acc = test_step(model, calib_loader, loss_fn, device)
        learning_curve["val_loss"].append(val_loss)
        learning_curve["val_acc"].append(val_acc)


        # Save "last checkpoint" (always overwrite)
        save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, last_checkpoint_path)
        print(f"Last checkpoint saved: {last_checkpoint_path}")

        # Save "best model" if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, epoch + 1, val_loss, val_acc, best_model_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print(f"\nTraining completed!")
    print(f"Best model: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%")

    plot_loss_curves(learning_curve)
    return learning_curve, best_model_path, last_checkpoint_path

def inference(model_path, test_loader, device):
    print("Loading best model for inference...")
    model = load_best_model(model_path, device)

    test_predictions,test_targets, test_probs = [], [], []
    model.eval()

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            preds = torch.argmax(y_pred, dim=1)
            probs = torch.softmax(y_pred, dim=1)

            test_predictions.extend(preds.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    return test_predictions, test_targets, test_probs

def parse_arguments():
    parser = argparse.ArgumentParser(description=f'Train classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--train_path', type=str, default="features/ChestX_train.pt")
    parser.add_argument('--calib_path', type=str, default="features/ChestX_calib.pt")
    parser.add_argument('--test_path', type=str, default="features/ChestX_test.pt")
    parser.add_argument("--dataset", default="ChestX",
                        choices=["ChestX", "PadChest", 'VinDr', "MIMIC"])
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory to save checkpoints (default: checkpoints)')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Training with {args.epochs} epochs, batch size {args.batch_size}")
    main(args)






