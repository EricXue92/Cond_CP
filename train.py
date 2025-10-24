import os
import numpy as np
import pandas as pd
from pathlib import Path

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import set_seed
from data_utils import NIHDataset
from model_builder import create_medical_model

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
               'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
               'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']


def create_transform(img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform

def make_df_and_path(args):
    base_dir = Path(args.data_dir) / args.data_name
    csv_path = base_dir / "Data_Entry_2017_clean.csv"
    img_path = base_dir / "images"
    if not csv_path.exists() or not img_path.is_dir():
        raise FileNotFoundError(f"Missing data files in {base_dir}")
    df = pd.read_csv(csv_path)
    return df, img_path

def split_indices(df, seed=42):
    idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx

# def save_split(args, train_idx, val_idx, test_idx):
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
#     split_info = {
#         "train_idx": train_idx,
#         "val_idx": val_idx,
#         "test_idx": test_idx,
#         "pathologies": PATHOLOGIES,
#     }
#     torch.save(split_info, os.path.join(args.checkpoint_dir, f"split_indices_{args.data_name}.pth"))

def make_loader(df, img_path, indices, tfm):
    ds = NIHDataset(df, img_path, indices, PATHOLOGIES, tfm)
    return DataLoader(ds, batch_size=128, shuffle=True,
                      num_workers=4, pin_memory=True)

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, num_batches = 0, 0
    for images, labels in tqdm(loader, desc="[Val]", leave=False):
        images, labels = images.to(device), labels.to(device)
        loss = loss_fn(model(images), labels)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def train(args):
    df, img_path = make_df_and_path(args)
    train_idx, val_idx, test_idx = split_indices(df, args.seed)
    # save_split(args, train_idx, val_idx, test_idx)

    tfm = create_transform(args.img_size)

    train_loader = make_loader(df, img_path, train_idx, tfm)
    val_loader = make_loader(df, img_path, val_idx, tfm)

    model = create_medical_model(model_name=args.pretrained_model,
                                 num_classes=len(PATHOLOGIES)).to(device)

    print(f"Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable / "
          f"{sum(p.numel() for p in model.parameters()):,} total")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = None
    if args.scheduler:
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epochs
            )
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )

    best_val_loss = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs+1):
        model.train()
        total_loss, num_batches = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Train]")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': total_loss / num_batches})

        avg_train_loss = total_loss / num_batches
        avg_val_loss = evaluate(model, val_loader, loss_fn)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d}/{args.num_epochs:02d}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, "
              f"LR = {current_lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.checkpoint_dir, f"best_model_{args.data_name}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, ckpt_path)
            print(f"Saved best model (val_loss={avg_val_loss:.4f})")

        if scheduler:
            scheduler.step()

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to:       {args.checkpoint_dir}/best_model_{args.data_name}.pth")
    print(f"Split indices saved:  {args.checkpoint_dir}/split_indices_{args.data_name}.pth")

    return test_idx, df, img_path, tfm


def load_best_ckpt(model, args):
    ckpt = os.path.join(args.checkpoint_dir, f"best_model_{args.data_name}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Run with --mode train first (or set --checkpoint_dir correctly)."
        )
    blob = torch.load(ckpt, map_location=device)
    model.load_state_dict(blob["model_state_dict"])
    return blob

def load_saved_split(args, df):
    split_file = os.path.join(args.checkpoint_dir, f"split_indices_{args.data_name}.pth")
    if os.path.exists(split_file):
        split_info = torch.load(split_file)
        return split_info["train_idx"], split_info["val_idx"], split_info["test_idx"]
    # If not found, recompute (ensures extract still works)
    print("[WARN] split_indices file not found; recomputing 60/20/20 split with same seed.")
    return split_indices(df, args.seed)

def extract_logits_labels(args):
    df, img_path = make_df_and_path(args)
    _, _, test_idx = load_saved_split(args, df)
    tfm = create_transform(args.img_size)
    test_loader = make_loader(df, img_path, test_idx, tfm)
    model = create_medical_model(model_name=args.pretrained_model,
                                 num_classes=len(PATHOLOGIES)).to(device)
    load_best_ckpt(model, args)
    model.eval()

    print(f"Creating test dataloader from {len(test_idx)} samples...")
    print("Extracting features and logits...")

    all_logits, all_labels = [], []

    for images, labels in tqdm(test_loader, desc="[Test]"):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    test_metadata = df.iloc[test_idx].reset_index(drop=True)

    data_for_conformal = {
        "logits": logits,
        "labels": labels,
        "metadata": test_metadata
    }

    save_path = os.path.join(args.checkpoint_dir, f"test_data_{args.data_name}.pt")
    torch.save(data_for_conformal, save_path)

    print(f"✓ Logits shape:   {logits.shape}")
    print(f"✓ Labels shape:   {labels.shape}")
    print(f"✓ Metadata rows:  {len(test_metadata)}")
    print(f"✓ Saved to:       {save_path}")
    print(f"  File size:      {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train medical image classifier on NIH dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "extract"], help="train: fit model and save checkpoint; extract: skip training and only export test logits/labels/metadata")
    # Data
    parser.add_argument("--data_name", type=str, default="NIH", choices=["NIH", "ChestX"], help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to NIH dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",help="Directory to save checkpoints")
    # Model
    parser.add_argument("--pretrained_model", type=str, default="densenet121", choices=["densenet121","resnet50"], help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights")

    # Training
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,help="Initial learning rate")
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, "cosine", "step"], help="Learning rate scheduler")

    parser.add_argument("--img_size", type=int, default=224, help="Image size (height and width)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.mode == "train":
        train(args)
        print("\nTraining done. If you want to export test logits later, run with --mode extract.\n")
    elif args.mode == "extract":
        extract_logits_labels(args)
    else:
        raise ValueError("Unknown mode '{args.mode}'")

if __name__ == "__main__":
    main()
