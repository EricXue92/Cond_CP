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

from utils import set_seed, plot_loss_curves
from data_utils import NIHDataset
from model_builder import create_medical_model

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
               'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
               'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def create_transform(img_size=224, is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),             # ← Flip
            transforms.RandomRotation(degrees=10),              # ← Rotate
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # ← Color
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x[:3]),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x[:3]),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

def read_df_and_path(args, csv_path="Data_Entry_2017_clean.csv", img_path="images"):
    base_dir = Path(args.data_dir) / args.data_name
    csv_path = base_dir / csv_path
    img_path = base_dir / img_path
    if not csv_path.exists() or not img_path.is_dir():
        raise FileNotFoundError(f"Missing data files in {base_dir}")
    df = pd.read_csv(csv_path)
    return df, img_path

def split_indices(df, seed=42):
    idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx

def save_split(args, train_idx, val_idx, test_idx):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    split_info = {"train_idx": train_idx,"val_idx": val_idx,"test_idx": test_idx,"pathologies": PATHOLOGIES}
    torch.save(split_info, os.path.join(args.checkpoint_dir, f"split_indices_{args.data_name}.pth"))

def make_loader(df, img_path, indices, tfm, batch=128, workers=4, shuffle=True):
    ds = NIHDataset(df, img_path, indices, PATHOLOGIES, tfm)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,num_workers=workers, pin_memory=True)

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

class BCEWithLogitsLossSmooth(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_smooth)

def train_step(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, num_batches = 0, 0
    for x, y in tqdm(loader, desc="[Train]", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches

def train(args):
    # df, img_path = read_df_and_path(args, csv_path="Data_Entry_2017_clean.csv", img_path="images") # NIH
    df, img_path = read_df_and_path(args, csv_path="foundation_fair_meta/metadata_all.csv", img_path="images-224")  # NIH
    train_idx, val_idx, test_idx = split_indices(df, args.seed)
    save_split(args, train_idx, val_idx, test_idx)

    train_tfm = create_transform(is_train=True)
    train_loader = make_loader(df, img_path, train_idx, train_tfm)
    val_loader = make_loader(df, img_path, val_idx, train_tfm)
    model = create_medical_model(model_name=args.pretrained_model, num_classes=len(PATHOLOGIES)).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = BCEWithLogitsLossSmooth(smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )
    elif args.scheduler =="reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
    else:
        scheduler = None

    best_val_loss, train_losses, val_losses = float("inf"), [], []
    epochs_no_improve = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.num_epochs+1):
        train_loss = train_step(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Step the scheduler
        if args.scheduler == 'reduce_on_plateau':
            scheduler.step(val_loss)
        elif args.scheduler in ["cosine", "step"]:
            scheduler.step()
        else:
            print(f"Unknown scheduler {args.scheduler}")
        print(f"Epoch {epoch:02d}/{args.num_epochs:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.checkpoint_dir, f"best_model_{args.data_name}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, ckpt_path)
            print(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement since {epochs_no_improve} epochs")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (patience={args.patience}).")
            break

    plot_loss_curves(train_losses, val_losses, args.checkpoint_dir, args.data_name)
    return test_idx, df, img_path, train_tfm

def load_ckpt(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def load_saved_split(args, df):
    split_file = os.path.join(args.checkpoint_dir, f"split_indices_{args.data_name}.pth")
    if os.path.exists(split_file):
        split_info = torch.load(split_file)
        return split_info["train_idx"], split_info["val_idx"], split_info["test_idx"]
    print("[WARN] split_indices file not found; recomputing 60/20/20 split with same seed.")
    return split_indices(df, args.seed)

@torch.no_grad()
def extract_logits_labels(args):
    df, img_path = read_df_and_path(args, csv_path="Data_Entry_2017_clean.csv", img_path="images")
    _, _, test_idx = load_saved_split(args, df)
    val_tfm = create_transform(args.img_size, is_train=False)

    test_loader = make_loader(df, img_path, test_idx, val_tfm, batch=args.batch_size, workers=args.num_workers, shuffle=False)
    model = create_medical_model(model_name=args.pretrained_model, num_classes=len(PATHOLOGIES)).to(device)
    model = load_ckpt(model, os.path.join(args.checkpoint_dir, f"best_model_{args.data_name}.pth"))
    model.eval()

    logits_list, labels_list = [], []

    for images, labels in tqdm(test_loader, desc="[Test]"):
        images = images.to(device)
        logits = model(images)
        logits_list.append(logits.cpu())
        labels_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    test_metadata = df.iloc[test_idx].reset_index(drop=True)

    res = { "logits": logits, "labels": labels, "metadata": test_metadata }
    save_path = os.path.join(args.checkpoint_dir, f"test_data_{args.data_name}.pt")
    torch.save(res, save_path)

    print(f" Saved logits/labels/metadata {save_path} | logits {tuple(logits.shape)} | labels {tuple(labels.shape)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train NIH classifier (concise)")
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "extract"], help="train: fit model and save checkpoint; extract: skip training and only export test logits/labels/metadata")
    # Data
    parser.add_argument("--data_name", type=str, default="PadChest", choices=["NIH", "ChestX", "PadChest"], help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to NIH dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",help="Directory to save checkpoints")
    # Model
    parser.add_argument("--pretrained_model", type=str, default="densenet121", choices=["densenet121","resnet50"], help="Model architecture")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size (pixels)")

    # Training
    parser.add_argument("--weight_decay", type=float, default=5e-4,  help="L2 regularization weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs)")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4,help="Initial learning rate")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau", choices=["cosine", "step", "reduce_on_plateau", None])
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.mode == "train":
        train(args)
    elif args.mode == "extract":
        extract_logits_labels(args)
    else:
        raise ValueError("Unknown mode '{args.mode}'")

if __name__ == "__main__":
    main()
