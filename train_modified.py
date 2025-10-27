import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import set_seed, plot_loss_curves
from model_builder import create_medical_model
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score, average_precision_score

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
               'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
               'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

def calculate_pos_weights(dataset):
    labels = getattr(dataset, "labels", None)

    if labels is None and hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = dataset.dataset
        if hasattr(base, "labels"):
            labels = base.labels[dataset.indices]

    if labels is None:
        raise ValueError("Could not find labels on dataset or its base dataset.")

    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # labels shape: [N, C]
    pos = labels.sum(axis=0)                    # P per class
    N = labels.shape[0]
    neg = N - pos                               # N - P per class
    pos_weight = torch.tensor(neg / (pos + 1e-8), dtype=torch.float32)
    return pos_weight

def create_loaders(csv_path="data/NIH/Data_Entry_2017_clean.csv", img_path="data/NIH/images"):

    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = T.Compose(
        [ T.ToPILImage(), T.ToTensor(), T.Lambda(lambda x: x.repeat(3, 1, 1)),
          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Keep only 1 image per patient
    dataset = xrv.datasets.NIH_Dataset(imgpath=img_path, csvpath=csv_path,unique_patients=True)

    patient_ids = dataset.csv['Patient ID'].unique()
    train, temp = train_test_split(patient_ids, test_size=0.2, random_state=42, stratify=None)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train_idx = dataset.csv[dataset.csv['Patient ID'].isin(train)].index.tolist()
    val_idx = dataset.csv[dataset.csv['Patient ID'].isin(val)].index.tolist()
    test_idx = dataset.csv[dataset.csv['Patient ID'].isin(test)].index.tolist()

    train_dataset = xrv.datasets.SubsetDataset(dataset, train_idx)
    val_dataset = xrv.datasets.SubsetDataset(dataset, val_idx)
    test_dataset = xrv.datasets.SubsetDataset(dataset, test_idx)

    def train_collate(batch):
        images = [train_transform(item['img'].squeeze(0)) for item in batch]
        labels = [torch.from_numpy(item['lab']).float() for item in batch]
        return torch.stack(images), torch.stack(labels)

    def val_collate(batch):
        images = [val_transform(item['img'].squeeze(0)) for item in batch]
        labels = [torch.from_numpy(item['lab']).float() for item in batch]
        return torch.stack(images), torch.stack(labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=val_collate)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4,collate_fn=val_collate)

    return train_loader, val_loader, test_loader, test_dataset.csv, train_dataset

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
    train_loader, val_loader, _, _, train_dataset = create_loaders() # NIH
    pos_weights = calculate_pos_weights(train_loader.dataset).to(device)

    model = create_medical_model(model_name=args.pretrained_model,
                                 num_classes=len(PATHOLOGIES),
                                 dropout=0.2).to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
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
            optimizer, mode='min', factor=0.5, patience=4,
            min_lr=1e-6, verbose=True
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
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,'args': vars(args) }, ckpt_path)
            print(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement since {epochs_no_improve} epochs")
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered (patience={args.patience}).")
            break

    plot_loss_curves(train_losses, val_losses, args.checkpoint_dir, args.data_name)


def load_ckpt(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def extract_logits_labels(args):
    _,  _, test_loader, metadata = create_loaders()
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
    test_metadata = test_loader.csv

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
    parser.add_argument("--weight_decay", type=float, default=1e-3,  help="L2 regularization weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs)")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,help="Initial learning rate")
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

    # model = xrv.models.DenseNet(weights="densenet121-res224-nih")


if __name__ == "__main__":
    main()



