import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torchxrayvision as xrv
import torchvision
import pandas as pd

from feature_io import save_features
from model_builder import DenseNetFeaturizer, create_rxrx1_model
from data_utils import load_rxrx1_test_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "NIH":  "densenet121-res224-nih",
    "CheXpert": "densenet121-res224-chex",
    "PadChest": "densenet121-res224-pc",
    "MIMIC-NB": "densenet121-res224-mimic_nb",
    "MIMIC-CH": "densenet121-res224-mimic_ch",
    "RSNA": "densenet121-res224-rsna",
    "All":  "densenet121-res224-all"
}



TRANSFORM = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])

def load_dataset(dataset_name, data_root="data"):
    if dataset_name == "NIH":
        df = pd.read_csv(f"{data_root}/NIH/Data_Entry_2017.csv")
        df["Patient Age"] = df["Patient Age"].astype(str).str.extract(r"(\d+)").astype(int)
        df = df[(df["Patient Age"] >= 0) & (df["Patient Age"] <= 120)]
        df.to_csv(f"{data_root}/NIH/Data_Entry_2017_clean.csv", index=False)

    datasets = {
        "NIH": lambda: xrv.datasets.NIH_Dataset(
            imgpath=f"{data_root}/NIH/images",
            csvpath=f"{data_root}/NIH/Data_Entry_2017_clean.csv",
            transform=TRANSFORM
        ),
        "CheXpert": lambda: xrv.datasets.CheX_Dataset(
            imgpath=f"{data_root}/CheXpert/images",
            csvpath=f"{data_root}/CheXpert/chexpert_labels.csv",
            transform=TRANSFORM
        ),
        "PadChest": lambda: xrv.datasets.PC_Dataset(
            imgpath=f"{data_root}/PadChest/images",
            csvpath=f"{data_root}/PadChest/PadChest.csv",
            transform=TRANSFORM
        )
    }
    if dataset_name not in MODELS:
        raise ValueError(f"Unknown dataset {dataset_name}. Choose from {list(MODELS.keys())}")
    return datasets[dataset_name]()

def split_dataset(dataset, ratios=(0.5, 0.25, 0.25)):
    n = len(dataset)
    n_train = int(ratios[0] * n)
    n_calib = int(ratios[1] * n)
    n_test = n - n_train - n_calib
    return random_split(dataset, [n_train, n_calib, n_test])

def create_dataloaders(dataset, batch_size, ratios):
    train_set, calib_set, test_set = split_dataset(dataset, ratios)
    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4),
        "calib": DataLoader(calib_set, batch_size=batch_size, shuffle=False, num_workers=4),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    }

@torch.no_grad()
def extract_features(dataloader, model, featurizer, classifier, data_name ):
    model.eval()
    featurizer.eval()
    if classifier: classifier.eval()

    features_list, logits_list, labels_list, idx_list= [], [], [], []
    # sample = {
    #     "idx": idx,
    #     "lab": self.labels[idx],
    #     "img": normalized_image
    # }
    for batch in dataloader:
        if isinstance(batch, dict):
            images, labels, indices = batch["img"], batch["lab"], batch["idx"]
        else:
            images, labels, *rest = batch
            indices = torch.arange(len(labels))  # placeholder if no idx

        images = images.to(device)
        features = featurizer(images)
        features_list.append(features.cpu())
        labels_list.append(labels.cpu() if isinstance(labels, torch.Tensor) else labels)
        idx_list.append(indices.cpu() if isinstance(indices, torch.Tensor) else torch.tensor(indices))

        if data_name == "NIH":
            logits_list.append(model(images).cpu())
        elif data_name == "rxrx1" and classifier is not None:
            logits_list.append(classifier(features).cpu())
        else:
           raise ValueError(f"Unknown dataset {data_name}")

    return (
        torch.cat(features_list, dim=0),
        torch.cat(logits_list, dim=0),
        torch.cat(labels_list, dim=0),
        torch.cat(idx_list, dim=0)
    )

def main():
    parser = argparse.ArgumentParser(description="Extract features from chest X-ray or rxrx1 datasets")
    parser.add_argument("--dataset", default="NIH", choices=list(MODELS.keys()))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.5, 0.25, 0.25])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features")
    parser.add_argument("--output_dir", default="features", help="Directory to save features")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and featurizer

    if args.dataset == "rxrx1":
        dataset, _ = load_rxrx1_test_data()
        model, featurizer, classifier = create_rxrx1_model()
        out_path = os.path.join(args.output_dir, "rxrx1_features.pt")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[SKIP] rxrx1 - file exists")
        else:
            print(f"[EXTRACTING] rxrx1 dataset...")
            features, logits, labels, idxs = extract_features(
                DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4),
                model, featurizer, classifier, data_name="rxrx1"
            )
            save_features(features, logits, labels, out_path, idxs, overwrite=True)
            print(f"[SAVED] {out_path}")
    else:
        dataset = load_dataset(args.dataset)
        model = xrv.models.DenseNet(weights=MODELS[args.dataset]).to(device)
        featurizer = DenseNetFeaturizer(model).to(device)
        loaders = create_dataloaders(dataset, args.batch_size, args.split_ratio)

        # Extract features for each split
        for split_name, loader in loaders.items():
            file_name = f"{args.dataset}_{split_name}.pt"
            output_path = os.path.join(args.output_dir, file_name)
            if os.path.exists(output_path) and not args.overwrite:
                print(f"[SKIP] {split_name} - file exists")
                continue
            print(f"[EXTRACTING] {split_name} split...")
            features, logits, labels, idxs = extract_features(loader, model, featurizer)
            save_features(features, logits, labels, output_path, idxs, overwrite=args.overwrite)
            print(f"[SAVED] {output_path}")

if __name__ == "__main__":

    main()

