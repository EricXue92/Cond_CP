import os
gpu_choice = "1"   # or "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_choice

import torch
from torch.utils.data import DataLoader
from model_builder import create_rxrx1_model
from torchvision import transforms
from data_setup import ChestXray
from model_builder import git_vit_backbone

gpu_choice = "0"   # safer default (change to "1" if you KNOW you have >1 GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_choice
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(dataset, featurizer, classifier=None, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
    features_list, logits_list, labels_list = [], [], []

    featurizer.to(device).eval()
    if classifier is not None:
        classifier.to(device).eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
                metadata = None
            elif len(batch) == 3:
                images, labels, metadata = batch
            else:
                raise ValueError("Expected batch to have 2 or 3 elements (images, labels, [metadata])")
            images = images.to(device, non_blocking=True)
            features = featurizer(images)
            if classifier is not None:
                logits = classifier(features)
                logits_list.append(logits.detach().cpu())
            else:
                logits = None
            # Keep tensors, move to CPU for storage
            features_list.append(features.detach().cpu())
            labels_list.append(labels.detach().cpu())
            print(f"Processed batch with {images.size(0)} samples.")
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        logits = torch.cat(logits_list, dim=0) if classifier is not None else None
    return features,logits,labels

def save_features(features, logits, y,  filename, savedir="features", overwrite=False):
    os.makedirs(savedir, exist_ok=True)
    path = os.path.join(savedir, filename)

    if os.path.exists(path) and not overwrite:
        print(f"File already exists: {path}. Skipping save.")
        return

    payload = {"features": features.cpu(), "y": y.cpu()}
    if logits is not None:
        payload["logits"] = logits.cpu()

    print("Saving features to:", path)
    torch.save(payload, path)
    print("Features saved successfully.")

def load_features(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    #
    data = torch.load(path, map_location="cpu")
    features, y = data["features"], data["y"]
    logits = data.get("logits", None)
    print(f"Loaded features from {path} onto {device}")
    return features, logits, y


if __name__ == "__main__":
    # Load test data and metadata
    # test_images, test_metadata = load_rxrx1_test_data()
    # model, featurizer, classifier = create_rxrx1_model()
    # filepath = 'data/rxrx1_v1.0/rxrx1_features.pt'

    tfm = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to 224x224 for ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),  # ImageNet normalization
    ])
    metadata_csv = "data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv"
    train_ds = ChestXray(metadata_csv, split=0, transform=tfm)
    calib_ds = ChestXray(metadata_csv, split=1, transform=tfm)
    test_ds = ChestXray(metadata_csv, split=2, transform=tfm)
    featurizer = git_vit_backbone()  # Use the ViT backbone
    datasets_and_names = [
        (train_ds, "ChestX_train.pt"),
        (calib_ds, "ChestX_calib.pt"),
        (test_ds, "ChestX_test.pt")
    ]
    for dataset, name in datasets_and_names:
        path = os.path.join("features", name)
        if not os.path.exists(path):
            print(f"Extracting features for {path}...")
            features, logits, y = extract_features(dataset, featurizer, classifier=None, batch_size=1024)
            print(f"Saving features (torch) to {path}...")
            save_features(features, logits, y, filename=name)
        else:
            print(f"File does exist: {path}")

    features, logits, y = load_features("features/ChestXray_train.pt")
    print("Feature matrix:", features.shape, "Labels:", y.shape)





