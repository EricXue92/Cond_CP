import os
import torch
from torch.utils.data import DataLoader
from model_builder import create_rxrx1_model
from data_setup import load_rxrx1_test_data

def extract_rxrx_features(dataset, featurizer, classifier, batch_size=128, device=None):
    """Extract features and logits from dataset using provided models."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features_list, logits_list, labels_list = [], [], []

    featurizer.to(device).eval()
    classifier.to(device).eval()

    with torch.no_grad():
        for images, labels, metadata in dataloader:
            images = images.to(device)
            features = featurizer(images)
            logits = classifier(features)

            # Keep tensors, move to CPU for storage
            features_list.append(features.detach().cpu())
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
            print(f"Processed batch with {images.size(0)} samples.")

        features = torch.cat(features_list, dim=0)
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

    return features,logits,labels

def save_rxrx_features(features, logits, y,  path='data/rxrx1_v1.0/rxrx1_features.pt', overwrite=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and not overwrite:
        print(f"File already exists: {path}. Skipping save.")
        return

    print("Saving features to:", path)
    torch.save({"features": features, "y": y, "logits": logits}, path)
    print("Features saved successfully.")

def load_rxrx_features(filepath, device=None):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Features file not found: {filepath}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(filepath, map_location=device)
    features, logits, y = data["features"], data["logits"], data["y"]

    print(f"Loaded features from {filepath} onto {device}")
    return features, logits, y

if __name__ == "__main__":
    # Load test data and metadata
    test_images, test_metadata = load_rxrx1_test_data()
    model, featurizer, classifier = create_rxrx1_model()

    filepath = 'data/rxrx1_v1.0/rxrx1_features.pt'
    if not os.path.exists(filepath):
        print("Extracting features...")
        features, logits, y = extract_rxrx_features(test_images, featurizer, classifier, batch_size=1024)
        print("Saving features (torch) ...")
        save_rxrx_features(features, logits, y, filepath)
    else:
        print(f"File does exist: {filepath}")
