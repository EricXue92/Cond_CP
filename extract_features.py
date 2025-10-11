import os
import argparse
import torch
from torch.utils.data import DataLoader
from model_builder import create_model
from data_utils import load_rxrx1_test_data,  load_iwildcam_data, load_fmow_data
from feature_io import save_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def extract_features(dataset, featurizer, classifier,  batch_size=128):
    featurizer.eval()
    classifier.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_features, all_logits, all_labels = [], [], []

    for i, batch in enumerate(dataloader):
        if len(batch) == 3:
            images, labels, _ = batch
        elif len(batch) == 2:
            images, labels = batch
        else:
            raise ValueError("Expected batch to have 2 or 3 elements (images, labels, [metadata])")

        images = images.to(device)
        features = featurizer(images)
        logits = classifier(features)

        all_features.append(features.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i + 1:>4d}] Processed {len(images)} samples (total so far: {sum(len(f) for f in all_features)})")

    features = torch.cat(all_features)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    print(f"[DONE] Extracted {features.shape[0]} samples, feature dim={features.shape[1]}")
    return features, logits, labels

def load_dataset(dataset_name):
    if dataset_name == "rxrx1":
        return load_rxrx1_test_data()
    elif dataset_name == "iwildcam":
        return load_iwildcam_data()
    elif dataset_name == "fmow":
        return load_fmow_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Feature extraction for WILDS datasets")
    parser.add_argument("--dataset_name", default="fmow", choices=["rxrx1", "iwildcam", "fmow"],
                        help="Dataset to process", )
    # parser.add_argument("--dataset_name", type=str, required=True, choices=["rxrx1", "iwildcam", "fmow"], help="Dataset to process",)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--overwrite", action="store_true",help="Overwrite existing feature file if it exists",)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    batch_size = args.batch_size

    dataset, metadata = load_dataset(dataset_name)
    model, featurizer, classifier = create_model(dataset_name)

    save_dir = "features"
    os.makedirs(save_dir, exist_ok=True)
    feature_path = os.path.join(save_dir, f"{dataset_name}_features.pt")

    if not os.path.exists(feature_path) or args.overwrite:
        print(f"[INFO] Extracting features for {dataset_name} ...")
        features, logits, labels = extract_features(dataset, featurizer, classifier, batch_size)
        save_features(features, logits, labels, filename=f"{dataset_name}_features.pt", overwrite=args.overwrite)
        print(f"[SUCCESS] Saved features for {dataset_name}")
    else:
        print(f"[INFO] File already exists: {feature_path}. Use --overwrite to replace it.")

if __name__ == "__main__":
    main()




