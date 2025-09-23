import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms

from data_setup import ChestXray
from model_builder import git_vit_featurizer,create_rxrx1_model
from data_utils import load_rxrx1_test_data
from feature_io import save_features

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(dataset, featurizer, classifier=None, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

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

            features_list.append(features.detach().cpu())
            labels_list.append(labels.detach().cpu())

            if classifier is not None:
                logits = classifier(features)
                logits_list.append(logits.detach().cpu())

            print(f"Processed batch with {images.size(0)} samples.")

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    logits = torch.cat(logits_list, dim=0) if classifier is not None else None

    return features,logits,labels

def get_dataset_config(dataset_name):

    if dataset_name == "rxrx1":
        test_images, _ = load_rxrx1_test_data()
        model, featurizer, classifier = create_rxrx1_model()
        return [(test_images, "rxrx1_features.pt")], featurizer, classifier

    elif dataset_name == "ChestX":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        metadata_csv = "data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv"
        datasets = [
            (ChestXray(metadata_csv, split=0, transform=transform), "ChestX_train.pt"),
            (ChestXray(metadata_csv, split=1, transform=transform), "ChestX_calib.pt"),
            (ChestXray(metadata_csv, split=2, transform=transform), "ChestX_test.pt")
        ]
        return datasets, git_vit_featurizer(), None

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def parse_arguments():
    parser = argparse.ArgumentParser(description=f'Extract features using a pretrained model')
    parser.add_argument("--dataset", default="ChestX",
                        choices=["ChestX", "PadChest", 'VinDr', "MIMIC", "rxrx1"])
    return parser.parse_args()

def main():
    args = parse_arguments()
    datasets_and_names, featurizer, classifier = get_dataset_config(args.dataset)
    for dataset, filename in datasets_and_names:
        filepath = os.path.join("features", filename)
        if os.path.exists(filepath):
            print(f"Features already exist: {filepath}")
            continue
        print(f"Extracting features for {filename}...")
        features, logits, labels = extract_features(dataset, featurizer, classifier, batch_size=1024)
        save_features(features, logits, labels, filename)

if __name__ == "__main__":
    main()



