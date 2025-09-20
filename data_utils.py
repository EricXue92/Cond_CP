import torch
import pandas as pd
from wilds import get_dataset
from torch.utils.data import TensorDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from feature_io import load_features

def create_rxrx1_transform():
  """ Per-image, per-channel standardization for RxRx1. """
  standardize = lambda x: (x - x.mean(dim=(1, 2), keepdim=True)) / torch.clamp(x.std(dim=(1, 2),
                                                                                     keepdim=True), min=1e-8)
  return transforms.Compose([transforms.ToTensor(), transforms.Lambda(standardize)])

def load_rxrx1_test_data(seed=1):
    # Load dataset and test subset
    transform = create_rxrx1_transform()
    dataset = get_dataset(dataset="rxrx1", download=False)
    test_data = dataset.get_subset("test", transform=transform)
    metadata = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
    test_metadata = metadata[metadata['dataset'] == 'test']
    print("Got test dataset with size:", len(test_data))
    return test_data, test_metadata


def load_dataloaders(train_path, calib_path, test_path, batch_size=128):
    try:
        train_features, _, train_labels = load_features(train_path)
        calib_features, _, calib_labels = load_features(calib_path)
        test_features, _, test_labels   = load_features(test_path)
    except Exception as e:
        raise RuntimeError(f"Error loading features: {e}")

    train_ds = TensorDataset(train_features, train_labels)
    calib_ds = TensorDataset(calib_features, calib_labels)
    test_ds  = TensorDataset(test_features, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    calib_loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, calib_loader, test_loader
