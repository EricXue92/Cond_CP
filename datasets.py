import os
import torch
import numpy as np
import random
import pandas as pd
from torchvision import transforms
from wilds import get_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torch.utils.data import DataLoader
from data_setup import ChestXrayDataset

def create_rxrx1_transform():
  """ Per-image, per-channel standardization for RxRx1. """
  standardize = lambda x: (x - x.mean(dim=(1, 2), keepdim=True)) / torch.clamp(x.std(dim=(1, 2),
                                                                                     keepdim=True), min=1e-8)
  return transforms.Compose([transforms.ToTensor(), transforms.Lambda(standardize)])

def load_rxrx1_test_data(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Load dataset and test subset
    transform = create_rxrx1_transform()
    dataset = get_dataset(dataset="rxrx1", download=False)
    test_data = dataset.get_subset("test", transform=transform)
    metadata = pd.read_csv('data/rxrx1_v1.0/metadata.csv')
    test_metadata = metadata[metadata['dataset'] == 'test']
    print("Got test dataset with size:", len(test_data))
    return test_data, test_metadata



def build_dataloaders(metadata_csv, batch_size=32, num_workers=4, image_size=224):
    """Build dataloaders for train/cal/test splits."""
    # Common transforms
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = ChestXrayDataset(metadata_csv, split=0, transform=train_tfms)
    calib_dataset = ChestXrayDataset(metadata_csv, split=1, transform=test_tfms)
    test_dataset = ChestXrayDataset(metadata_csv, split=2, transform=test_tfms)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, calib_loader, test_loader