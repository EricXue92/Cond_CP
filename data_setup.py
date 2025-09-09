import torch
import numpy as np
import random
import pandas as pd
from torchvision import transforms
from wilds import get_dataset

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

# test_images, metaData = load_rxrx1_test_data()
# print("Test dataset size:", len(test_images))
# print("Metadata dataset size:", len(metaData))