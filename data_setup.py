import os
import torch
import numpy as np
import random
import pandas as pd
from torchvision import transforms
from wilds import get_dataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ChestXrayDataset(Dataset):
    """Custom dataset for ChestXray8 with splits from metadata.csv"""
    def __init__(self, metadata_csv, split, transform=None):
        """
        Args:
            metadata_csv (str): Path to metadata.csv
            split (int): 0=train, 1=calibration, 2=test
            transform (callable, optional): Optional transforms for images
        """
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filepath"]
        label = self.df.loc[idx, "labels"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

