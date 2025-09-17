import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import ast
import torch


def get_nih_labels(csv_path='data/NIH/foundation_fair_meta/metadata_attr_lr.csv', label_col="labels"):
    df = pd.read_csv(csv_path)
    all_labels = set()
    for entry in df[label_col]:
        if isinstance(entry, str):
            labels = ast.literal_eval(entry)  # safely parse "['A', 'B']"
            all_labels.update(labels)
    return sorted(all_labels)

def encode_nih_labels(label_str, all_labels):
    if isinstance(label_str, str):
        labels = ast.literal_eval(label_str)  # safely parse "['A', 'B']"
    elif isinstance(label_str, list):
        labels = label_str
    else:
        raise TypeError("label_str must be str or list")
    vec = torch.zeros(len(all_labels), dtype=torch.float32)
    for disease in labels:
        if disease in all_labels:
            vec[all_labels.index(disease)] = 1.0
    return vec


class ChestXray(Dataset):
    def __init__(self, metadata_csv, split, transform=None,
                 path_col="filename", label_col="labels", all_labels=None):
        """
        Args:
            metadata_csv (str): Path to metadata.csv
            split (int): 0=train, 1=calibration, 2=test
            transform (callable, optional): Optional transforms for images
        """
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.all_labels = all_labels if all_labels is not None else get_nih_labels(metadata_csv, label_col)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, self.path_col]
        label_str = self.df.loc[idx, self.label_col]
        label = encode_nih_labels(label_str, self.all_labels)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

