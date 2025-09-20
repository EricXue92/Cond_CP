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

    all_labels = sorted(all_labels)  # stable order
    label_to_index = {label: i for i, label in enumerate(all_labels)}
    return all_labels, label_to_index

def encode_nih_labels(label_str, label_to_index):
    if isinstance(label_str, str):
        labels = ast.literal_eval(label_str)  # safely parse "['A', 'B']"
    elif isinstance(label_str, list):
        labels = label_str
    else:
        raise TypeError("label_str must be str or list")

    vec = torch.zeros(len(label_to_index), dtype=torch.float32)
    for disease in labels:
        if disease in label_to_index:
            vec[label_to_index[disease]] = 1.0
    return vec

class ChestXray(Dataset):
    def __init__(self, metadata_csv, split, transform=None,
                 path_col="filename", label_col="labels",
                 all_labels=None, label_to_index=None):

        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform

        if all_labels is None or label_to_index is None:
            all_labels, label_to_index = get_nih_labels(metadata_csv, label_col)

        self.all_labels = all_labels
        self.label_to_index = label_to_index

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

