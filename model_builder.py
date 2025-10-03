import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_rxrx1_model(num_classes=1139,
                       checkpoint_path='checkpoints/rxrx1_seed_0_epoch_best_model.pth'):
    model = models.resnet50()
    feature_dim = model.fc.in_features # 2048 for ResNet-50
    model.fc = nn.Linear(feature_dim, num_classes) # (2048, 1139)
    model.d_out = feature_dim  # Store feature dimension

    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = {k[6:] if k.startswith('model.') else k: v
                         for k, v in checkpoint["algorithm"].items()}
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

    model = model.to(device).eval()

    featurizer = deepcopy(model)
    featurizer.fc = nn.Identity()
    featurizer.d_out = feature_dim
    featurizer.eval()
    classifier = deepcopy(model.fc)
    return model, featurizer, classifier
#
# # https://github.com/mlmed/torchxrayvision
# # https://mlmed.org/torchxrayvision/datasets.html#torchxrayvision.datasets.NIH_Dataset
#



class DenseNetFeaturizer(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.d_out = 1024
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)



