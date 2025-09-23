import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from torchvision.models import vit_b_32, ViT_B_32_Weights
from copy import deepcopy
from config import DATASET_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_rxrx1_model(num_classes=1139, checkpoint_path='checkpoints/rxrx1_seed_0_epoch_best_model.pth'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50()
    feature_dim = model.fc.in_features # 2048 for ResNet-50
    model.fc = nn.Linear(feature_dim, num_classes) # (2048, 1139)
    model.d_out = feature_dim  # Store feature dimension

    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Remove 'model.' prefix from keys if present
            state_dict = {k[6:] if k.startswith('model.') else k: v
                         for k, v in checkpoint["algorithm"].items()}
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
    # Move to device and set eval mode
    model = model.to(device).eval()

    # Create featurizer (model without classifier)
    featurizer = deepcopy(model)
    featurizer.fc = nn.Identity()
    featurizer.d_out = feature_dim
    featurizer.eval()

    # Extract classifier
    classifier = model.fc.eval()
    return model, featurizer, classifier

def git_vit_featurizer():
    weights = ViT_B_32_Weights.DEFAULT
    model = vit_b_32(weights=weights).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Remove classification head -> outputs 768-dim embeddings
    model.heads = torch.nn.Identity() # shape: (batch_size, 768)
    return model

class ChestXClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=15, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)

def load_classifier(dataset_name):
    """Load trained classifier for generating logits from features."""
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    checkpoint = torch.load(config['classifier_path'], map_location=device)
    model = ChestXClassifier(num_classes=config.get('num_classes', 15))

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print(f"[INFO] Loaded classifier for '{dataset_name}' with {config.get('num_classes', 15)} classes")
    return model






