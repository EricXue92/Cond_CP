import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from copy import deepcopy

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

# rx1Model, featurizer, classifier = create_rxrx1_model()