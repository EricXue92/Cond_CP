import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(dataset_name, checkpoint_path=None):
    if dataset_name == "rxrx1":
        num_classes = 1139
        default_ckpt = "checkpoints/rxrx1_seed_0_epoch_best_model.pth"
        model = models.resnet50(weights=None)
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, num_classes)
        classifier_attr = 'fc'

    elif dataset_name == "iwildcam":
        num_classes = 182
        default_ckpt = "checkpoints/iwildcam_best_model.pth"
        model = models.resnet50(weights=None)
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, num_classes)  # (2048, 182)
        classifier_attr = 'fc'

    elif dataset_name == "fmow":
        num_classes = 62
        default_ckpt = "checkpoints/best_model_fmow.pth"
        model = models.densenet121(weights=None)
        feature_dim = model.classifier.in_features
        model.classifier = nn.Linear(feature_dim, num_classes)
        classifier_attr = 'classifier'

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'rxrx1' or 'camelyon17'")

    checkpoint_path = checkpoint_path or default_ckpt
    if Path(checkpoint_path).exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("algorithm", checkpoint)
            state_dict = {k[6:] if k.startswith('model.') else k: v
                          for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
    else:
        print(f"[WARN] Checkpoint not found at {checkpoint_path}. Using randomly initialized weights.")

    model = model.to(device).eval()

    featurizer = deepcopy(model)

    if classifier_attr == "fc":
        featurizer.fc = nn.Identity()
    elif classifier_attr == "classifier":
        featurizer.classifier = nn.Identity()
    else:
        raise ValueError(f"Unexpected classifier attribute: {classifier_attr}")
    featurizer.d_out = feature_dim
    classifier = deepcopy(getattr(model, classifier_attr))

    print(f"[INFO] Created {dataset_name} model | feature_dim={feature_dim} | num_classes={num_classes}")
    return model, featurizer, classifier

# class DenseNetFeaturizer(torch.nn.Module):
#     def __init__(self, base):
#         super().__init__()
#         self.features = base.features
#         self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
#         self.d_out = 1024
#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)


