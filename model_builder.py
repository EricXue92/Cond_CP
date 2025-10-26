import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from copy import deepcopy
import torchxrayvision as xrv
from torchvision.models import densenet121, resnet50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For WILDS datasets: rxrx1, iwildcam, fmow
# Returns (model, featurizer, classifier)
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

    # elif dataset_name == "globalwheat":
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


class DenseNetFeaturizer(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.features = base.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.d_out = 1024
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


# def get_nih_model(weights="densenet121-res224-nih"):
#     base_model = xrv.models.DenseNet(weights=weights)
#     # Filter empty pathologies
#     valid_indices = [i for i, name in enumerate(base_model.targets)
#                      if name.strip() != ""]
#     print(base_model.targets)
#     valid_pathologies = [base_model.targets[i] for i in valid_indices]
#     num_classes = len(valid_pathologies)
#     featurizer = DenseNetFeaturizer(base_model) # [batch, 1, 224, 224] → [batch, 1024]
#     classifier = nn.Linear(featurizer.d_out, num_classes) # [batch, 1024] → [batch, 14]
#
#     # Copy weights - CRITICAL: must index if filtered
#     if hasattr(base_model, 'classifier'):
#         original_shape = base_model.classifier.weight.shape[0]
#         if original_shape == num_classes:
#             classifier.weight.data = base_model.classifier.weight.data.clone()
#             classifier.bias.data = base_model.classifier.bias.data.clone()
#             print(f"[INFO] Copied all {num_classes} pretrained weights")
#         else:
#             classifier.weight.data = base_model.classifier.weight.data[valid_indices].clone()
#             classifier.bias.data = base_model.classifier.bias.data[valid_indices].clone()
#             print(f"[INFO] Copied {num_classes}/{original_shape} pretrained weights (filtered)")
#     else:
#         print(f"[WARNING] No classifier found, using random initialization")
#
#     print(f"[INFO] Pathologies: {valid_pathologies}")
#     return featurizer, classifier, valid_pathologies


# def create_nih_model(num_classes=14):
#     model = densenet121(weights="DEFAULT")
#     model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     num_features = model.classifier.in_features
#     model.classifier = nn.Linear(num_features, num_classes)
#     return model

from torchvision.models import densenet121, resnet50

def create_medical_model(model_name='densenet121', num_classes=14):
    if model_name == 'densenet121':
        model = densenet121(weights='DEFAULT')
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'densenet121' or 'resnet50'")

    return model


# model = xrv.models.DenseNet(weights="densenet121-res224-all")
# model = xrv.models.DenseNet(weights="densenet121-res224-rsna") # RSNA Pneumonia Challenge
# model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
# model = xrv.models.DenseNet(weights="densenet121-res224-pc") # PadChest (University of Alicante)
# model = xrv.models.DenseNet(weights="densenet121-res224-chex") # CheXpert (Stanford)
# model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") # MIMIC-CXR (MIT)
# model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)
