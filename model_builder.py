import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from copy import deepcopy
import torchxrayvision as xrv

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


def get_nih_model(weights="densenet121-res224-nih"):
    base_model = xrv.models.DenseNet(weights=weights)

    valid_indices = [i for i, name in enumerate(base_model.pathologies) if name.strip() != ""]
    valid_pathologies = [base_model.pathologies[i] for i in valid_indices]
    num_classes = len(valid_pathologies)

    featurizer = DenseNetFeaturizer(base_model)
    classifier = nn.Linear(featurizer.d_out, num_classes)

    if hasattr(base_model, 'classifier'):
        classifier.weight.data = base_model.classifier.weight.data.clone()
        classifier.bias.data = base_model.classifier.bias.data.clone()

    print(f"[INFO] Model: {weights}")
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Pathologies: {valid_pathologies}")

    return featurizer, classifier, valid_pathologies


# model = xrv.models.DenseNet(weights="densenet121-res224-all")
# model = xrv.models.DenseNet(weights="densenet121-res224-rsna") # RSNA Pneumonia Challenge
# model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
# model = xrv.models.DenseNet(weights="densenet121-res224-pc") # PadChest (University of Alicante)
# model = xrv.models.DenseNet(weights="densenet121-res224-chex") # CheXpert (Stanford)
# model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") # MIMIC-CXR (MIT)
# model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)
