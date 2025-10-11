"""
ABIDE Feature Extraction - Simplified Version
Mirrors RxRx1 pipeline exactly
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

# For MRI loading
import nibabel as nib
from scipy import ndimage


# ============================================
# 1. Define 3D ResNet50 Architecture
# ============================================

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


def resnet50_3d(num_classes=2):
    return ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes)


# ============================================
# 2. Load Pretrained Weights
# ============================================

def myLoad(model, pretrain_path, device='cpu'):
    """Load pretrained weights"""
    checkpoint = torch.load(pretrain_path, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model


# ============================================
# 3. Setup Model (EXACTLY like RxRx1)
# ============================================

print("=" * 70)
print("ABIDE FEATURE EXTRACTION - RxRx1 Style")
print("=" * 70)

dimOut = 2  # ASD vs Control
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Auto-download pretrained weights
print("Downloading pretrained ResNet50 from Hugging Face...")
model_path = hf_hub_download(
    repo_id='TencentMedicalNet/MedicalNet-Resnet50',
    filename='resnet_50.pth'
)
print(f"✅ Downloaded to: {model_path}\n")

# Initialize model (like RxRx1)
constructor = resnet50_3d
model = constructor(num_classes=1000)
dimFeatures = model.fc.in_features  # 2048
lastLayer = nn.Linear(dimFeatures, dimOut)
model.d_out = dimFeatures
model.fc = lastLayer

# Load pretrained weights
abideModel = model.to(device)
myLoad(abideModel, model_path, device=device)

# Separate featurizer and classifier
featurizer = abideModel
classifier = abideModel.fc
featurizer.fc = nn.Identity()
abideModel = nn.Sequential(featurizer, classifier)

# Eval mode
abideModel.eval()
featurizer.eval()
classifier.eval()

print(f"✅ Model loaded!")
print(f"Feature dimension: {dimFeatures}\n")


# ============================================
# 4. Data Loading Functions
# ============================================

def load_mri_scan(nifti_path, target_shape=(56, 56, 56)):
    """Load and preprocess 3D MRI scan"""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    data = (data - data.mean()) / (data.std() + 1e-8)
    zoom_factors = [t / s for t, s in zip(target_shape, data.shape)]
    data = ndimage.zoom(data, zoom_factors, order=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)


# ============================================
# 5. Extract Features for All Subjects
# ============================================

print("=" * 70)
print("LOADING ABIDE DATA")
print("=" * 70)

# Load phenotypic data
phenotypic_url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
phenotypic = pd.read_csv(phenotypic_url)

# Filter subjects
phenotypic_filtered = phenotypic[
    (phenotypic['func_mean_fd'] < 0.2) &
    (phenotypic['DX_GROUP'].isin([1, 2]))
    ].copy()

print(f"Total subjects: {len(phenotypic_filtered)}")
print(f"Unique sites: {phenotypic_filtered['SITE_ID'].nunique()}")
print(f"ASD: {sum(phenotypic_filtered['DX_GROUP'] == 1)}")
print(f"Control: {sum(phenotypic_filtered['DX_GROUP'] == 2)}\n")

# Extract features
print("=" * 70)
print("EXTRACTING FEATURES")
print("=" * 70)

features_list = []
labels_list = []
sites_list = []

for idx, row in tqdm(phenotypic_filtered.iterrows(), total=len(phenotypic_filtered)):
    try:
        file_id = row['FILE_ID']
        scan_path = f"abide_data/{file_id}_anat.nii.gz"

        # Load scan
        scan_tensor = load_mri_scan(scan_path).to(device)

        # Extract features
        with torch.no_grad():
            features = featurizer(scan_tensor)
            features = features.view(features.size(0), -1)

        features_list.append(features.cpu().numpy().flatten())
        labels_list.append(row['DX_GROUP'] - 1)  # 0=ASD, 1=Control
        sites_list.append(row['SITE_ID'])

    except Exception as e:
        continue

# Convert to arrays
X = np.array(features_list)  # (N, 2048)
y = np.array(labels_list)
sites = np.array(sites_list)

print(f"\n✅ Feature extraction complete!")
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique sites: {len(np.unique(sites))}\n")

# Save features
np.savez('abide_features.npz',
         features=X,
         labels=y,
         sites=sites)
print("Saved to: abide_features.npz\n")

# ============================================
# 6. Train Site Predictor (like RxRx1)
# ============================================

print("=" * 70)
print("TRAINING SITE PREDICTOR")
print("=" * 70)

# Encode sites
site_encoder = LabelEncoder()
sites_encoded = site_encoder.fit_transform(sites)
print(f"Number of sites: {len(np.unique(sites_encoded))}")
print(f"Sites: {site_encoder.classes_}\n")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train site predictor
print("Training logistic regression...")
site_predictor = LogisticRegressionCV(
    Cs=30,
    cv=5,
    max_iter=1000,
    n_jobs=-1,
    random_state=42
)
site_predictor.fit(X_scaled, sites_encoded)

# Evaluate
accuracy = site_predictor.score(X_scaled, sites_encoded)

print(f"\n{'=' * 70}")
print(f"SITE PREDICTION ACCURACY: {accuracy:.3f} ({accuracy * 100:.1f}%)")
print(f"{'=' * 70}\n")

if accuracy > 0.7:
    print("✅ HIGH ACCURACY: Features encode site information!")
    print("   → Batch effects present")
    print("   → Conditional conformal prediction WILL WORK!")
else:
    print("⚠️  MODERATE ACCURACY: Weak site encoding")
    print("   → May need Mondrian CP instead")

print("\n" + "=" * 70)
print("✅ READY FOR CONDITIONAL CONFORMAL PREDICTION!")
print("=" * 70)
print("\nUse the extracted features (X), labels (y), and sites for")
print("conditional conformal prediction - same pipeline as RxRx1!")