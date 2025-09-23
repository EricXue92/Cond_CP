import os
import torch
import numpy as np
import pandas as pd
from model_builder import load_classifier
from feature_io import load_features
from utils import categorical_to_numeric, one_hot_encode
from config import DATASET_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_logits_fn(features, model, batch_size=512):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size].to(device)
            logits_list.append(model(batch).cpu())
    return torch.cat(logits_list, dim=0)

def to_numpy(x):
    return x.numpy() if hasattr(x, 'numpy') else x

def load_split_data(dataset_name, compute_logits=True):
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    data = {}
    for split in ['train', 'calib', 'test']:
        path = os.path.join(config['features_base_path'], f"{dataset_name}_{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing feature file: {path}")

        features, logits, labels = load_features(path)

        if hasattr(labels, 'long'):
            labels = labels.long()
        elif hasattr(labels, 'astype'):
            labels = labels.astype(np.int64)

        data[split] = {'features': features, 'logits': logits,  'labels': labels }

    if compute_logits and data['calib']['logits'] is None:
        model = load_classifier(dataset_name)
        for split in data:
            data[split]['logits'] = compute_logits_fn(data[split]['features'], model)

    metadata_path = config.get('metadata_path')
    data['metadata'] = pd.read_csv(metadata_path) \
        if metadata_path and os.path.exists(metadata_path) else None
    return data

def get_metadata_splits(metadata, cal_size, test_size):
    if 'split' in metadata.columns:
        cal_meta = metadata[metadata['split'] == 1]
        test_meta = metadata[metadata['split'] == 2]
    else:
        cal_meta = metadata.iloc[:cal_size]
        test_meta = metadata.iloc[-test_size:]
    return cal_meta, test_meta

def process_feature(cal_meta, test_meta, feature_name, n_bins=5):
    cal_values = cal_meta[feature_name]
    test_values = test_meta[feature_name]

    if cal_values.dtype in ['object', 'category']:
        codes_cal = pd.factorize(cal_values)[0]
        codes_test = pd.factorize(test_values)[0]
        return one_hot_encode(codes_cal), one_hot_encode(codes_test)

    combined = pd.concat([cal_values, test_values])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)

    cal_bins = np.clip(np.digitize(cal_values, bins) - 1, 0, n_bins - 1)
    test_bins = np.clip(np.digitize(test_values, bins) - 1, 0, n_bins - 1)
    return one_hot_encode(cal_bins), one_hot_encode(test_bins)

def create_group_features(metadata, config, cal_size, test_size, n_bins=5):
    main_group = config['main_group']
    cal_meta, test_meta = get_metadata_splits(metadata, cal_size, test_size)
    return process_feature(cal_meta, test_meta, main_group, n_bins)

def add_additional_features(phi_cal, phi_test, metadata, config, cal_size, test_size, n_bins=5):
    """Add additional metadata features to existing phi matrices."""
    additional_features = config.get('additional_features', [])
    if not additional_features:
        return phi_cal, phi_test
    cal_meta, test_meta = get_metadata_splits(metadata, cal_size, test_size)

    for feature in additional_features:
        if feature not in cal_meta.columns:
            continue
        try:
            cal_feat, test_feat = process_feature(cal_meta, test_meta, feature, n_bins)
            phi_cal = np.hstack([phi_cal, cal_feat])
            phi_test = np.hstack([phi_test, test_feat])
        except Exception as e:
            print(f"Error processing feature '{feature}': {e}. Skipping.")

    return phi_cal, phi_test


def create_phi_split(dataset_name, data=None, use_groups=True, use_logits=False, add_features=False):
    """Create phi matrices for calibration and test splits."""
    if data is None:
        data = load_split_data(dataset_name)

    config = DATASET_CONFIG.get(dataset_name)
    cal_data, test_data, metadata = data['calib'], data['test'], data['metadata']

    if use_logits and cal_data['logits'] is not None:
        return to_numpy(cal_data['logits']), to_numpy(test_data['logits'])

    if use_groups and metadata is not None and config.get('main_group'):
        phi_cal, phi_test = create_group_features(
            metadata, config, len(cal_data['features']), len(test_data['features'])
        )

        if add_features:
            phi_cal, phi_test = add_additional_features(
                phi_cal, phi_test, metadata, config,
                len(cal_data['features']), len(test_data['features'])
            )

        return phi_cal, phi_test

    # Default: return raw features
    return to_numpy(cal_data['features']), to_numpy(test_data['features'])


def create_phi_chestx(use_groups=True, use_logits=False, add_features=True):
    return create_phi_split('ChestX', use_groups=use_groups, use_logits=use_logits, add_features=add_features)

def create_phi_padchest(use_groups=True, add_features=True):
    return create_phi_split('PadChest', use_groups=use_groups, add_features=add_features)