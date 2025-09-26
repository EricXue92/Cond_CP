
DATASET_CONFIG = {
    "rxrx1": {
        "features_path": "data/rxrx1_v1.0/rxrx1_features.pt",
        "metadata_path": "data/rxrx1_v1.0/metadata.csv",
        "filter_key": "dataset",
        "filter_value": "test",
        "main_group": "cell_type", #cell_type
        "additional_features": ["experiment"],  # metadata columns experiment
        "grouping_columns": ["experiment", "cell_type"],
        "features_base_path": "features"  # optional for split datasets
    },
    'ChestX': {
        'features_base_path': 'features',
        'metadata_path': 'data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv',
        'classifier_path': 'checkpoints/best_model_ChestX.pth',
        'main_group': 'Patient Gender', # Patient Age  Finding Labels
        'additional_features': ['Patient Age'], # 'Patient Age'
        "grouping_columns": ["Patient Age", "Patient Gender"],
        'num_classes': 15,  # Number of diseases in ChestX-ray8
    },
    'PadChest': {
        'features_base_path': 'features',
        'metadata_path': 'data/PadChest/metadata.csv',
        'classifier_path': 'checkpoints/best_model_PadChest.pth',
        'main_group': 'gender',
        'additional_features': ['age'],
        "grouping_columns": ["***", "***"],
        'num_classes': 10,
    },
    'VinDr': {
        'features_base_path': 'features',
        'metadata_path': 'data/VinDr/metadata.csv',
        'classifier_path': 'checkpoints/best_model_VinDr.pth',
        'main_group': 'gender',
        'additional_features': [],
        "grouping_columns": ["***", "***"],
        'num_classes': 5,
    },
    'MIMIC': {
        'features_base_path': 'features',
        'metadata_path': 'data/MIMIC/metadata.csv',
        'classifier_path': 'checkpoints/best_model_MIMIC.pth',
        'main_group': 'gender',
        'additional_features': ['age_group'],
        'num_classes': 14,
    }

}