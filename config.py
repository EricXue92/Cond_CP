
DATASET_CONFIG = {
    "rxrx1": {
        "features_path": "data/rxrx1_v1.0/rxrx1_features.pt",
        "metadata_path": "data/rxrx1_v1.0/metadata.csv",
        "filter_key": "dataset",
        "filter_value": "test",
        "main_group": "experiment",
        "additional_features": ["cell_type"],  # metadata columns
        "grouping_columns": ["experiment", "cell_type"],
        "features_base_path": "features"  # optional for split datasets
    },
    "ChestX": {
        "features_path": "features/ChestX_test.pt",  # still keep for compatibility
        "metadata_path": "data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv",
        "main_group": "age",
        "additional_features": ["age"],  # metadata columns
        "grouping_columns": ["sex", "age"],
        "features_base_path": "features"
    },
    # Add more datasets easily...
}