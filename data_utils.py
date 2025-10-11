import torch
from wilds import get_dataset
from torchvision import transforms
from collections import Counter
import pandas as pd


def load_dataset(dataset_name):
    if dataset_name == "rxrx1":
        return load_rxrx1_test_data()
    elif dataset_name == "iwildcam":
        return load_iwildcam_data()
    elif dataset_name == "fmow":
        return load_fmow_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_rxrx1_test_data(metadata_path='data/rxrx1_v1.0/metadata.csv'):
    transform = create_rxrx1_transform()
    dataset = get_dataset(dataset="rxrx1", download=False, root_dir="data")
    test_data = dataset.get_subset("test", transform=transform)
    metadata = pd.read_csv(metadata_path)
    test_metadata = metadata[metadata['dataset'] == 'test']

    test_metadata['experiment'] = test_metadata['experiment'].astype(str)
    test_metadata['cell_type'] = test_metadata['cell_type'].astype(str)

    print(f"[INFO] Loaded RxRx1 test set with {len(test_data)} samples.")
    print(f"[INFO] Experiments: {test_metadata['experiment'].nunique()} | Cell types: {test_metadata['cell_type'].nunique()}")

    return test_data, test_metadata

def load_iwildcam_data(root_dir="data", top_k_locations=8): # 48
    transform = create_iwildcam_transform()
    dataset = get_dataset(dataset="iwildcam", download=False, root_dir=root_dir)
    test_data = dataset.get_subset("test", transform=transform)

    metadata_array = dataset.metadata_array
    full_metadata = pd.DataFrame(
        metadata_array.numpy() if hasattr(metadata_array, 'numpy') else metadata_array,
        columns=dataset.metadata_fields
    )

    test_metadata = full_metadata.iloc[test_data.indices].reset_index(drop=True)

    # "year", "month", "day",
    for col in ("location", "hour"):
        if col in test_metadata.columns:
            test_metadata[col] = pd.to_numeric(test_metadata[col], errors="coerce").astype("Int64")

    # test_metadata.rename(columns={'location': 'camera', 'y': 'species'}, inplace=True)
    if "hour" in test_metadata.columns:
        test_metadata['time_of_day'] = test_metadata['hour'].apply(lambda h:
                                            'night' if 0 <= h < 6 else
                                             'morning' if 6 <= h < 12 else
                                             'afternoon' if 12 <= h < 18 else
                                             'evening')
    else:
        test_metadata['time_of_day'] = 'unknown'
    #test_metadata["location"] = test_metadata["location"].astype(str) # 48
    test_metadata["time_of_day"] = test_metadata["time_of_day"].astype(str)

    counts = Counter(test_metadata["location"])
    top_locations = {loc for loc, _ in counts.most_common(top_k_locations)}
    test_metadata["location_grouped"] = test_metadata["location"].apply(
        lambda loc: str(loc) if loc in top_locations else "Other"
    )
    test_metadata["location_grouped"] = test_metadata["location_grouped"].astype(str)

    print(f"[INFO] Loaded iWildCam test set: {len(test_data)} samples")
    print(f"[INFO] Total unique locations: {test_metadata['location'].nunique()}")
    print(f"[INFO] Using top-{top_k_locations} locations + 'Other'")

    # Location grouped distribution
    grouped_counts = test_metadata["location_grouped"].value_counts().sort_index()
    total = grouped_counts.sum()
    print(f"\n[INFO] Location Grouped Distribution:")
    for location, count in grouped_counts.items():
        print(f"  {location:<15}: {count:6d} samples ({100 * count / total:6.2f}%)")

    # Time of day distribution
    time_counts = test_metadata["time_of_day"].value_counts()
    print(f"\n[INFO] Time of Day Distribution:")
    for time_period, count in time_counts.items():
        print(f"  {time_period:<12}: {count:6d} samples ({100 * count / total:6.2f}%)")

    return test_data, test_metadata


    return test_data, test_metadata

def load_fmow_data(root_dir="data"):
    transform = create_iwildcam_transform()
    dataset = get_dataset(dataset="fmow", download=True, root_dir=root_dir)
    test_data = dataset.get_subset("test", transform=transform)
    metadata_array = dataset.metadata_array
    full_metadata = pd.DataFrame(
        metadata_array.numpy() if hasattr(metadata_array, 'numpy') else metadata_array,
        columns=dataset.metadata_fields
    )
    test_metadata = full_metadata.iloc[test_data.indices].reset_index(drop=True)

    for col in ("region", "year"):
        if col in test_metadata.columns:
            test_metadata[col] = pd.to_numeric(test_metadata[col], errors="coerce").astype("Int64")
    region_map = {
        0: 'Asia',
        1: 'Europe',
        2: 'Africa',
        3: 'Americas',
        4: 'Oceania',
        5: 'Other'
    }
    test_metadata["region_name"] = test_metadata["region"].map(region_map).fillna("Other")
    # merge the other category into Oceania to avoid too few samples (only 4)
    test_metadata["region_name"] = test_metadata["region_name"].replace("Other", "Oceania")

    test_metadata["year"] = test_metadata["year"].astype("string")
    test_metadata["region_name"] = test_metadata["region_name"].astype("string")
    counts = test_metadata["region_name"].value_counts()
    total = counts.sum()
    print("[INFO] Region Distribution (after merge):")
    for region, count in counts.items():
        print(f"  {region:<10}: {count:6d} samples ({100 * count / total:6.2f}%)")
    return test_data, test_metadata


def create_fmow_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_iwildcam_transform():
    return transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_rxrx1_transform():
    def standardize(x):
        m = x.mean(dim=(1, 2), keepdim=True)
        s = x.std(dim=(1, 2), keepdim=True)
        return (x - m) / torch.clamp(s, min=1e-8)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(standardize),
    ])




# load_fmow_data(root_dir="data")

# conditioning_map = {
#     "rxrx1": {
#         "primary": "experiment",  # Batch effect
#         "secondary": "cell_type",  # Biological variation
#         "tertiary": "site_id"  # Technical variation
#     },
#     "fmow": {
#         "primary": "region",  # Geographic shift
#         "secondary": "year"  # Temporal shift
#     },
#     "iwildcam": {
#         "primary": "location",  # Camera location
#         "secondary": "time_of_day"  # Temporal pattern (derived)
#     },
#     "camelyon17": {
#         "primary": "center"  # Hospital (only one strong group)
#     },
#     "poverty": {
#         "primary": "country",  # Geographic
#         "secondary": "urban"  # Urban/rural
#     },
#     "amazon": {
#         "primary": "category",  # Product category
#         "secondary": "year"  # Temporal
#     },
#     "civilcomments": {
#         "primary": ["male", "female"],  # Gender
#         "secondary": ["black", "white"],  # Race
#         "tertiary": ["christian", "muslim"]  # Religion
#     }
# }


