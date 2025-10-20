import torch
from wilds import get_dataset
from torchvision import transforms
from collections import Counter
import pandas as pd
import torchxrayvision as xrv
import torchvision.transforms as transforms

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
    for col in ("location", "hour", "month"):
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

    if "month" in test_metadata.columns:
        test_metadata['season'] = test_metadata['month'].apply(lambda m:
                                                               'winter' if m in [12, 1, 2] else
                                                               'spring' if m in [3, 4, 5] else
                                                               'summer' if m in [6, 7, 8] else
                                                               'fall' if m in [9, 10, 11] else
                                                               'unknown')
    else:
        test_metadata['season'] = 'unknown'

    #test_metadata["location"] = test_metadata["location"].astype(str) # 48
    test_metadata["time_of_day"] = test_metadata["time_of_day"].astype(str)
    test_metadata["season"] = test_metadata["season"].astype(str)

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

    # Season distribution
    season_counts = test_metadata["season"].value_counts().sort_index()
    print(f"\n[INFO] Season Distribution:")
    for season, count in season_counts.items():
        print(f"  {season:<12}: {count:6d} samples ({100 * count / total:6.2f}%)")

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

def load_globalwheat_data(root_dir="data"):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = get_dataset(dataset="globalwheat", download=True, root_dir=root_dir)
    test_data = dataset.get_subset("test", transform=transform)

    metadata_array = dataset.metadata_array
    full_metadata = pd.DataFrame(
        metadata_array.numpy() if hasattr(metadata_array, 'numpy') else metadata_array,
        columns=dataset.metadata_fields
    )
    test_metadata = full_metadata.iloc[test_data.indices].reset_index(drop=True)

    test_metadata['domain'] = pd.to_numeric(test_metadata['domain'], errors='coerce').astype('Int64')

    # Two-level geographic grouping: (Location, Region)
    DOMAIN_INFO = {
        0: ('France', 'Western_Europe'),
        1: ('UK', 'Western_Europe'),
        2: ('Switzerland', 'Western_Europe'),
        3: ('Canada', 'North_America'),
        4: ('USA', 'North_America'),
        5: ('Australia', 'Oceania'),
        6: ('Japan', 'Asia'),
        7: ('Germany', 'Western_Europe'),
        8: ('Netherlands', 'Western_Europe'),
        9: ('Sweden', 'Northern_Europe')
    }

    # Create location and region columns
    test_metadata['country'] = test_metadata['domain'].map(
        {k: v[0] for k, v in DOMAIN_INFO.items()}
    ).fillna('Unknown').astype('string')

    test_metadata['region'] = test_metadata['domain'].map(
        {k: v[1] for k, v in DOMAIN_INFO.items()}
    ).fillna('Unknown').astype('string')

    print(f"[INFO] Loaded GlobalWheat test set: {len(test_data)} samples")
    print(f"[INFO] Total domains: {test_metadata['domain'].nunique()}")
    print(f"[INFO] Locations: {test_metadata['location'].nunique()}")
    print(f"[INFO] Regions: {test_metadata['region'].nunique()}")

    # Location distribution (fine-grained)
    print(f"\n[INFO] Location Distribution (10 countries):")
    loc_counts = test_metadata['location'].value_counts().sort_index()
    total = loc_counts.sum()
    for loc, count in loc_counts.items():
        print(f"  {loc:<15}: {count:5d} samples ({100 * count / total:5.2f}%)")

    # Region distribution (coarse-grained)
    print(f"\n[INFO] Region Distribution (5 regions):")
    reg_counts = test_metadata['region'].value_counts().sort_index()
    for reg, count in reg_counts.items():
        print(f"  {reg:<20}: {count:5d} samples ({100 * count / total:5.2f}%)")

    # Cross-tabulation
    print(f"\n[INFO] Location-Region Mapping:")
    cross_tab = test_metadata.groupby(['region', 'location']).size().unstack(fill_value=0)
    print(cross_tab)

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


def load_nih_data(root_dir="data"):
    imgpath = f"{root_dir}/NIH/images"  # image folder
    csvpath = f"{root_dir}/NIH/Data_Entry_2017_clean.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # grayscale normalization
    ])

    nih_dataset = xrv.datasets.NIH_Dataset(
        imgpath=imgpath,
        csvpath=csvpath,
        views=["PA"],
        transform=transform,
        unique_patients=True
    )

    print("=" * 70)
    print(f"[INFO] NIH Dataset Loaded: {len(nih_dataset)} samples")
    print(f"[INFO] Labels (pathologies): {nih_dataset.pathologies}")
    print(f"[INFO] CSV columns: {nih_dataset.csv.columns.tolist()}")
    print("=" * 70)

    # Print a few metadata entries
    print("[INFO] Sample metadata rows:")
    print(nih_dataset.csv.head(5))

    # Count number of positive labels per pathology
    label_counts = (nih_dataset.csv[nih_dataset.pathologies] > 0).sum().sort_values(ascending=False)
    print("\n[INFO] Label distribution (number of positives per class):")
    print(label_counts)

    # Print age and gender info if available
    if "Patient Age" in nih_dataset.csv.columns:
        ages = pd.to_numeric(nih_dataset.csv["Patient Age"], errors="coerce").dropna()
        print(f"\n[INFO] Patient age range: {ages.min()} - {ages.max()}")
        age_bins = pd.cut(ages, bins=[0,20,40,60,80,100], right=False)
        print("[INFO] Age group counts:")
        print(age_bins.value_counts().sort_index())

    if "Patient Gender" in nih_dataset.csv.columns:
        genders = nih_dataset.csv["Patient Gender"].value_counts()
        print("\n[INFO] Gender distribution:")
        print(genders)

    # Verify dataset __getitem__ output format
    print("\n[INFO] Inspecting one sample:")
    img, label = nih_dataset[0]
    print(f"  Image tensor shape: {img.shape}")
    print(f"  Label vector shape: {label.shape}")
    print(f"  Label example: {label}")

    return nih_dataset

if __name__ == "__main__":
    ds = load_nih_data()





