import pandas as pd
import numpy as np

def analyze_metadata(csv_path, cols=None, bin_numeric=True, q=5):

    metadata = pd.read_csv(csv_path)
    print(f"[INFO] Loaded metadata with shape {metadata.shape}\n")

    if cols is None:
        cols = metadata.columns.tolist()

    for col in cols:
        if col not in metadata.columns:
            print(f"[WARNING] Column '{col}' not found in metadata, skipping.\n")
            continue

        print(f"=== Column: {col} ===")
        print(f"Type: {metadata[col].dtype}")
        print(f"Missing values: {metadata[col].isnull().sum()}")
        print(f"Unique values: {metadata[col].nunique()}")

        if np.issubdtype(metadata[col].dtype, np.number):
            # Numeric column
            print(metadata[col].describe())
            if bin_numeric:
                try:
                    metadata[f"{col}_bin"] = pd.qcut(
                        metadata[col], q=q, duplicates="drop", labels=False
                    )
                    print(f"[INFO] Added binned column '{col}_bin' with {metadata[f'{col}_bin'].nunique()} bins\n")
                except Exception as e:
                    print(f"[ERROR] Could not bin column '{col}': {e}\n")
        else:
            # Categorical column
            print(metadata[col].value_counts().head(10))
            metadata[f"{col}_num"] = pd.factorize(metadata[col])[0]
            print(f"[INFO] Added numeric-encoded column '{col}_num'\n")

    return metadata

csv_path = "data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv"
meta = analyze_metadata(csv_path, cols=["Patient Age", "Patient Gender"])