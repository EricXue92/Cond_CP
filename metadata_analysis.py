# import pandas as pd
# import numpy as np
#
# def analyze_metadata(csv_path, cols=None, bin_numeric=True, q=5):
#
#     metadata = pd.read_csv(csv_path)
#     print(f"[INFO] Loaded metadata with shape {metadata.shape}\n")
#
#     if cols is None:
#         cols = metadata.columns.tolist()
#
#     for col in cols:
#         if col not in metadata.columns:
#             print(f"[WARNING] Column '{col}' not found in metadata, skipping.\n")
#             continue
#
#         print(f"=== Column: {col} ===")
#         print(f"Type: {metadata[col].dtype}")
#         print(f"Missing values: {metadata[col].isnull().sum()}")
#         print(f"Unique values: {metadata[col].nunique()}")
#
#         if np.issubdtype(metadata[col].dtype, np.number):
#             # Numeric column
#             print(metadata[col].describe())
#             if bin_numeric:
#                 try:
#                     metadata[f"{col}_bin"] = pd.qcut(
#                         metadata[col], q=q, duplicates="drop", labels=False
#                     )
#                     print(f"[INFO] Added binned column '{col}_bin' with {metadata[f'{col}_bin'].nunique()} bins\n")
#                 except Exception as e:
#                     print(f"[ERROR] Could not bin column '{col}': {e}\n")
#         else:
#             # Categorical column
#             print(metadata[col].value_counts().head(10))
#             metadata[f"{col}_num"] = pd.factorize(metadata[col])[0]
#             print(f"[INFO] Added numeric-encoded column '{col}_num'\n")
#
#     return metadata
#
# csv_path = "data/ChestXray8/foundation_fair_meta/metadata_attr_lr.csv"
# meta = analyze_metadata(csv_path, cols=["Patient Age", "Patient Gender"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("data/NIH/Data_Entry_2017_clean.csv")

print("="*60)
print("PATIENT AGE ANALYSIS")
print("="*60)

# Check data type and unique values
print(f"\nColumn dtype: {df['Patient Age'].dtype}")
print(f"Total records: {len(df)}")
print(f"\nSample values:")
print(df['Patient Age'].head(20))

# Count value types
print(f"\nValue counts (top 20):")
print(df['Patient Age'].value_counts().head(20))

# Check for non-numeric values
print(f"\nNon-numeric values:")
non_numeric = df[pd.to_numeric(df['Patient Age'], errors='coerce').isna()]
print(non_numeric['Patient Age'].value_counts())

# Try to clean and extract numeric ages
df['Age_Clean'] = df['Patient Age'].astype(str).str.extract(r'(\d+)').astype(float)

print(f"\nAfter cleaning:")
print(f"Valid ages: {df['Age_Clean'].notna().sum()}")
print(f"Missing/invalid: {df['Age_Clean'].isna().sum()}")

# Statistics on valid ages
valid_ages = df['Age_Clean'].dropna()
print(f"\nAge statistics:")
print(f"Min: {valid_ages.min()}")
print(f"Max: {valid_ages.max()}")
print(f"Mean: {valid_ages.mean():.1f}")
print(f"Median: {valid_ages.median():.1f}")
print(f"Std: {valid_ages.std():.1f}")

# Age distribution
print(f"\nAge distribution by decade:")
for start in range(0, 100, 10):
    count = ((valid_ages >= start) & (valid_ages < start+10)).sum()
    print(f"{start:3d}-{start+9:3d}: {count:6d} ({count/len(valid_ages)*100:5.2f}%)")

# Check for outliers
outliers = valid_ages[(valid_ages < 0) | (valid_ages > 120)]
print(f"\nOutliers (age < 0 or > 120): {len(outliers)}")
if len(outliers) > 0:
    print(outliers.value_counts())

# Suggest bins
print(f"\n" + "="*60)
print("SUGGESTED BINS:")
print("="*60)
print("\nOption 1 (3 groups): [0, 40, 65, 120]")
print("Option 2 (4 groups): [0, 30, 50, 70, 120]")
print("Option 3 (5 groups): [0, 20, 40, 60, 80, 120]")

# Test binning with your current bins
for bins in [[0, 50, 100], [0, 40, 65, 120], [0, 30, 50, 70, 120], [0, 18, 40, 60, 80, 120]]:
    binned = pd.cut(valid_ages, bins=bins, right=False, include_lowest=True)
    print(f"\nBins {bins}:")
    print(binned.value_counts().sort_index())