import pandas as pd
import numpy as np
from pathlib import Path
import os
import json

def save_csv(df, filename, save_dir="results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    name = f"{filename}.csv"
    save_path = os.path.join(save_dir, name)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved: {save_path}")

def save_prediction_sets(results, filepath=None, save_dir="results"):

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if filepath is None:
        filepath = os.path.join(save_dir, f"prediction_sets.csv")
    else:
        filepath = Path(filepath)

    n_test = len(results['split']['labels'])
    t_split = results['thresholds']['split']
    t_cond  = results['thresholds']['cond']

    rows = []
    for i in range(n_test):
        split_set = list(results['split']['sets'][i])
        cond_set  = list(results['cond']['sets'][i])

        if isinstance(t_cond, (list, np.ndarray)) and not np.isscalar(t_cond):
            cond_threshold = float(t_cond[i])
        else:
            cond_threshold = float(t_cond) if np.isscalar(t_cond) else str(t_cond)

        row = {
            "Index": i,
            "Split_Label": int(results['split']['labels'][i]),
            "Split_Set": json.dumps([int(x) for x in split_set]),
            "Split_Size": len(split_set),
            "Split_Threshold": float(t_split) if np.isscalar(t_split) else str(t_split),

            "Cond_Label": int(results['cond']['labels'][i]),
            "Cond_Set": json.dumps([int(x) for x in cond_set]),
            "Cond_Size": len(cond_set),
            "Cond_Threshold": cond_threshold,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Saved prediction sets to {filepath}")
    return df

def build_cov_df(coverages_split, coverages_cond, subgrouping, group_name):
    # Start with overall marginal rows
    cov_df = pd.DataFrame({
        group_name: ['Marginal', 'Marginal'],
        'Type': ['Split Conformal', 'Conditional Conformal'],
        'Coverage': [np.mean(coverages_split), np.mean(coverages_cond)],
        'SampleSize': [len(coverages_split), len(coverages_cond)]
    })

    subgrouping = pd.Series(subgrouping).reset_index(drop=True)


    if isinstance(subgrouping.dtype, pd.CategoricalDtype) or isinstance(subgrouping.iloc[0], pd.Interval):
        categories = subgrouping.cat.categories
        subgrouping = subgrouping.astype(str)
        ordered_groups = [str(c) for c in categories]  # keep order
    else:
        ordered_groups = pd.unique(subgrouping.astype(str))

    # --- Loop in preserved order ---
    for g in ordered_groups:
        mask = (subgrouping == str(g)).to_numpy()
        group_size = int(mask.sum())
        if group_size == 0:
            continue

        new_df = pd.DataFrame({
            group_name: [g, g],
            'Type': ['Split Conformal', 'Conditional Conformal'],
            'Coverage': [np.mean(coverages_split[mask]), np.mean(coverages_cond[mask])],
            'SampleSize': [group_size, group_size]
        })
        cov_df = pd.concat([cov_df, new_df], ignore_index=True)

    cov_df['error'] = 1.96 * np.sqrt(
        cov_df['Coverage'] * (1 - cov_df['Coverage']) / cov_df['SampleSize']
    )
    return cov_df