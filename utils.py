import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math, random # for seed setting
import json
from pathlib import Path
from datetime import datetime
import seaborn as sns



def create_train_calib_test_split(n_samples, train_ratio=0.25, calib_ratio=0.25):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    # Calculate split points
    train_end = int(n_samples * train_ratio)
    calib_end = int(n_samples * (train_ratio + calib_ratio))
    return (
        indices[:train_end],  # train
        indices[train_end:calib_end],  # calibration
        indices[calib_end:]  # test
    )

# yTrain is the encoded experiment labels for the training set like [0,1,2,0,1,2,...]
def computeFeatures(x_train, x_cal, x_test, y_train, best_c):

    x_train = x_train.detach().cpu().numpy()
    x_cal = x_cal.detach().cpu().numpy()
    x_test = x_test.detach().cpu().numpy()
    y_train = np.asarray(y_train)

    model = LogisticRegression(C=best_c, max_iter=5000)  # 5000
    reg = model.fit(x_train, y_train)
    features_cal = reg.predict_proba(x_cal)  # Shape: (n_cal, 14): probabilities for each of 14 classes
    features_test = reg.predict_proba(x_test)  # shape: (n_test, 14): probabilities for each of 14 classes
    print("Calibration score shape:", features_cal.shape)
    print("Test score shape:", features_test.shape)
    return features_cal, features_test

def encode_labels(data, col="experiment"):
    if isinstance(data, pd.DataFrame):
        if col is None:
            raise ValueError("Column name must be provided when metadata is a DataFrame.")
        arr = data[col].copy().to_numpy()
    else:
        arr = np.asarray(data).copy()
    unique_vals = np.unique(arr)
    for i, val in enumerate(unique_vals):
        arr[arr == val] = i
    return arr.astype(int)

def build_cov_df(coverages_split, coverages_cond, subgrouping, group_name):
    """
    Build a tidy DataFrame of (overall + per-subgroup) coverages for two methods.
    """
    cov_df = pd.DataFrame({
        group_name: ['Marginal', 'Marginal'],
        'Type': ['Split Conformal', 'Conditional Calibration'],
        'Coverage': [np.mean(coverages_split), np.mean(coverages_cond)],
        'SampleSize': [len(coverages_split), len(coverages_cond)]
    })

    subgrouping = pd.Series(subgrouping).reset_index(drop=True)
    mapping = {g: i + 1 for i, g in enumerate(np.unique(subgrouping))}

    for g in np.unique(subgrouping):
        msk = (subgrouping == g).to_numpy()
        gid = mapping[g]

        new_df = pd.DataFrame({
            group_name: [gid, gid],
            'Type': ['Split Conformal', 'Conditional Calibration'],
            'Coverage': [np.mean(coverages_split[msk]), np.mean(coverages_cond[msk])],
            'SampleSize': [int(msk.sum()), int(msk.sum())]
        })
        cov_df = pd.concat([cov_df, new_df], ignore_index=True)
    cov_df['error'] = 1.96 * np.sqrt(cov_df['Coverage'] * (1 - cov_df['Coverage']) / cov_df['SampleSize'])
    return cov_df

def split_threshold(scores_cal, alpha):
    scores_cal = np.asarray(scores_cal, dtype=float).ravel()
    n = len(scores_cal)
    q_idx = math.ceil((n+1)*(1-alpha))/n
    return float(np.quantile(scores_cal, q_idx, method="higher"))

def encode_columns(df, cols):
    df_encoded, mappings = df.copy(), {}
    for col in cols:
        uniques = np.unique(df_encoded[col])
        mapping = {val: i for i, val in enumerate(uniques)}
        df_encoded[col] = df_encoded[col].map(mapping)
        mappings[col] = mapping
    return df_encoded, mappings


def plot_miscoverage(cells_file='results/cells.csv', experiments_file='results/experiments.csv',
                     figsize=(20.7, 8.27), font_scale=2, target_miscoverage=0.1,
                     x_label_fontsize=14, show_plot=True, save_dir="Figures", save_name="Experiment_Cell_Miscoverage"):
    covDfCells = pd.read_csv(cells_file)
    covDfExperiments = pd.read_csv(experiments_file)

    covDfCells['Miscoverage'] = 1 - covDfCells['Coverage']
    covDfExperiments['Miscoverage'] = 1 - covDfExperiments['Coverage']

    # Set plotting style
    sns.set(rc={'figure.figsize': figsize})
    sns.set(font_scale=font_scale)
    sns.set_style(style='white')

    # Create figure and subplots
    fig = plt.figure()

    # First subplot - Cell Types
    ax1 = fig.add_subplot(1, 2, 1)
    f = sns.barplot(data=covDfCells, x='Cell Type', y='Miscoverage', hue='Type', ax=ax1)
    ax1.axhline(target_miscoverage, color='red')

    # Add error bars for cells plot
    add_error_bars(f, covDfCells)
    ax1.legend([], [], frameon=False)

    # Second subplot - Experiments
    ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)
    f2 = sns.barplot(data=covDfExperiments, x='Experiment', y='Miscoverage', hue='Type', ax=ax2)
    ax2.axhline(target_miscoverage, color='red')
    add_error_bars(f2, covDfExperiments)

    ax2.tick_params(axis='x', labelsize=x_label_fontsize)  # Smaller font size for x-axis
    ax2.legend(title='', loc='upper center')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_name = f"{save_name}_{timestamp}.pdf"
    save_path = os.path.join(save_dir, timestamped_name)

    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"[INFO] Plot saved as PDF: {save_path}")
    if show_plot:
        plt.show()
    return fig, (ax1, ax2)


def add_error_bars(barplot_obj, dataframe, err_col="error"):
    ax = barplot_obj.axes
    for i, patch in enumerate(barplot_obj.patches):
        x_coord = patch.get_x() + 0.5 * patch.get_width()
        y_coord = patch.get_height()
        error_val = dataframe.iloc[i][err_col] if i < len(dataframe) else 0.0
        ax.errorbar(x=x_coord, y=y_coord, yerr=error_val,
                    fmt="none", c="k", capsize=3, elinewidth=1)



def save_or_append_csv(df, filename, save_dir="results"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{filename}_{timestamp}.csv"
    save_path = os.path.join(save_dir, name)
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved: {save_path}")
    return

def find_best_regularization(X, y, c_range=(1e-4, 1e+2), n_values=12, cv_folds=5,
    scoring="neg_log_loss",  solver="saga",  max_iter=2000, tol=1e-3, n_jobs=-1, class_weight=None, verbose=1):

    x_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y_np = np.asarray(y)
    cs = np.logspace(np.log10(c_range[0]), np.log10(c_range[1]), n_values)

    est = LogisticRegression( penalty="l2",  solver=solver, max_iter=max_iter,
        tol=tol, random_state=42,  class_weight=class_weight )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    gs = GridSearchCV(estimator=est, param_grid={"C": cs}, scoring=scoring, cv=cv,
        n_jobs=n_jobs, refit=True,  verbose=verbose, return_train_score=False,
        pre_dispatch="2*n_jobs")

    gs.fit(x_np, y_np)

    tested = gs.cv_results_["param_C"].data.astype(float)
    mean_loss = -gs.cv_results_["mean_test_score"]
    loss_map = dict(zip(tested, mean_loss))
    losses = np.array([loss_map[c] for c in cs])

    best_C = gs.best_params_["C"]
    best_loss = -gs.best_score_
    best_model = gs.best_estimator_
    return best_C

def one_hot_encode(labels):
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1 if labels.size else 0
    return np.eye(K, dtype=float)[labels] # shape (n, K)

def set_seed(seed, enforce_determinism=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if enforce_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    return seed

def save_prediction_sets(results, filepath=None, outdir="results"):

    # Ensure parent directory exists
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(outdir, f"prediction_sets_{timestamp}.csv")
    else:
        filepath = Path(filepath)

    n_test = len(results['split']['labels'])
    t_split = results['thresholds']['split']
    t_cond  = results['thresholds']['cond']

    rows = []
    for i in range(n_test):
        split_set = list(results['split']['sets'][i])
        cond_set  = list(results['cond']['sets'][i])

        row = {
            "Index": i,
            "Split_Label": int(results['split']['labels'][i]),
            "Split_Set": json.dumps([int(x) for x in split_set]),
            "Split_Size": len(split_set),
            "Split_Threshold": float(t_split) if np.isscalar(t_split) else str(t_split),

            "Cond_Label": int(results['cond']['labels'][i]),
            "Cond_Set": json.dumps([int(x) for x in cond_set]),
            "Cond_Size": len(cond_set),
        }

        if isinstance(t_cond, (list, np.ndarray)) and not np.isscalar(t_cond):
            row["Cond_Threshold"] = float(t_cond[i])
        else:
            row["Cond_Threshold"] = float(t_cond) if np.isscalar(t_cond) else str(t_cond)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Saved prediction sets to {filepath}")
    return df

def plot_size_hist_comparison(csv_file, figsize=(8, 6), save_path=None):
    df = pd.read_csv(csv_file)
    # Count frequencies for each size
    split_counts = df["Split_Size"].value_counts().sort_index()
    cond_counts = df["Cond_Size"].value_counts().sort_index()
    # Ensure both indices cover the same range
    all_sizes = sorted(set(split_counts.index).union(set(cond_counts.index)))
    split_counts = split_counts.reindex(all_sizes, fill_value=0)
    cond_counts = cond_counts.reindex(all_sizes, fill_value=0)
    # Plot side-by-side bars
    x = range(len(all_sizes))
    width = 0.4
    plt.figure(figsize=figsize)
    plt.bar([i - width/2 for i in x], split_counts, width=width, label="Split Size", alpha=0.7)
    plt.bar([i + width/2 for i in x], cond_counts, width=width, label="Cond Size", alpha=0.7)

    plt.xlabel("Set Size")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Set Sizes")
    plt.xticks(x, all_sizes)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved histogram to {save_path}")
    else:
        plt.show()



plot_size_hist_comparison(csv_file="results/pred_sets_20250910_155005.csv", save_path="Figures/pred_set_size.pdf")