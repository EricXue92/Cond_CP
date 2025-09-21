import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os
from datetime import datetime

def plot_size_hist_comparison(csv_file, figsize=(8, 6), save_path=None):
    df = pd.read_csv(csv_file)
    # Count frequencies for each size
    split_counts = df["Split_Size"].value_counts().sort_index()
    cond_counts = df["Cond_Size"].value_counts().sort_index()
    # Ensure both indices cover the same range
    all_sizes = sorted(set(split_counts.index).union(set(cond_counts.index)))
    split_counts = split_counts.reindex(all_sizes, fill_value=0)
    cond_counts = cond_counts.reindex(all_sizes, fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(all_sizes))
    width = 0.4

    ax.bar(x - width / 2, split_counts, width=width, label="Split Size", alpha=0.7)
    ax.bar(x + width / 2, cond_counts, width=width, label="Cond Size", alpha=0.7)

    ax.set_xlabel("Set Size")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Prediction Set Sizes")
    ax.set_xticks(x)
    ax.set_xticklabels(all_sizes)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved histogram to {save_path}")
    else:
        plt.show()

def plot_loss_curves(results, save_dir="Figures", filename="learning_curve.pdf"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(len(results["train_loss"]))
    metrics = {
        'Loss': (results["train_loss"], results["val_loss"]),
        'Accuracy': (results["train_acc"], results["val_acc"])
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for idx, (metric_name, (train_data, val_data)) in enumerate(metrics.items()):
        ax = axes[idx]
        # Plot train and validation curves
        ax.plot(epochs, train_data, label=f"train_{metric_name.lower()}")
        ax.plot(epochs, val_data, label=f"val_{metric_name.lower()}")

        ax.set_title(metric_name)
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Learning curves saved to: {save_path}")
    plt.show()

def plot_miscoverage(cells_file='results/cells.csv', experiments_file='results/experiments.csv',
                    target_miscoverage=0.1, save_dir="Figures",
                     save_name="Experiment_Cell_Miscoverage"):
    df_cells = pd.read_csv(cells_file)
    df_experiments = pd.read_csv(experiments_file)

    df_cells['Miscoverage'] = 1 - df_cells['Coverage']
    df_experiments['Miscoverage'] = 1 - df_experiments['Coverage']

    sns.set_style("white")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.7, 8.27), sharey=True)

    # Cell types plot
    sns.barplot(data=df_cells, x='Cell Type', y='Miscoverage', hue='Type', ax=ax1)
    ax1.axhline(target_miscoverage, color='red', linestyle='--', alpha=0.7)
    add_error_bars(ax1.collections[0] if ax1.collections else ax1, df_cells)
    ax1.legend().remove()
    ax1.set_title("Cell Types")

    # Experiments plot
    sns.barplot(data=df_experiments, x='Experiment', y='Miscoverage', hue='Type', ax=ax2)
    ax2.axhline(target_miscoverage, color='red', linestyle='--', alpha=0.7)
    add_error_bars(ax2.collections[0] if ax2.collections else ax2, df_experiments)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.legend(title='', loc='upper center')
    ax2.set_title("Experiments")

    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")

    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.show()
    return fig, (ax1, ax2)

def add_error_bars(barplot_obj, dataframe, err_col="error"):
    ax = barplot_obj.axes

    for i, patch in enumerate(barplot_obj.patches):
        if i >= len(dataframe):
            continue

        x_coord = patch.get_x() + 0.5 * patch.get_width()
        y_coord = patch.get_height()
        error_val = dataframe.iloc[i][err_col]

        ax.errorbar(x=x_coord, y=y_coord, yerr=error_val,
                    fmt="none", c="k", capsize=3, elinewidth=1)


