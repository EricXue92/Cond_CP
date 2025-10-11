import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os

def plot_size_hist_comparison(csv_file, figsize=(7, 4), save_path=None):
    df = pd.read_csv(csv_file)
    split_counts = df["Split_Size"].value_counts().sort_index()
    cond_counts = df["Cond_Size"].value_counts().sort_index()
    all_sizes = sorted(set(split_counts.index).union(set(cond_counts.index)))

    split_counts = split_counts.reindex(all_sizes, fill_value=0)
    cond_counts = cond_counts.reindex(all_sizes, fill_value=0)

    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({'font.size': 12})  # readable font size
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(all_sizes))
    width = 0.4
    ax.bar(x - width / 2, split_counts, width=width, label="Split Size",)
    ax.bar(x + width / 2, cond_counts, width=width, label="Cond Size",)
    ax.set_xlabel("Set Size")
    ax.set_ylabel("Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels(all_sizes)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved histogram to {save_path}")
    else:
        plt.show()

# def plot_loss_curves(results, save_dir="Figures", filename="learning_curve.pdf"):
#     Path(save_dir).mkdir(parents=True, exist_ok=True)
#     epochs = range(len(results["train_loss"]))
#     metrics = {
#         'Loss': (results["train_loss"], results["val_loss"]),
#         'Accuracy': (results["train_acc"], results["val_acc"])
#     }
#     fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#     for idx, (metric_name, (train_data, val_data)) in enumerate(metrics.items()):
#         ax = axes[idx]
#         # Plot train and validation curves
#         ax.plot(epochs, train_data, label=f"train_{metric_name.lower()}")
#         ax.plot(epochs, val_data, label=f"val_{metric_name.lower()}")
#
#         ax.set_title(metric_name)
#         ax.set_xlabel("Epochs")
#         ax.legend()
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.tight_layout()
#     save_path = Path(save_dir) / filename
#     plt.savefig(save_path, bbox_inches='tight')
#     print(f"Learning curves saved to: {save_path}")
#     plt.show()

def plot_loss_curves(results, save_dir="Figures", filename="learning_curve.pdf"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(results["train_loss"])+1)

    metrics = {
        'Loss': (results["train_loss"], results["val_loss"]),
        'AUROC': (results["train_auroc"], results["val_auroc"]),
        'F1 Score': (results["train_f1"], results["val_f1"])
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (metric_name, (train_data, val_data)) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.plot(epochs, train_data, label=f"Train", marker='o', markersize=3)
        ax.plot(epochs, val_data, label=f"Val", marker='s', markersize=3)

        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Learning curves saved to: {save_path}")
    plt.close()  # Changed from plt.show() to plt.close() to avoid blocking

def plot_miscoverage(main_group, additional_group,
                     target_miscoverage, save_dir,
                     save_name):

    df_main_group = pd.read_csv(main_group).dropna()
    df_additional_group = pd.read_csv(additional_group).dropna()

    # Add miscoverage column
    df_main_group['Miscoverage'] = 1 - df_main_group['Coverage']
    df_additional_group['Miscoverage'] = 1 - df_additional_group['Coverage']

    # Columns we do NOT want as x-axis
    standard_cols = {'Type', 'Coverage', 'SampleSize', 'error', 'Miscoverage'}

    main_group_col = [c for c in df_main_group.columns if c not in standard_cols][0]
    additional_group_col = [c for c in df_additional_group.columns if c not in standard_cols][0]

    # ---- Fix ordering of categorical bins ----
    def order_bins(df, col):
        df[col] = df[col].astype(str)
        bin_labels = [x for x in df[col].unique() if x != "Marginal"]
        # Sort bin labels in natural order (remove brackets then sort by numbers)
        def parse_bin(label):
            # e.g. "[0,20)" -> (0,20)
            nums = [float(s) for s in label.replace("[", "").replace(")", "").split(",") if
                    s.strip().replace('.', '', 1).isdigit()]
            return nums[0] if nums else float("inf")
        bin_labels_sorted = sorted(bin_labels, key=parse_bin)
        ordered_labels = ["Marginal"] + bin_labels_sorted
        df[col] = pd.Categorical(df[col], categories=ordered_labels, ordered=True)
        return df

    df_main_group = order_bins(df_main_group, main_group_col)
    df_additional_group = order_bins(df_additional_group, additional_group_col)

    # ---- Plot ----
    sns.set_theme(style="white", context="notebook", font_scale=2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.7, 8.27), sharey=True)

    sns.barplot(data=df_additional_group, x=additional_group_col, y='Miscoverage', hue='Type', ax=ax1)
    ax1.axhline(target_miscoverage, color='red', linestyle='--', alpha=0.7)
    if 'error' in df_additional_group.columns:
        add_error_bars_to_plot(ax1, df_additional_group)
    ax1.legend().remove()
    ax1.set_xlabel(additional_group_col)

    sns.barplot(data=df_main_group, x=main_group_col, y='Miscoverage', hue='Type', ax=ax2)
    ax2.axhline(target_miscoverage, color='red', linestyle='--', alpha=0.7)

    if 'error' in df_main_group.columns:
        add_error_bars_to_plot(ax2, df_main_group)
    ax2.tick_params(axis='x', labelsize=14, rotation=45)  # rotate labels for readability
    ax2.legend(title='', loc='upper center')
    ax2.set_xlabel(main_group_col)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved: {save_path}")
    plt.show()
    return fig, (ax1, ax2)

def add_error_bars_to_plot(ax, df):
    patches = ax.patches
    if not patches or 'error' not in df.columns:
        return
    for i, patch in enumerate(patches):
        if i >= len(df):
            break
        x_coord = patch.get_x() + 0.5 * patch.get_width()
        y_coord = patch.get_height()
        error_val = df.iloc[i]['error']

        ax.errorbar(
            x=x_coord, y=y_coord, yerr=error_val,
            fmt="none", c="k", capsize=2, elinewidth=0.8
        )


def plot_training_history(history):
    """Plot training curves with loss and accuracy only."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"Figures/training_history.pdf", bbox_inches='tight')
    plt.show()


def main():
    # plot_miscoverage("results/ChestX_patient_age_logits.csv",
    #                 "results/ChestX_patient_gender_logits.csv",
    #                 target_miscoverage=0.1, save_dir="Figures",save_name="ChestX_miscoverage_logits"
    #                  )
    # plot_size_hist_comparison("results/pred_sets_groups_features.csv", figsize=(8, 6),
    #                           save_path="Figures/rxrx1_size_histogram_features.pdf")
    # plot_size_hist_comparison("results/pred_sets_groups_groups.csv", figsize=(8, 6),
    #                           save_path="Figures/rxrx1_size_histogram_groups.pdf")

    plot_miscoverage(main_group="results/iwildcam_location_grouped.csv", additional_group="results/iwildcam_time_of_day.csv",
                     target_miscoverage=0.1, save_dir="Figures",
                     save_name="iwildcam_miscoverage")





    plot_size_hist_comparison("results/rxrx1_pred_sets.csv", figsize=(8, 6),
                              save_path="Figures/rxrx1_size_histogram_groups.pdf")
    pass

if __name__ == "__main__":
    main()
