

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os


def ensure_1d(arr):
    """Ensure array is 1D."""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        return arr.flatten()
    return arr


def plot_coverage_heatmap(coverage_split, coverage_cond, metadata,
                          age_col='age_group', gender_col='Patient Gender',
                          target=0.9, save_path='Figures/coverage_heatmap.png'):
    """Create a heatmap showing coverage across age and gender groups."""

    # Prepare data - ensure everything is 1D
    age_data = ensure_1d(metadata[age_col].values)
    gender_data = ensure_1d(metadata[gender_col].values)
    split_data = ensure_1d(coverage_split)
    cond_data = ensure_1d(coverage_cond)

    # Create DataFrame
    df = pd.DataFrame({
        'age': age_data,
        'gender': gender_data,
        'split': split_data,
        'cond': cond_data
    })

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Age categories
    age_order = ['0-18', '18-40', '40-60', '60-100']
    gender_order = sorted(df['gender'].unique())

    # Compute mean coverage for each group
    for idx, (method, method_name) in enumerate([('split', 'Split Conformal'),
                                                 ('cond', 'Conditional Conformal')]):
        ax = axes[idx]

        # Pivot table for heatmap
        pivot = df.groupby(['age', 'gender'])[method].mean().reset_index()
        pivot_table = pivot.pivot(index='gender', columns='age', values=method)
        pivot_table = pivot_table.reindex(index=gender_order, columns=age_order)

        # Create heatmap
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=target, vmin=0.85, vmax=0.95,
                    ax=ax, cbar_kws={'label': 'Coverage'},
                    linewidths=1, linecolor='gray')

        ax.set_title(f'{method_name}\n(Target: {target:.1%})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Age Group', fontsize=12)
        ax.set_ylabel('Gender', fontsize=12)

        # Add target line annotation
        for i, gender in enumerate(gender_order):
            for j, age in enumerate(age_order):
                if age in pivot_table.columns and gender in pivot_table.index:
                    val = pivot_table.loc[gender, age]
                    if not pd.isna(val) and val < target - 0.01:
                        # Add warning marker for below target
                        ax.add_patch(Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor='red', linewidth=3))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to {save_path}")
    plt.close()
    return fig


def plot_coverage_scatter(coverage_split, coverage_cond, metadata,
                          age_col='age_group', gender_col='Patient Gender',
                          target=0.9, save_path='Figures/coverage_scatter.png'):
    """Create scatter plot similar to the reference image."""

    # Map age groups to numeric values
    age_mapping = {'0-18': 0, '18-40': 1, '40-60': 2, '60-100': 3}
    gender_mapping = {'F': 0, 'M': 1}

    # Ensure 1D and create mappings
    age_data = ensure_1d(metadata[age_col].values)
    gender_data = ensure_1d(metadata[gender_col].values)
    split_data = ensure_1d(coverage_split)
    cond_data = ensure_1d(coverage_cond)

    # Map to numeric
    age_num = np.array([age_mapping.get(a, 0) for a in age_data])
    gender_num = np.array([gender_mapping.get(g, 0) for g in gender_data])

    # Create DataFrame
    df = pd.DataFrame({
        'age_num': age_num,
        'gender_num': gender_num,
        'age': age_data,
        'gender': gender_data,
        'split': split_data,
        'cond': cond_data
    })

    # Add jitter for better visualization
    np.random.seed(42)
    jitter_age = np.random.uniform(-0.15, 0.15, len(df))
    jitter_gender = np.random.uniform(-0.08, 0.08, len(df))
    df['age_jitter'] = df['age_num'] + jitter_age
    df['gender_jitter'] = df['gender_num'] + jitter_gender

    # Compute group coverage for coloring
    df['group_cov'] = df.groupby(['age', 'gender'])['cond'].transform('mean')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot
    scatter = ax.scatter(df['age_jitter'], df['gender_jitter'],
                         c=df['group_cov'], cmap='RdYlGn',
                         vmin=0.85, vmax=0.95,
                         s=50, alpha=0.6, edgecolors='none')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Group Coverage (Conditional)', fontsize=12, fontweight='bold')

    # Customize axes
    ax.set_xlabel('Age Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Gender', fontsize=14, fontweight='bold')
    ax.set_title('Estimated Coverage by Age and Gender\n(Conditional Conformal Prediction)',
                 fontsize=16, fontweight='bold', pad=20)

    # Set ticks
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['0-18', '18-40', '40-60', '60-100'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Female', 'Male'])

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plot to {save_path}")
    plt.close()
    return fig


def plot_coverage_by_group(coverage_split, coverage_cond, metadata,
                           age_col='age_group', gender_col='Patient Gender',
                           target=0.9, save_path='Figures/coverage_by_group.pdf'):

    # Ensure proper shapes
    age_data = ensure_1d(metadata[age_col].values)
    gender_data = ensure_1d(metadata[gender_col].values)
    split_data = ensure_1d(coverage_split)
    cond_data = ensure_1d(coverage_cond)

    # Combine into DataFrame
    df = pd.DataFrame({
        'age': age_data,
        'gender': gender_data,
        'split': split_data,
        'cond': cond_data
    })

    # Compute mean coverage per (age, gender)
    group_stats = df.groupby(['age', 'gender']).agg({
        'split': 'mean',
        'cond': 'mean'
    }).reset_index()

    # Combine into a single group label for plotting
    group_stats['group'] = group_stats['age'].astype(str) + '\n' + group_stats['gender'].astype(str)
    group_stats = group_stats.sort_values(['age', 'gender'])

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(group_stats))
    width = 0.35

    # Plot bars
    bars_split = ax.bar(x - width / 2, group_stats['split'], width,
                        label='Split Conformal',  edgecolor='black', alpha=0.8)
    bars_cond = ax.bar(x + width / 2, group_stats['cond'], width,
                       label='Conditional Conformal',  edgecolor='black', alpha=0.8)

    # Target coverage line
    ax.axhline(y=target, color='red', linestyle='--', linewidth=2,
               label=f'Target ({target:.1%})', alpha=0.8)

    # Axes labels & formatting
    ax.set_xlabel('Demographic Group', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coverage', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_stats['group'], fontsize=11)
    ax.set_ylim(0.84, 0.96)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)

    # Add value labels on Conditional bars (right bar)
    for bar in bars_cond:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')

    print(f"✓ Saved coverage plot to {save_path}")
    plt.close(fig)
    return fig




def plot_coverage_grid(coverage_split, coverage_cond, metadata,
                       age_col='age_group', gender_col='Patient Gender',
                       target=0.9, save_path='Figures/coverage_grid.png'):
    """Create a grid plot showing individual and aggregate views."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Map to numeric for scatter
    age_mapping = {'0-18': 9, '18-40': 29, '40-60': 50, '60-100': 75}
    gender_mapping = {'F': 0, 'M': 1}

    # Ensure 1D
    age_data = ensure_1d(metadata[age_col].values)
    gender_data = ensure_1d(metadata[gender_col].values)
    split_data = ensure_1d(coverage_split)
    cond_data = ensure_1d(coverage_cond)

    # Map to numeric
    age_mid = np.array([age_mapping.get(a, 50) for a in age_data])
    gender_num = np.array([gender_mapping.get(g, 0) for g in gender_data])

    # Create DataFrame
    df = pd.DataFrame({
        'age_mid': age_mid,
        'gender_num': gender_num,
        'age': age_data,
        'gender': gender_data,
        'split': split_data,
        'cond': cond_data
    })

    # Add group coverage
    df['group_cov'] = df.groupby(['age', 'gender'])['cond'].transform('mean')

    # Plot 1: Scatter
    ax1 = fig.add_subplot(gs[0, :])

    # Add jitter
    np.random.seed(42)
    jitter_age = np.random.uniform(-2, 2, len(df))
    jitter_gender = np.random.uniform(-0.05, 0.05, len(df))

    scatter = ax1.scatter(df['age_mid'] + jitter_age,
                          df['gender_num'] + jitter_gender,
                          c=df['group_cov'], cmap='RdYlGn',
                          vmin=0.85, vmax=0.95, s=30, alpha=0.5, edgecolors='none')

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Estimated Coverage', fontsize=11, fontweight='bold')

    ax1.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gender', fontsize=12, fontweight='bold')
    ax1.set_title('Estimated Coverage by Age and Gender\n(Conditional Conformal Prediction)',
                  fontsize=14, fontweight='bold')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Female', 'Male'])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coverage by age
    ax2 = fig.add_subplot(gs[1, 0])
    age_stats = df.groupby('age')['cond'].mean().reindex(['0-18', '18-40', '40-60', '60-100'])
    colors_age = plt.cm.RdYlGn(np.interp(age_stats.values, [0.85, 0.95], [0, 1]))
    ax2.bar(range(len(age_stats)), age_stats.values, color=colors_age, edgecolor='black')
    ax2.axhline(y=target, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Age Group', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Coverage', fontsize=11, fontweight='bold')
    ax2.set_title('Coverage by Age', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(age_stats)))
    ax2.set_xticklabels(age_stats.index)
    ax2.set_ylim(0.85, 0.95)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Coverage by gender
    ax3 = fig.add_subplot(gs[1, 1])
    gender_stats = df.groupby('gender')['cond'].mean().sort_index()
    colors_gender = plt.cm.RdYlGn(np.interp(gender_stats.values, [0.85, 0.95], [0, 1]))
    ax3.bar(range(len(gender_stats)), gender_stats.values, color=colors_gender, edgecolor='black')
    ax3.axhline(y=target, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Target ({target:.0%})')
    ax3.set_xlabel('Gender', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Coverage', fontsize=11, fontweight='bold')
    ax3.set_title('Coverage by Gender', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(gender_stats)))
    ax3.set_xticklabels(['Female', 'Male'])
    ax3.set_ylim(0.85, 0.95)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved grid plot to {save_path}")
    plt.close()
    return fig


def create_all_visualizations(coverage_split, coverage_cond, metadata,
                              age_col='age_group', gender_col='Patient Gender',
                              target=0.9, output_dir='Figures'):
    """Create all visualization types."""

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    try:
        # 1. Heatmap
        print("\n[1/4] Creating heatmap...")
        plot_coverage_heatmap(coverage_split, coverage_cond, metadata,
                              age_col, gender_col, target,
                              f'{output_dir}/coverage_heatmap.png')

        # 2. Scatter plot
        print("[2/4] Creating scatter plot...")
        plot_coverage_scatter(coverage_split, coverage_cond, metadata,
                              age_col, gender_col, target,
                              f'{output_dir}/coverage_scatter.png')

        # 3. Bar plot
        print("[3/4] Creating bar plot...")
        plot_coverage_by_group(coverage_split, coverage_cond, metadata,
                               age_col, gender_col, target,
                               f'{output_dir}/coverage_by_group.pdf')

        # 4. Grid plot
        print("[4/4] Creating grid plot (reference style)...")
        plot_coverage_grid(coverage_split, coverage_cond, metadata,
                           age_col, gender_col, target,
                           f'{output_dir}/coverage_grid.png')

        print("\n" + "=" * 80)
        print("✓ All visualizations created successfully!")
        print(f"  Location: {output_dir}/")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        print("\nContinuing without visualizations...")


if __name__ == "__main__":
    print("This module should be imported and used with your conformal prediction results.")
    print("Example usage:")
    print("  from visualize_coverage_fixed import create_all_visualizations")
    print("  create_all_visualizations(coverage_split, coverage_cond, metadata)")