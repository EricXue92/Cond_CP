# verify_rxrx1_correlation.py
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def load_rxrx1_data(features_path='features/rxrx1_test.pt',
                    metadata_path='data/rxrx1_v1.0/metadata.csv'):
    print(f"[LOADING] RxRx1 data...")

    # Load features
    data = torch.load(features_path)
    features = data['features']
    logits = data['logits']
    labels = data['labels']

    print(f"  Features shape: {features.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Filter to test set
    metadata = metadata[metadata['dataset'] == 'test'].reset_index(drop=True)

    # Verify alignment
    assert len(features) == len(metadata), \
        f"Mismatch: {len(features)} features vs {len(metadata)} metadata rows"

    print(f"  Metadata samples: {len(metadata)}")
    print(f"  Metadata columns: {metadata.columns.tolist()}")
    print(f"\n[INFO] Key features:")
    print(f"  Experiments: {metadata['experiment'].nunique()} unique")
    print(f"  Cell types: {metadata['cell_type'].nunique()} unique")
    print(f"  Sites: {metadata['site'].nunique()} unique")

    return features, metadata


def verify_visual_correlation(features, metadata, feature_name,
                              sample_size=5000, save_path=None):
    """
    Verify if metadata feature correlates with image features.

    Args:
        features: torch.Tensor [N, D] - CNN features
        metadata: pd.DataFrame - metadata with feature_name column
        feature_name: str - column to analyze (e.g., 'experiment', 'cell_type')
        sample_size: int - subsample for faster computation
        save_path: str - path to save figure
    """

    # Convert to numpy
    if hasattr(features, 'numpy'):
        features = features.numpy()

    # Subsample if needed
    if len(features) > sample_size:
        indices = np.random.choice(len(features), sample_size, replace=False)
        features_sub = features[indices]
        metadata_sub = metadata.iloc[indices].reset_index(drop=True)
    else:
        features_sub = features
        metadata_sub = metadata

    print(f"\n{'=' * 70}")
    print(f"VISUAL CORRELATION ANALYSIS: {feature_name.upper()}")
    print(f"{'=' * 70}")
    print(f"Samples analyzed: {len(features_sub)}")
    print(f"Feature dimension: {features_sub.shape[1]}")
    print(f"Unique {feature_name} values: {metadata_sub[feature_name].nunique()}")

    # ================================================================
    # STEP 1: Silhouette Score (Clustering Quality)
    # ================================================================
    print(f"\n[STEP 1] Computing Silhouette Score...")

    le = LabelEncoder()
    labels = le.fit_transform(metadata_sub[feature_name])

    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(features_sub, labels)
        print(f"  Silhouette Score: {silhouette:.4f}")

        if silhouette > 0.5:
            print("  ✓✓✓ EXCELLENT - Very strong visual correlation!")
            print("      → Highly recommended for conditional CP")
        elif silhouette > 0.3:
            print("  ✓✓ STRONG - Good visual correlation")
            print("      → Recommended for conditional CP")
        elif silhouette > 0.15:
            print("  ✓ MODERATE - Some visual correlation")
            print("      → Can be used for conditional CP")
        elif silhouette > 0.05:
            print("  ⚠ WEAK - Limited visual correlation")
            print("      → May provide limited benefit")
        else:
            print("  ✗ NONE - No visual correlation")
            print("      → Not recommended for conditioning")

    # ================================================================
    # STEP 2: Variance Ratio Analysis
    # ================================================================
    print(f"\n[STEP 2] Computing Variance Ratio...")

    unique_values = metadata_sub[feature_name].unique()
    sample_values = unique_values[:20] if len(unique_values) > 20 else unique_values

    group_means = []
    within_group_vars = []

    for val in sample_values:
        mask = metadata_sub[feature_name] == val
        if mask.sum() > 1:
            group_features = features_sub[mask]
            group_means.append(group_features.mean(axis=0))
            within_group_vars.append(group_features.var(axis=0).mean())

    if len(group_means) > 1:
        group_means = np.array(group_means)
        between_group_var = np.var(group_means, axis=0).mean()
        within_group_var = np.mean(within_group_vars)
        variance_ratio = between_group_var / (within_group_var + 1e-10)

        print(f"  Between-group variance: {between_group_var:.6f}")
        print(f"  Within-group variance:  {within_group_var:.6f}")
        print(f"  Variance ratio: {variance_ratio:.4f}")

        if variance_ratio > 0.2:
            print("  ✓✓ HIGH - Groups have very distinct visual features")
        elif variance_ratio > 0.1:
            print("  ✓ GOOD - Groups have distinct visual features")
        elif variance_ratio > 0.05:
            print("  ⚠ MODERATE - Groups have some visual differences")
        else:
            print("  ✗ LOW - Groups are visually very similar")

    # ================================================================
    # STEP 3: PCA Visualization
    # ================================================================
    print(f"\n[STEP 3] Computing PCA projection...")

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_sub)

    explained_var = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_var[0]:.2%} of variance")
    print(f"  PC2 explains {explained_var[1]:.2%} of variance")
    print(f"  Total explained: {explained_var.sum():.2%}")

    # ================================================================
    # STEP 4: t-SNE (for better cluster visualization)
    # ================================================================
    compute_tsne = len(features_sub) <= 10000

    if compute_tsne:
        print(f"\n[STEP 4] Computing t-SNE projection...")
        print("  (This may take 1-2 minutes...)")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_tsne = tsne.fit_transform(features_sub)
        print("  ✓ t-SNE complete")

    # ================================================================
    # STEP 5: Create Visualizations
    # ================================================================
    print(f"\n[STEP 5] Generating visualizations...")

    n_plots = 3 if compute_tsne else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))

    if n_plots == 2:
        axes = [axes[0], axes[1]]

    # Determine color scheme
    unique_vals = metadata_sub[feature_name].unique()
    n_unique = len(unique_vals)

    print(f"  Creating plots for {n_unique} unique {feature_name} values...")

    if n_unique <= 20:
        # Discrete colors for categorical with few values
        palette = sns.color_palette("tab20", n_unique)
        color_map = {val: palette[i] for i, val in enumerate(unique_vals)}

        # Plot 1: PCA with discrete colors
        for i, val in enumerate(unique_vals):
            mask = metadata_sub[feature_name] == val
            axes[0].scatter(features_pca[mask, 0], features_pca[mask, 1],
                            c=[palette[i]], label=f'{val}', alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       title=feature_name.replace('_', ' ').title())
    else:
        # Continuous colormap for many values
        colors = le.transform(metadata_sub[feature_name])
        scatter = axes[0].scatter(features_pca[:, 0], features_pca[:, 1],
                                  c=colors, cmap='viridis', alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label(feature_name.replace('_', ' ').title(), fontsize=11)

    axes[0].set_xlabel(f'PC1 ({explained_var[0]:.2%} variance)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({explained_var[1]:.2%} variance)', fontsize=12)
    axes[0].set_title(f'PCA: Colored by {feature_name.replace("_", " ").title()}\n' +
                      f'(Silhouette: {silhouette:.3f})', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3, linestyle='--')

    # Plot 2: PCA colored by PC1 (shows intrinsic structure)
    scatter2 = axes[1].scatter(features_pca[:, 0], features_pca[:, 1],
                               c=features_pca[:, 0], cmap='coolwarm',
                               alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
    axes[1].set_xlabel(f'PC1 ({explained_var[0]:.2%} variance)', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({explained_var[1]:.2%} variance)', fontsize=12)
    axes[1].set_title('PCA: Colored by PC1 Value\n(Shows feature space structure)',
                      fontsize=13, fontweight='bold')
    axes[1].grid(alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('PC1 Value', fontsize=11)

    # Plot 3: t-SNE (if computed)
    if compute_tsne:
        if n_unique <= 20:
            for i, val in enumerate(unique_vals):
                mask = metadata_sub[feature_name] == val
                axes[2].scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                                c=[palette[i]], label=f'{val}', alpha=0.7, s=30,
                                edgecolors='w', linewidth=0.5)
            axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                           title=feature_name.replace('_', ' ').title())
        else:
            scatter3 = axes[2].scatter(features_tsne[:, 0], features_tsne[:, 1],
                                       c=colors, cmap='viridis', alpha=0.7, s=30,
                                       edgecolors='w', linewidth=0.5)
            cbar3 = plt.colorbar(scatter3, ax=axes[2])
            cbar3.set_label(feature_name.replace('_', ' ').title(), fontsize=11)

        axes[2].set_xlabel('t-SNE Dimension 1', fontsize=12)
        axes[2].set_ylabel('t-SNE Dimension 2', fontsize=12)
        axes[2].set_title(f't-SNE: Colored by {feature_name.replace("_", " ").title()}\n' +
                          '(Better for revealing clusters)', fontsize=13, fontweight='bold')
        axes[2].grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {save_path}")

    plt.show()

    # ================================================================
    # STEP 6: Summary and Recommendations
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"SUMMARY FOR {feature_name.upper()}")
    print(f"{'=' * 70}")

    results = {
        'feature_name': feature_name,
        'silhouette_score': silhouette if len(np.unique(labels)) > 1 else None,
        'variance_ratio': variance_ratio if len(group_means) > 1 else None,
        'explained_variance': explained_var.sum(),
        'n_unique_values': n_unique
    }

    if results['silhouette_score'] and results['silhouette_score'] > 0.15:
        print("✓ RECOMMENDATION: USE this feature for conditional CP")
        print(f"  - Good visual correlation (Silhouette: {results['silhouette_score']:.3f})")
        print(f"  - Features capture differences in {feature_name}")
    elif results['silhouette_score'] and results['silhouette_score'] > 0.05:
        print("⚠ RECOMMENDATION: CAN USE but expect modest improvements")
        print(f"  - Moderate visual correlation (Silhouette: {results['silhouette_score']:.3f})")
    else:
        print("✗ RECOMMENDATION: DO NOT USE for conditioning")
        print(f"  - Weak/no visual correlation (Silhouette: {results['silhouette_score']:.3f})")

    print(f"{'=' * 70}\n")

    return results


def main():
    """Main function to verify RxRx1 correlations"""

    print("\n" + "=" * 70)
    print("RxRx1 VISUAL CORRELATION VERIFICATION")
    print("=" * 70)

    # Load data
    features, metadata = load_rxrx1_data(
        features_path='features/rxrx1_test.pt',
        metadata_path='data/rxrx1_v1.0/metadata.csv'
    )

    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)

    # Verify correlation for 'experiment' (primary feature)
    print("\n" + "#" * 70)
    print("# ANALYZING: EXPERIMENT (Primary Feature)")
    print("#" * 70)
    results_exp = verify_visual_correlation(
        features,
        metadata,
        'experiment',
        sample_size=5000,
        save_path='figures/rxrx1_experiment_correlation.png'
    )

    # Verify correlation for 'cell_type' (secondary feature)
    print("\n" + "#" * 70)
    print("# ANALYZING: CELL_TYPE (Secondary Feature)")
    print("#" * 70)
    results_cell = verify_visual_correlation(
        features,
        metadata,
        'cell_type',
        sample_size=5000,
        save_path='figures/rxrx1_celltype_correlation.png'
    )

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"Experiment:")
    print(f"  Silhouette: {results_exp['silhouette_score']:.4f}")
    print(f"  Variance Ratio: {results_exp['variance_ratio']:.4f}")
    print(f"  Unique Values: {results_exp['n_unique_values']}")
    print(f"\nCell Type:")
    print(f"  Silhouette: {results_cell['silhouette_score']:.4f}")
    print(f"  Variance Ratio: {results_cell['variance_ratio']:.4f}")
    print(f"  Unique Values: {results_cell['n_unique_values']}")

    if results_exp['silhouette_score'] > results_cell['silhouette_score']:
        print(f"\n✓ 'experiment' shows STRONGER visual correlation")
        print(f"  → Use as PRIMARY conditioning variable")
    else:
        print(f"\n✓ 'cell_type' shows STRONGER visual correlation")
        print(f"  → Use as PRIMARY conditioning variable")

    print("\nBoth features can be used together for multi-variate conditional CP!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
    #
    # features, metadata = load_rxrx1_data()
    # verify_visual_correlation(features[:1000], metadata[:1000], 'experiment',
    #                           sample_size=1000, save_path='test_exp.png')