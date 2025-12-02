#!/usr/bin/env python3
"""Create comprehensive visualizations for Real Datasets with Winkler score"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("CREATING COMPREHENSIVE VISUALIZATIONS - REAL DATASETS")
print("="*80)

# Load results
results_df = pd.read_csv('results/all_results.csv')
print(f"\n✓ Loaded {len(results_df)} experiment results")

os.makedirs('figures', exist_ok=True)

datasets = results_df['Dataset'].unique()
methods = ['Conformal', 'Parametric', 'Bootstrap']
models = ['AutoARIMA', 'LSTM', 'Theta']

# ===========================================================================
# 1. Coverage Comparison
# ===========================================================================
print("\n[1/7] Creating coverage comparison chart...")

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(methods))
width = 0.25
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, model in enumerate(models):
    model_data = results_df[results_df['Model'] == model]
    coverage_by_method = [model_data[model_data['Method'] == m]['Coverage'].mean() for m in methods]

    ax.bar(x + i*width, coverage_by_method, width, label=model, color=colors[i], alpha=0.8)

ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='Target (90%)', alpha=0.7)

ax.set_xlabel('Prediction Interval Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Coverage', fontsize=12, fontweight='bold')
ax.set_title('Coverage Comparison Across Real Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig('figures/coverage_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: figures/coverage_comparison.png")

# ===========================================================================
# 2. Winkler Score Comparison
# ===========================================================================
print("\n[2/7] Creating Winkler score comparison chart...")

fig, ax = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    model_data = results_df[results_df['Model'] == model]
    winkler_by_method = [model_data[model_data['Method'] == m]['Winkler'].mean() for m in methods]

    ax.bar(x + i*width, winkler_by_method, width, label=model, color=colors[i], alpha=0.8)

ax.set_xlabel('Prediction Interval Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Winkler Score (lower is better)', fontsize=12, fontweight='bold')
ax.set_title('Winkler Score Comparison Across Real Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/winkler_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: figures/winkler_comparison.png")

# ===========================================================================
# 3. Coverage vs Winkler Scatter
# ===========================================================================
print("\n[3/7] Creating coverage vs Winkler scatter plot...")

fig, ax = plt.subplots(figsize=(10, 8))

method_markers = {'Conformal': 'o', 'Parametric': 's', 'Bootstrap': '^'}
model_colors = {'AutoARIMA': '#1f77b4', 'LSTM': '#ff7f0e', 'Theta': '#2ca02c'}

for method in methods:
    for model in models:
        data = results_df[(results_df['Method'] == method) & (results_df['Model'] == model)]
        if not data.empty:
            ax.scatter(data['Winkler'], data['Coverage'],
                      s=100, alpha=0.7,
                      marker=method_markers[method],
                      color=model_colors[model],
                      label=f'{model} - {method}')

ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target Coverage')
ax.set_xlabel('Winkler Score (lower is better)', fontsize=12, fontweight='bold')
ax.set_ylabel('Coverage', fontsize=12, fontweight='bold')
ax.set_title('Coverage vs Winkler Score Trade-off (Real Datasets)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('figures/coverage_vs_winkler.png', dpi=150, bbox_inches='tight')
print("✓ Saved: figures/coverage_vs_winkler.png")

# ===========================================================================
# 4. Per-Dataset Heatmaps
# ===========================================================================
print("\n[4/7] Creating per-dataset heatmaps...")

for dataset_name in datasets:
    dataset_results = results_df[results_df['Dataset'] == dataset_name]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Coverage heatmap
    coverage_pivot = dataset_results.pivot(index='Model', columns='Method', values='Coverage')
    coverage_pivot = coverage_pivot.reindex(index=models, columns=methods)

    im1 = ax1.imshow(coverage_pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=1.0)
    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(methods)
    ax1.set_yticklabels(models)
    ax1.set_title('Coverage (Target: 0.90)', fontweight='bold')

    for i in range(len(models)):
        for j in range(len(methods)):
            value = coverage_pivot.values[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=text_color, fontweight='bold', fontsize=10)

    plt.colorbar(im1, ax=ax1, label='Coverage')

    # Width heatmap
    width_pivot = dataset_results.pivot(index='Model', columns='Method', values='Width')
    width_pivot = width_pivot.reindex(index=models, columns=methods)

    im2 = ax2.imshow(width_pivot.values, cmap='YlOrRd_r', aspect='auto')
    ax2.set_xticks(np.arange(len(methods)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(methods)
    ax2.set_yticklabels(models)
    ax2.set_title('Interval Width (Smaller Better)', fontweight='bold')

    for i in range(len(models)):
        for j in range(len(methods)):
            value = width_pivot.values[i, j]
            if not np.isnan(value):
                ax2.text(j, i, f'{value:.1f}', ha='center', va='center',
                        color='black', fontweight='bold', fontsize=10)

    plt.colorbar(im2, ax=ax2, label='Width')

    # Winkler heatmap
    winkler_pivot = dataset_results.pivot(index='Model', columns='Method', values='Winkler')
    winkler_pivot = winkler_pivot.reindex(index=models, columns=methods)

    im3 = ax3.imshow(winkler_pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax3.set_xticks(np.arange(len(methods)))
    ax3.set_yticks(np.arange(len(models)))
    ax3.set_xticklabels(methods)
    ax3.set_yticklabels(models)
    ax3.set_title('Winkler Score (Smaller Better)', fontweight='bold')

    for i in range(len(models)):
        for j in range(len(methods)):
            value = winkler_pivot.values[i, j]
            if not np.isnan(value):
                ax3.text(j, i, f'{value:.1f}', ha='center', va='center',
                        color='black', fontweight='bold', fontsize=10)

    plt.colorbar(im3, ax=ax3, label='Winkler Score')

    fig.suptitle(f'Dataset: {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    safe_name = dataset_name.replace(' ', '_').replace('/', '_')
    plt.savefig(f'figures/heatmap_{safe_name}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: figures/heatmap_{safe_name}.png")

# ===========================================================================
# 5. Summary Statistics
# ===========================================================================
print("\n[5/7] Creating summary statistics...")

summary_stats = []

for method in methods:
    method_data = results_df[results_df['Method'] == method]
    summary_stats.append({
        'Method': method,
        'Avg_Coverage': method_data['Coverage'].mean(),
        'Std_Coverage': method_data['Coverage'].std(),
        'Avg_Width': method_data['Width'].mean(),
        'Std_Width': method_data['Width'].std(),
        'Avg_Winkler': method_data['Winkler'].mean(),
        'Std_Winkler': method_data['Winkler'].std(),
        'Coverage_MAE': np.abs(method_data['Coverage'] - 0.9).mean()
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('results/summary_statistics.csv', index=False)
print("✓ Saved: results/summary_statistics.csv")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(summary_df.to_string(index=False))

# ===========================================================================
# 6. Comparison Table by Model
# ===========================================================================
print("\n[6/7] Creating comparison by model...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, model in enumerate(models):
    ax = axes[idx]
    model_data = results_df[results_df['Model'] == model]

    metrics = ['Coverage', 'Width', 'Winkler']
    method_scores = {method: [] for method in methods}

    for method in methods:
        method_subset = model_data[model_data['Method'] == method]
        method_scores[method] = [
            method_subset['Coverage'].mean(),
            method_subset['Width'].mean(),
            method_subset['Winkler'].mean()
        ]

    x_pos = np.arange(len(metrics))
    width_bar = 0.25

    for i, method in enumerate(methods):
        scores = method_scores[method]
        ax.bar(x_pos + i*width_bar, scores, width_bar, label=method, alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax.set_title(f'{model} Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos + width_bar)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: figures/model_comparison.png")

# ===========================================================================
# 7. Summary Report
# ===========================================================================
print("\n[7/7] Creating summary report...")

with open('results/summary.md', 'w') as f:
    f.write("# Conformal Prediction Experiments - Real Datasets Results\n\n")
    f.write(f"**Total experiments:** {len(results_df[results_df['Coverage'].notna()])}/{len(results_df)} ✓\n\n")
    f.write(f"**Target coverage:** 90%\n\n")

    f.write("## Overall Results by Method\n\n")
    f.write("| Method | Avg Coverage | Avg Width | Avg Winkler | Coverage MAE |\n")
    f.write("|--------|--------------|-----------|-------------|---------------|\n")
    for _, row in summary_df.iterrows():
        f.write(f"| {row['Method']} | {row['Avg_Coverage']:.3f} | "
                f"{row['Avg_Width']:.2f} | {row['Avg_Winkler']:.2f} | {row['Coverage_MAE']:.3f} |\n")

    f.write("\n## Results by Dataset\n\n")
    for dataset_name in datasets:
        f.write(f"### {dataset_name}\n\n")
        dataset_results = results_df[results_df['Dataset'] == dataset_name]
        f.write("| Model | Method | Coverage | Width | Winkler |\n")
        f.write("|-------|--------|----------|-------|----------|\n")
        for _, row in dataset_results.iterrows():
            f.write(f"| {row['Model']} | {row['Method']} | {row['Coverage']:.3f} | "
                   f"{row['Width']:.2f} | {row['Winkler']:.2f} |\n")
        f.write("\n")

    f.write("## Key Findings\n\n")
    best_coverage = summary_df.loc[summary_df['Coverage_MAE'].idxmin(), 'Method']
    best_winkler = summary_df.loc[summary_df['Avg_Winkler'].idxmin(), 'Method']
    narrowest = summary_df.loc[summary_df['Avg_Width'].idxmin(), 'Method']

    f.write(f"- **Best coverage (closest to 90%):** {best_coverage}\n")
    f.write(f"- **Best Winkler score:** {best_winkler}\n")
    f.write(f"- **Narrowest intervals:** {narrowest}\n\n")

    f.write("## Observations\n\n")
    f.write("- Real-world datasets present different challenges than synthetic data\n")
    f.write("- Conformal methods should maintain coverage guarantees across different real-world scenarios\n")
    f.write("- Comparison with synthetic results reveals method robustness to real-world complexity\n")
    f.write("- Financial data (Gold, Stock, Exchange Rate) may show different patterns than demand data\n")

print("✓ Saved: results/summary.md")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - figures/coverage_comparison.png")
print("  - figures/winkler_comparison.png")
print("  - figures/coverage_vs_winkler.png")
print("  - figures/model_comparison.png")
for dataset_name in datasets:
    safe_name = dataset_name.replace(' ', '_').replace('/', '_')
    print(f"  - figures/heatmap_{safe_name}.png")
print("  - results/summary_statistics.csv")
print("  - results/summary.md")
