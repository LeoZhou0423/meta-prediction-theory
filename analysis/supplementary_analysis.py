"""
Supplementary Analysis Using Public Datasets
Meta-Prediction Theory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project root: parent of analysis/ directory
PROJECT_ROOT = Path(__file__).parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"

# =============================================================================
# 1. Data Loading
# =============================================================================

print("=" * 60)
print("Supplementary Analysis Using Public Datasets")
print("=" * 60)

# Temporal Dynamics dataset
td_path = PROJECT_ROOT / "data" / "temporal_dynamics"
phq9_td = pd.read_csv(td_path / "phq9.csv")
gad7_td = pd.read_csv(td_path / "gad7.csv")

# NIMH dataset
nimh_path = PROJECT_ROOT / "data" / "nimh_dataset" / "ds005752" / "phenotype"
bdi_nimh = pd.read_csv(nimh_path / "bdi.tsv", sep='\t')
phq9_nimh = pd.read_csv(nimh_path / "phq9.tsv", sep='\t')

print(f"Temporal Dynamics: {phq9_td.shape[0]} participants")
print(f"NIMH: {bdi_nimh.shape[0]} participants (BDI), {phq9_nimh.shape[0]} participants (PHQ-9)")

# =============================================================================
# 2. Temporal Dynamics: Subgroup Analysis
# =============================================================================

print("\n" + "=" * 60)
print("2. Temporal Dynamics: Subgroup Analysis")
print("=" * 60)

# Calculate variables
phq9_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7', 'question8', 'question9']
phq9_times = ['time1', 'time2', 'time3', 'time4', 'time5', 
              'time6', 'time7', 'time8', 'time9']
gad7_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7']

phq9_td['score'] = phq9_td[phq9_items].sum(axis=1)
phq9_td['rt_mean'] = phq9_td[phq9_times].mean(axis=1)
phq9_td['rt_std'] = phq9_td[phq9_times].std(axis=1)
phq9_td['rt_cv'] = phq9_td['rt_std'] / phq9_td['rt_mean']

gad7_td['gad7_total'] = gad7_td[gad7_items].sum(axis=1)

merged_td = pd.merge(phq9_td[['export_id', 'score', 'rt_mean', 'rt_std', 'rt_cv']], 
                     gad7_td[['export_id', 'gad7_total']], on='export_id')

merged_td['phq9_norm'] = (merged_td['score'] - merged_td['score'].mean()) / merged_td['score'].std()
merged_td['gad7_norm'] = (merged_td['gad7_total'] - merged_td['gad7_total'].mean()) / merged_td['gad7_total'].std()
merged_td['scale_awareness'] = 1 - abs(merged_td['phq9_norm'] - merged_td['gad7_norm'])
merged_td['meta_monitoring'] = merged_td['rt_cv']

# Subgroup analysis by score level (standard PHQ-9 clinical cutoffs)
# 0-4: Minimal, 5-9: Mild, 10-14: Moderate, 15-27: Moderately Severe/Severe
merged_td['score_group'] = pd.cut(merged_td['score'], bins=[-0.1, 4, 9, 14, 27], 
                                   labels=['Minimal', 'Mild', 'Moderate', 'Severe'])

print("\nSubgroup Analysis by Depression Severity:")
print("-" * 60)

for group in ['Minimal', 'Mild', 'Moderate', 'Severe']:
    subset = merged_td[merged_td['score_group'] == group]
    if len(subset) > 10:
        r_sa, p_sa = stats.pearsonr(subset['scale_awareness'], subset['score'])
        print(f"{group} (n={len(subset)}): Scale Awareness vs PHQ-9: r = {r_sa:.3f}, p = {p_sa:.3e}")

# =============================================================================
# 3. Temporal Dynamics: Non-linear Relationships
# =============================================================================

print("\n" + "=" * 60)
print("3. Temporal Dynamics: Non-linear Relationships")
print("=" * 60)

# Test quadratic relationship
merged_td['sa_squared'] = merged_td['scale_awareness'] ** 2

X_linear = merged_td[['scale_awareness']].dropna()
X_quadratic = merged_td[['scale_awareness', 'sa_squared']].dropna()
y = merged_td.loc[X_linear.index, 'score']

model_linear = LinearRegression().fit(X_linear, y)
model_quadratic = LinearRegression().fit(X_quadratic, y)

r2_linear = model_linear.score(X_linear, y)
r2_quadratic = model_quadratic.score(X_quadratic, y)

print(f"Linear model R-squared: {r2_linear:.4f}")
print(f"Quadratic model R-squared: {r2_quadratic:.4f}")
print(f"Improvement: {r2_quadratic - r2_linear:.4f}")

# =============================================================================
# 4. Temporal Dynamics: Sensitivity Analysis
# =============================================================================

print("\n" + "=" * 60)
print("4. Temporal Dynamics: Sensitivity Analysis")
print("=" * 60)

# Remove outliers (beyond 3 SD)
merged_td_clean = merged_td[
    (np.abs(merged_td['score'] - merged_td['score'].mean()) < 3 * merged_td['score'].std()) &
    (np.abs(merged_td['scale_awareness'] - merged_td['scale_awareness'].mean()) < 3 * merged_td['scale_awareness'].std())
]

print(f"Original sample: {merged_td.shape[0]}")
print(f"Clean sample: {merged_td_clean.shape[0]}")

r_original, _ = stats.pearsonr(merged_td['scale_awareness'], merged_td['score'])
r_clean, _ = stats.pearsonr(merged_td_clean['scale_awareness'], merged_td_clean['score'])

print(f"Original correlation: {r_original:.3f}")
print(f"Clean correlation: {r_clean:.3f}")
print(f"Difference: {abs(r_original - r_clean):.3f}")

# =============================================================================
# 5. NIMH Dataset: Consistency Analysis
# =============================================================================

print("\n" + "=" * 60)
print("5. NIMH Dataset: Consistency Analysis")
print("=" * 60)

# Calculate BDI and PHQ-9 scores
bdi_items = [col for col in bdi_nimh.columns if col.startswith('Q') and ',' in col]
bdi_nimh['bdi_total'] = bdi_nimh[bdi_items].replace(-999, np.nan).sum(axis=1)

phq9_items_nimh = ['LITTLE_INTEREST', 'FEELING_DOWN', 'TROUBLE_SLEEPING', 'FEELING_TIRED', 
                   'POOR_APPETITE', 'FEELING_BAD', 'TROUBLE_CONCENTRATING', 'MOVING_OR_SPEAKING', 
                   'BETTER_OFF_DEAD']
phq9_nimh['phq9_total'] = phq9_nimh[phq9_items_nimh].sum(axis=1)

# Merge NIMH data
nimh_merged = pd.merge(
    bdi_nimh[['participant_id', 'bdi_total']],
    phq9_nimh[['participant_id', 'phq9_total']],
    on='participant_id',
    how='inner'
)

print(f"NIMH merged sample: {nimh_merged.shape[0]}")

if nimh_merged.shape[0] > 5:
    r_nimh, p_nimh = stats.pearsonr(nimh_merged['bdi_total'], nimh_merged['phq9_total'])
    print(f"BDI vs PHQ-9 correlation: {r_nimh:.3f}, p = {p_nimh:.3e}")
    
    # Calculate scale awareness
    nimh_merged['bdi_norm'] = (nimh_merged['bdi_total'] - nimh_merged['bdi_total'].mean()) / nimh_merged['bdi_total'].std()
    nimh_merged['phq9_norm'] = (nimh_merged['phq9_total'] - nimh_merged['phq9_total'].mean()) / nimh_merged['phq9_total'].std()
    nimh_merged['scale_awareness'] = 1 - abs(nimh_merged['bdi_norm'] - nimh_merged['phq9_norm'])
    
    r_sa_nimh, p_sa_nimh = stats.pearsonr(nimh_merged['scale_awareness'], nimh_merged['bdi_total'])
    print(f"Scale Awareness vs BDI: {r_sa_nimh:.3f}, p = {p_sa_nimh:.3e}")

# =============================================================================
# 6. Create Visualization
# =============================================================================

print("\n" + "=" * 60)
print("6. Creating Visualization")
print("=" * 60)

import os
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Subgroup analysis
groups = ['Minimal', 'Mild', 'Moderate', 'Severe']
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
for i, group in enumerate(groups):
    subset = merged_td[merged_td['score_group'] == group]
    if len(subset) > 10:
        r, _ = stats.pearsonr(subset['scale_awareness'], subset['score'])
        axes[0, 0].scatter(subset['scale_awareness'], subset['score'], 
                          alpha=0.3, s=10, color=colors[i], label=f'{group} (r={r:.2f})')
axes[0, 0].set_xlabel('Scale Awareness')
axes[0, 0].set_ylabel('PHQ-9 Score')
axes[0, 0].set_title('A. Subgroup Analysis by Depression Severity')
axes[0, 0].legend()

# Plot 2: Linear vs Quadratic
x_range = np.linspace(merged_td['scale_awareness'].min(), merged_td['scale_awareness'].max(), 100)
y_linear = model_linear.predict(x_range.reshape(-1, 1))
y_quadratic = model_quadratic.predict(np.column_stack([x_range, x_range**2]))
axes[0, 1].scatter(merged_td['scale_awareness'], merged_td['score'], alpha=0.1, s=10, color='steelblue')
axes[0, 1].plot(x_range, y_linear, 'r-', linewidth=2, label=f'Linear (R²={r2_linear:.3f})')
axes[0, 1].plot(x_range, y_quadratic, 'g--', linewidth=2, label=f'Quadratic (R²={r2_quadratic:.3f})')
axes[0, 1].set_xlabel('Scale Awareness')
axes[0, 1].set_ylabel('PHQ-9 Score')
axes[0, 1].set_title('B. Linear vs Quadratic Relationship')
axes[0, 1].legend()

# Plot 3: Sensitivity analysis
axes[1, 0].scatter(merged_td['scale_awareness'], merged_td['score'], alpha=0.1, s=10, color='steelblue', label='Original')
axes[1, 0].scatter(merged_td_clean['scale_awareness'], merged_td_clean['score'], alpha=0.1, s=10, color='red', label='Cleaned')
axes[1, 0].set_xlabel('Scale Awareness')
axes[1, 0].set_ylabel('PHQ-9 Score')
axes[1, 0].set_title(f'C. Sensitivity Analysis\n(Original r={r_original:.3f}, Clean r={r_clean:.3f})')
axes[1, 0].legend()

# Plot 4: NIMH dataset
if nimh_merged.shape[0] > 5:
    axes[1, 1].scatter(nimh_merged['bdi_total'], nimh_merged['phq9_total'], 
                       alpha=0.5, s=30, color='purple')
    axes[1, 1].set_xlabel('BDI Score')
    axes[1, 1].set_ylabel('PHQ-9 Score')
    axes[1, 1].set_title(f'D. NIMH Dataset: BDI vs PHQ-9 (n={nimh_merged.shape[0]})')

plt.suptitle('Supplementary Analysis Results', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "supplementary_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to: {FIGURES_DIR / 'supplementary_analysis.png'}")

# =============================================================================
# 7. Save Results
# =============================================================================

print("\n" + "=" * 60)
print("7. Saving Results")
print("=" * 60)

with open(PAPER_DIR / "supplementary_results.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Supplementary Analysis Results\n")
    f.write("Meta-Prediction Theory\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now()}\n\n")
    
    f.write("=== Temporal Dynamics Dataset (N = 24,292) ===\n\n")
    
    f.write("Subgroup Analysis:\n")
    for group in ['Minimal', 'Mild', 'Moderate', 'Severe']:
        subset = merged_td[merged_td['score_group'] == group]
        if len(subset) > 10:
            r, p = stats.pearsonr(subset['scale_awareness'], subset['score'])
            f.write(f"  {group} (n={len(subset)}): r = {r:.3f}, p = {p:.3e}\n")
    
    f.write(f"\nNon-linear Analysis:\n")
    f.write(f"  Linear R-squared: {r2_linear:.4f}\n")
    f.write(f"  Quadratic R-squared: {r2_quadratic:.4f}\n")
    f.write(f"  Improvement: {r2_quadratic - r2_linear:.4f}\n")
    
    f.write(f"\nSensitivity Analysis:\n")
    f.write(f"  Original: n = {merged_td.shape[0]}, r = {r_original:.3f}\n")
    f.write(f"  Cleaned: n = {merged_td_clean.shape[0]}, r = {r_clean:.3f}\n")
    f.write(f"  Difference: {abs(r_original - r_clean):.3f}\n")
    
    if nimh_merged.shape[0] > 5:
        f.write(f"\n=== NIMH Dataset (n = {nimh_merged.shape[0]}) ===\n")
        f.write(f"BDI vs PHQ-9: r = {r_nimh:.3f}, p = {r_nimh:.3e}\n")
        f.write(f"Scale Awareness vs BDI: r = {r_sa_nimh:.3f}, p = {p_sa_nimh:.3e}\n")

print(f"Results saved to: {PAPER_DIR / 'supplementary_results.txt'}")

print("\n" + "=" * 60)
print("Supplementary Analysis Complete!")
print("=" * 60)
