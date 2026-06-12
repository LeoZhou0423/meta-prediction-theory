"""
Create High-Quality Figures for Meta-Prediction Theory Paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project root: parent of paper/ directory
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = Path(__file__).parent / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "temporal_dynamics"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Figure 1: Meta-Prediction Model Diagram
# =============================================================================

print("Creating Figure 1: Meta-Prediction Model Diagram...")

fig, ax = plt.subplots(figsize=(12, 8))

# Define box positions
boxes = {
    'Actual Self': (2, 6),
    'Desired Self': (6, 6),
    'Social Desirability': (10, 6),
    'Scale Cognition': (4, 4),
    'Meta-Cognitive\nMonitoring': (8, 4),
    'Meta-Cognitive\nRegulation': (6, 2),
    'Desired Answer': (6, 0)
}

# Draw boxes
for label, (x, y) in boxes.items():
    if 'Answer' in label:
        color = '#e74c3c'
    elif 'Meta-Cognitive' in label:
        color = '#3498db'
    elif 'Scale' in label:
        color = '#2ecc71'
    else:
        color = '#f39c12'
    
    rect = mpatches.FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, 
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Draw arrows
arrows = [
    ((2, 5.6), (6, 0.4)),  # Actual Self -> Desired Answer
    ((6, 5.6), (6, 0.4)),  # Desired Self -> Desired Answer
    ((10, 5.6), (6, 0.4)),  # Social Desirability -> Desired Answer
    ((4, 3.6), (6, 0.4)),  # Scale Cognition -> Desired Answer
    ((8, 3.6), (6, 0.4)),  # Meta-Cognitive Monitoring -> Desired Answer
    ((6, 1.6), (6, 0.4)),  # Meta-Cognitive Regulation -> Desired Answer
]

for start, end in arrows:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Add labels
ax.text(1, 7, 'Three-Layer\nModel', fontsize=12, fontweight='bold', ha='center')
ax.text(11, 7, 'Extended\nModel', fontsize=12, fontweight='bold', ha='center')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#f39c12', label='Three-Layer Model'),
    mpatches.Patch(facecolor='#2ecc71', label='Scale Cognition'),
    mpatches.Patch(facecolor='#3498db', label='Meta-Cognitive Processes'),
    mpatches.Patch(facecolor='#e74c3c', label='Output'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax.set_xlim(-1, 13)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('The Meta-Prediction Model', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure1_model_diagram.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Correlation Matrix
# =============================================================================

print("Creating Figure 2: Correlation Matrix...")

# Load data
data_path = DATA_DIR
phq9 = pd.read_csv(f"{data_path}/phq9.csv")
gad7 = pd.read_csv(f"{data_path}/gad7.csv")

# Calculate variables
phq9_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7', 'question8', 'question9']
phq9_times = ['time1', 'time2', 'time3', 'time4', 'time5', 
              'time6', 'time7', 'time8', 'time9']
gad7_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7']

phq9['score'] = phq9[phq9_items].sum(axis=1)
phq9['rt_mean'] = phq9[phq9_times].mean(axis=1)
phq9['rt_std'] = phq9[phq9_times].std(axis=1)
phq9['rt_cv'] = phq9['rt_std'] / phq9['rt_mean']

gad7['gad7_total'] = gad7[gad7_items].sum(axis=1)

# Merge
merged = pd.merge(phq9[['export_id', 'score', 'rt_mean', 'rt_std', 'rt_cv']], 
                  gad7[['export_id', 'gad7_total']], on='export_id')

# Calculate proxy variables
merged['phq9_norm'] = (merged['score'] - merged['score'].mean()) / merged['score'].std()
merged['gad7_norm'] = (merged['gad7_total'] - merged['gad7_total'].mean()) / merged['gad7_total'].std()
merged['scale_awareness'] = 1 - abs(merged['phq9_norm'] - merged['gad7_norm'])
merged['meta_monitoring'] = merged['rt_cv']

# Create correlation matrix
vars_for_corr = ['score', 'gad7_total', 'rt_mean', 'rt_cv', 'scale_awareness', 'meta_monitoring']
corr_matrix = merged[vars_for_corr].corr()

# Rename for display
labels = ['PHQ-9', 'GAD-7', 'RT Mean', 'RT CV', 'Scale\nAwareness', 'Meta\nMonitoring']

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, fmt='.2f',
            xticklabels=labels, yticklabels=labels)
ax.set_title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure2_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: Response Time Patterns
# =============================================================================

print("Creating Figure 3: Response Time Patterns...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# RT Mean distribution
axes[0, 0].hist(merged['rt_mean'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Mean Response Time (seconds)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('A. Distribution of Mean Response Time', fontsize=12, fontweight='bold')
axes[0, 0].axvline(merged['rt_mean'].mean(), color='red', linestyle='--', label=f'Mean = {merged["rt_mean"].mean():.2f}')
axes[0, 0].legend()

# RT vs Score
axes[0, 1].scatter(merged['rt_mean'], merged['score'], alpha=0.1, s=10, color='#3498db')
axes[0, 1].set_xlabel('Mean Response Time (seconds)', fontsize=12)
axes[0, 1].set_ylabel('PHQ-9 Score', fontsize=12)
axes[0, 1].set_title('B. Response Time vs PHQ-9 Score', fontsize=12, fontweight='bold')
z = np.polyfit(merged['rt_mean'], merged['score'], 1)
p = np.poly1d(z)
axes[0, 1].plot(sorted(merged['rt_mean']), p(sorted(merged['rt_mean'])), "r--", alpha=0.8)

# Scale Awareness vs Score
axes[1, 0].scatter(merged['scale_awareness'], merged['score'], alpha=0.1, s=10, color='#2ecc71')
axes[1, 0].set_xlabel('Scale Awareness', fontsize=12)
axes[1, 0].set_ylabel('PHQ-9 Score', fontsize=12)
axes[1, 0].set_title('C. Scale Awareness vs PHQ-9 Score', fontsize=12, fontweight='bold')
z = np.polyfit(merged['scale_awareness'].dropna(), merged.loc[merged['scale_awareness'].notna(), 'score'], 1)
p = np.poly1d(z)
x_sorted = sorted(merged['scale_awareness'].dropna())
axes[1, 0].plot(x_sorted, p(x_sorted), "r--", alpha=0.8)

# Meta Monitoring vs Score
axes[1, 1].scatter(merged['meta_monitoring'], merged['score'], alpha=0.1, s=10, color='#e74c3c')
axes[1, 1].set_xlabel('Meta-Cognitive Monitoring (RT CV)', fontsize=12)
axes[1, 1].set_ylabel('PHQ-9 Score', fontsize=12)
axes[1, 1].set_title('D. Meta-Cognitive Monitoring vs PHQ-9 Score', fontsize=12, fontweight='bold')
z = np.polyfit(merged['meta_monitoring'].dropna(), merged.loc[merged['meta_monitoring'].notna(), 'score'], 1)
p = np.poly1d(z)
x_sorted = sorted(merged['meta_monitoring'].dropna())
axes[1, 1].plot(x_sorted, p(x_sorted), "r--", alpha=0.8)

plt.suptitle('Response Time Patterns and Meta-Cognitive Proxies', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure3_response_time_patterns.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: MPC Correction Results
# =============================================================================

print("Creating Figure 4: MPC Correction Results...")

# Apply No-SA correction (primary model: GAD-7 + MM + DEL, excluding SA)
from sklearn.linear_model import LinearRegression

X = merged[['gad7_total', 'meta_monitoring']].dropna()
y = merged.loc[X.index, 'score']

model = LinearRegression()
model.fit(X, y)

# Use only MM for residualization (consistent with No-SA primary model)
merged['phq9_corrected'] = merged['score'] - model.coef_[1] * merged['meta_monitoring']
merged['gad7_corrected'] = merged['gad7_total'] - model.coef_[1] * merged['meta_monitoring']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Raw vs Corrected PHQ-9
axes[0].scatter(merged['score'], merged['phq9_corrected'], alpha=0.1, s=10, color='#3498db')
axes[0].plot([0, 27], [0, 27], 'r--', linewidth=2, label='Perfect Agreement')
axes[0].set_xlabel('PHQ-9 Raw Score', fontsize=12)
axes[0].set_ylabel('PHQ-9 Corrected Score', fontsize=12)
axes[0].set_title('A. PHQ-9: Raw vs Corrected', fontsize=12, fontweight='bold')
axes[0].legend()

# Raw vs Corrected GAD-7
axes[1].scatter(merged['gad7_total'], merged['gad7_corrected'], alpha=0.1, s=10, color='#2ecc71')
axes[1].plot([0, 21], [0, 21], 'r--', linewidth=2, label='Perfect Agreement')
axes[1].set_xlabel('GAD-7 Raw Score', fontsize=12)
axes[1].set_ylabel('GAD-7 Corrected Score', fontsize=12)
axes[1].set_title('B. GAD-7: Raw vs Corrected', fontsize=12, fontweight='bold')
axes[1].legend()

# Correlation comparison
raw_corr = merged['score'].corr(merged['gad7_total'])
corrected_corr = merged['phq9_corrected'].corr(merged['gad7_corrected'])

bars = axes[2].bar(['Raw', 'Corrected'], [raw_corr, corrected_corr], 
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[2].set_ylabel('Correlation', fontsize=12)
axes[2].set_title('C. PHQ-9/GAD-7 Correlation Comparison', fontsize=12, fontweight='bold')
axes[2].set_ylim(0, 1)

# Add value labels
for bar, val in zip(bars, [raw_corr, corrected_corr]):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Add reduction percentage
reduction = (raw_corr - corrected_corr) / raw_corr * 100
axes[2].text(0.5, 0.5, f'Reduction: {reduction:.1f}%', ha='center', va='center',
            transform=axes[2].transAxes, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.suptitle('Meta-Prediction Correction Results', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure4_correction_results.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 5: Model Comparison
# =============================================================================

print("Creating Figure 5: Model Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Null Model', 'Concurrent-Validity\nModel', 'Meta-Prediction\nModel (No-SA)']
aic_values = [132614, 111131, 94096]
colors = ['#95a5a6', '#3498db', '#e74c3c']

bars = ax.bar(models, aic_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels
for bar, val in zip(bars, aic_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
            f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('AIC (lower is better)', fontsize=12)
ax.set_title('Model Comparison: AIC Values', fontsize=14, fontweight='bold')
ax.set_ylim(80000, 140000)

# Add best model annotation
ax.annotate('Best Model', xy=(2, 94096), xytext=(2, 97000),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold', color='red', ha='center')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "figure5_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll figures created successfully!")
print(f"Figures saved to: {FIGURES_DIR}")
