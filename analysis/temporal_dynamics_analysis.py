"""
Meta-Prediction Correction Analysis
Using Temporal Dynamics in Psychological Assessments Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project root: parent of analysis/ directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output"

# =============================================================================
# 1. Data Loading
# =============================================================================

print("=" * 60)
print("Meta-Prediction Correction Analysis")
print("Using Temporal Dynamics Dataset")
print("=" * 60)

# Set data path
data_path = PROJECT_ROOT / "data" / "temporal_dynamics"

# Load data
print("\n1. Loading data...")
phq9 = pd.read_csv(f"{data_path}/phq9.csv")
gad7 = pd.read_csv(f"{data_path}/gad7.csv")

print(f"PHQ-9: {phq9.shape[0]} participants")
print(f"GAD-7: {gad7.shape[0]} participants")

# =============================================================================
# 2. Data Preparation
# =============================================================================

print("\n2. Preparing data...")

# Calculate response time variables for PHQ-9
phq9_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7', 'question8', 'question9']
phq9_times = ['time1', 'time2', 'time3', 'time4', 'time5', 
              'time6', 'time7', 'time8', 'time9']

# Calculate response time statistics
phq9['rt_mean'] = phq9[phq9_times].mean(axis=1)
phq9['rt_std'] = phq9[phq9_times].std(axis=1)
phq9['rt_median'] = phq9[phq9_times].median(axis=1)
phq9['rt_total'] = phq9[phq9_times].sum(axis=1)

# Calculate extreme responding
phq9['extreme_response'] = phq9[phq9_items].apply(
    lambda row: np.mean([1 if x in [0, 3] else 0 for x in row]), axis=1
)

# Calculate midpoint responding
phq9['midpoint_response'] = phq9[phq9_items].apply(
    lambda row: np.mean([1 if x in [1, 2] else 0 for x in row]), axis=1
)

# Calculate GAD-7 scores and response times
gad7_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7']
gad7_times = ['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7']

gad7['gad7_total'] = gad7[gad7_items].sum(axis=1)
gad7['rt_mean'] = gad7[gad7_times].mean(axis=1)
gad7['rt_std'] = gad7[gad7_times].std(axis=1)

# Merge datasets
merged_data = pd.merge(
    phq9[['export_id', 'score', 'rt_mean', 'rt_std', 'rt_median', 'rt_total', 
          'extreme_response', 'midpoint_response']],
    gad7[['export_id', 'gad7_total', 'rt_mean', 'rt_std']],
    on='export_id',
    how='inner',
    suffixes=('_phq9', '_gad7')
)

print(f"Merged data: {merged_data.shape[0]} participants")

# =============================================================================
# 3. Create Meta-Cognitive Proxy Variables
# =============================================================================

print("\n3. Creating meta-cognitive proxy variables...")

# Based on response time theory:
# 1. Longer response times suggest more deliberation (meta-cognitive processing)
# 2. Higher variability suggests uncertainty and monitoring
# 3. Extreme responding may indicate less deliberation

# Scale Awareness proxy: Consistency between PHQ-9 and GAD-7
merged_data['phq9_normalized'] = (merged_data['score'] - merged_data['score'].mean()) / merged_data['score'].std()
merged_data['gad7_normalized'] = (merged_data['gad7_total'] - merged_data['gad7_total'].mean()) / merged_data['gad7_total'].std()
merged_data['scale_consistency'] = abs(merged_data['phq9_normalized'] - merged_data['gad7_normalized'])
merged_data['scale_awareness'] = 1 - merged_data['scale_consistency']

# Response Adjustment proxy: Response time (longer = more deliberation)
merged_data['response_adjustment'] = merged_data['rt_mean_phq9']

# Meta-Cognitive Monitoring proxy: Response time variability
merged_data['meta_monitoring'] = merged_data['rt_std_phq9']

# Extreme responding (inverse of deliberation)
merged_data['deliberation'] = 1 - merged_data['extreme_response']

print("Meta-cognitive proxy variables:")
print("  - scale_awareness: Consistency between PHQ-9 and GAD-7")
print("  - response_adjustment: Mean response time (longer = more deliberation)")
print("  - meta_monitoring: Response time variability")
print("  - deliberation: Inverse of extreme responding")

# =============================================================================
# 4. Descriptive Statistics
# =============================================================================

print("\n4. Descriptive Statistics:")
print("-" * 60)

# Summary statistics
vars_of_interest = ['score', 'gad7_total', 'rt_mean_phq9', 'rt_std_phq9', 
                    'scale_awareness', 'response_adjustment', 'meta_monitoring']
print(merged_data[vars_of_interest].describe().round(3))

# Correlation matrix
print("\nCorrelation Matrix:")
cor_matrix = merged_data[vars_of_interest].corr()
print(cor_matrix.round(3))

# =============================================================================
# 5. Test Meta-Prediction Correction Method
# =============================================================================

print("\n5. Testing Meta-Prediction Correction Method...")
print("-" * 60)

from sklearn.linear_model import LinearRegression

# Prepare data for regression
X = merged_data[['gad7_total', 'scale_awareness', 'response_adjustment', 'meta_monitoring']].dropna()
y = merged_data.loc[X.index, 'score']

# Model 1: Main effects
model1 = LinearRegression()
model1.fit(X, y)

print("\nModel 1: PHQ-9 predicted by GAD-7 and Meta-Cognitive Proxies")
print(f"R-squared = {model1.score(X, y):.4f}")
print(f"Intercept = {model1.intercept_:.4f}")
print("Coefficients:")
for var, coef in zip(X.columns, model1.coef_):
    print(f"  {var}: {coef:.4f}")

# =============================================================================
# 6. Apply Meta-Prediction Correction
# =============================================================================

print("\n6. Applying Meta-Prediction Correction...")
print("-" * 60)

# Extract coefficients
beta_gad7 = model1.coef_[0]
beta_sa = model1.coef_[1]
beta_ra = model1.coef_[2]
beta_mcm = model1.coef_[3]

print(f"Correction coefficients:")
print(f"  GAD-7: {beta_gad7:.4f}")
print(f"  Scale Awareness (SA): {beta_sa:.4f}")
print(f"  Response Adjustment (RA): {beta_ra:.4f}")
print(f"  Meta-Cognitive Monitoring (MCM): {beta_mcm:.4f}")

# Apply correction
merged_data['phq9_corrected'] = merged_data['score'] - \
    beta_sa * merged_data['scale_awareness'] - \
    beta_ra * merged_data['response_adjustment'] - \
    beta_mcm * merged_data['meta_monitoring']

merged_data['gad7_corrected'] = merged_data['gad7_total'] - \
    beta_sa * merged_data['scale_awareness'] - \
    beta_ra * merged_data['response_adjustment'] - \
    beta_mcm * merged_data['meta_monitoring']

# =============================================================================
# 7. Validation
# =============================================================================

print("\n7. Validation Results...")
print("-" * 60)

# Compare raw and corrected scores
print(f"PHQ-9 Raw - Corrected Correlation: {merged_data['score'].corr(merged_data['phq9_corrected']):.4f}")
print(f"GAD-7 Raw - Corrected Correlation: {merged_data['gad7_total'].corr(merged_data['gad7_corrected']):.4f}")

# Test if correction improves concordance between PHQ-9 and GAD-7
print(f"\nConcordance Improvement:")
print(f"Raw PHQ9-GAD7 Correlation: {merged_data['score'].corr(merged_data['gad7_total']):.4f}")
print(f"Corrected PHQ9-GAD7 Correlation: {merged_data['phq9_corrected'].corr(merged_data['gad7_corrected']):.4f}")

# Test if response time predicts scores
print(f"\nResponse Time and Scores:")
print(f"PHQ-9 - RT Mean Correlation: {merged_data['score'].corr(merged_data['rt_mean_phq9']):.4f}")
print(f"PHQ-9 - RT Std Correlation: {merged_data['score'].corr(merged_data['rt_std_phq9']):.4f}")

# =============================================================================
# 8. Visualization
# =============================================================================

print("\n8. Creating visualizations...")

# Create output directory
import os
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Correlation matrix
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temporal_dynamics_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Response time distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PHQ-9 response time
axes[0].hist(merged_data['rt_mean_phq9'], bins=50, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Mean Response Time (seconds)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('PHQ-9 Response Time Distribution', fontsize=14, fontweight='bold')

# Response time vs score
axes[1].scatter(merged_data['rt_mean_phq9'], merged_data['score'], 
                alpha=0.1, s=10, color='steelblue')
axes[1].set_xlabel('Mean Response Time (seconds)', fontsize=12)
axes[1].set_ylabel('PHQ-9 Score', fontsize=12)
axes[1].set_title('Response Time vs PHQ-9 Score', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "response_time_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Raw vs Corrected scores
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PHQ-9
axes[0].scatter(merged_data['score'], merged_data['phq9_corrected'], 
                alpha=0.1, s=10, color='steelblue')
axes[0].plot([0, merged_data['score'].max()], [0, merged_data['score'].max()], 
             'r--', linewidth=2, label='Perfect Agreement')
axes[0].set_xlabel('PHQ-9 Raw Score', fontsize=12)
axes[0].set_ylabel('PHQ-9 Corrected Score', fontsize=12)
axes[0].set_title('PHQ-9: Raw vs Corrected', fontsize=14, fontweight='bold')
axes[0].legend()

# GAD-7
axes[1].scatter(merged_data['gad7_total'], merged_data['gad7_corrected'], 
                alpha=0.1, s=10, color='steelblue')
axes[1].plot([0, merged_data['gad7_total'].max()], [0, merged_data['gad7_total'].max()], 
             'r--', linewidth=2, label='Perfect Agreement')
axes[1].set_xlabel('GAD-7 Raw Score', fontsize=12)
axes[1].set_ylabel('GAD-7 Corrected Score', fontsize=12)
axes[1].set_title('GAD-7: Raw vs Corrected', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_vs_corrected_temporal.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Meta-cognitive variables vs scores
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scale Awareness
axes[0].scatter(merged_data['scale_awareness'], merged_data['score'], 
                alpha=0.1, s=10, color='steelblue')
axes[0].set_xlabel('Scale Awareness', fontsize=12)
axes[0].set_ylabel('PHQ-9 Score', fontsize=12)
axes[0].set_title('Scale Awareness vs PHQ-9', fontsize=14, fontweight='bold')

# Response Adjustment
axes[1].scatter(merged_data['response_adjustment'], merged_data['score'], 
                alpha=0.1, s=10, color='steelblue')
axes[1].set_xlabel('Response Adjustment (RT)', fontsize=12)
axes[1].set_ylabel('PHQ-9 Score', fontsize=12)
axes[1].set_title('Response Time vs PHQ-9', fontsize=14, fontweight='bold')

# Meta Monitoring
axes[2].scatter(merged_data['meta_monitoring'], merged_data['score'], 
                alpha=0.1, s=10, color='steelblue')
axes[2].set_xlabel('Meta Monitoring (RT Variability)', fontsize=12)
axes[2].set_ylabel('PHQ-9 Score', fontsize=12)
axes[2].set_title('RT Variability vs PHQ-9', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metacognitive_temporal.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations saved to: {OUTPUT_DIR}")

# =============================================================================
# 9. Save Results
# =============================================================================

print("\n9. Saving results...")

# Save analysis data
merged_data.to_csv(OUTPUT_DIR / "temporal_dynamics_analysis.csv", index=False)

# Save model results
with open(OUTPUT_DIR / "temporal_dynamics_results.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Meta-Prediction Correction Analysis Results\n")
    f.write("Using Temporal Dynamics Dataset\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now()}\n\n")
    
    f.write("=== Sample Information ===\n")
    f.write(f"Total participants: {merged_data.shape[0]}\n\n")
    
    f.write("=== Model 1: Main Effects ===\n")
    f.write(f"R-squared = {model1.score(X, y):.4f}\n")
    f.write(f"Intercept = {model1.intercept_:.4f}\n")
    f.write("Coefficients:\n")
    for var, coef in zip(X.columns, model1.coef_):
        f.write(f"  {var}: {coef:.4f}\n")
    
    f.write("\n=== Correction Coefficients ===\n")
    f.write(f"GAD-7: {beta_gad7:.4f}\n")
    f.write(f"Scale Awareness (SA): {beta_sa:.4f}\n")
    f.write(f"Response Adjustment (RA): {beta_ra:.4f}\n")
    f.write(f"Meta-Cognitive Monitoring (MCM): {beta_mcm:.4f}\n")
    
    f.write("\n=== Validation Results ===\n")
    f.write(f"PHQ-9 Raw - Corrected Correlation: {merged_data['score'].corr(merged_data['phq9_corrected']):.4f}\n")
    f.write(f"GAD-7 Raw - Corrected Correlation: {merged_data['gad7_total'].corr(merged_data['gad7_corrected']):.4f}\n")
    f.write(f"Raw PHQ9-GAD7 Correlation: {merged_data['score'].corr(merged_data['gad7_total']):.4f}\n")
    f.write(f"Corrected PHQ9-GAD7 Correlation: {merged_data['phq9_corrected'].corr(merged_data['gad7_corrected']):.4f}\n")
    f.write(f"PHQ-9 - RT Mean Correlation: {merged_data['score'].corr(merged_data['rt_mean_phq9']):.4f}\n")
    f.write(f"PHQ-9 - RT Std Correlation: {merged_data['score'].corr(merged_data['rt_std_phq9']):.4f}\n")

print(f"Results saved to: {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
