"""
Meta-Prediction Correction Analysis V2
Using NIMH Healthy Research Volunteer Dataset
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
print("Meta-Prediction Correction Analysis V2")
print("=" * 60)

# Set data path
data_path = PROJECT_ROOT / "data" / "nimh_dataset" / "ds005752" / "phenotype"

# Load BDI data
print("\n1. Loading data...")
bdi_data = pd.read_csv(f"{data_path}/bdi.tsv", sep='\t')

# Load PHQ-9 data
phq9_data = pd.read_csv(f"{data_path}/phq9.tsv", sep='\t')

print(f"BDI data: {bdi_data.shape[0]} participants")
print(f"PHQ-9 data: {phq9_data.shape[0]} participants")

# =============================================================================
# 2. Data Preparation
# =============================================================================

print("\n2. Preparing data...")

# Get BDI item columns (columns starting with 'Q' followed by number)
bdi_item_cols = []
for col in bdi_data.columns:
    if col.startswith('Q') and ',' in col:
        # Extract Q number from the beginning
        q_num = col.split(',')[0].strip()
        if q_num.startswith('Q') and q_num[1:].isdigit():
            bdi_item_cols.append(col)

print(f"Found {len(bdi_item_cols)} BDI items")

# Calculate BDI total score
bdi_data['bdi_total'] = bdi_data[bdi_item_cols].replace(-999, np.nan).sum(axis=1)

# Get PHQ-9 item columns
phq9_items = ['LITTLE_INTEREST', 'FEELING_DOWN', 'TROUBLE_SLEEPING', 'FEELING_TIRED', 
              'POOR_APPETITE', 'FEELING_BAD', 'TROUBLE_CONCENTRATING', 'MOVING_OR_SPEAKING', 
              'BETTER_OFF_DEAD']

# Calculate PHQ-9 total score
phq9_data['phq9_total'] = phq9_data[phq9_items].sum(axis=1)

# Merge datasets on participant_id AND visit to match same time points
merged_data = pd.merge(
    bdi_data[['participant_id', 'visit', 'age_at_visit', 'bdi_total']],
    phq9_data[['participant_id', 'visit', 'age_at_visit', 'phq9_total']],
    on=['participant_id', 'visit'],
    how='inner',
    suffixes=('_bdi', '_phq9')
)

# Use BDI age_at_visit as primary
merged_data['age_at_visit'] = merged_data['age_at_visit_bdi'].fillna(merged_data['age_at_visit_phq9'])

print(f"Merged data: {merged_data.shape[0]} participants")

# =============================================================================
# 3. Develop Proxy Variables for Meta-Cognitive Processes
# =============================================================================

print("\n3. Creating proxy variables...")

# Calculate response pattern indicators from BDI items
bdi_responses = bdi_data[bdi_item_cols].replace(-999, np.nan)

# Extreme responding (proportion of 0 and 3 responses)
bdi_data['extreme_response'] = bdi_responses.apply(
    lambda row: np.mean([1 if x in [0, 3] else 0 for x in row if not np.isnan(x)]), axis=1
)

# Midpoint responding (proportion of 1 and 2 responses)
bdi_data['midpoint_response'] = bdi_responses.apply(
    lambda row: np.mean([1 if x in [1, 2] else 0 for x in row if not np.isnan(x)]), axis=1
)

# Response variability (standard deviation of responses)
bdi_data['response_variability'] = bdi_responses.std(axis=1)

# Missing data proportion
bdi_data['missing_prop'] = bdi_responses.isna().mean(axis=1)

# Merge proxy variables
merged_data = pd.merge(
    merged_data,
    bdi_data[['participant_id', 'extreme_response', 'midpoint_response', 'response_variability', 'missing_prop']],
    on='participant_id',
    how='left'
)

print("Proxy variables created:")
print("  - extreme_response: Proportion of extreme responses (0 or 3)")
print("  - midpoint_response: Proportion of midpoint responses (1 or 2)")
print("  - response_variability: Standard deviation of responses")
print("  - missing_prop: Proportion of missing data")

# =============================================================================
# 4. Create Meta-Cognitive Proxy Variables
# =============================================================================

print("\n4. Creating meta-cognitive proxy variables...")

# Normalize scores for comparison
merged_data['bdi_normalized'] = (merged_data['bdi_total'] - merged_data['bdi_total'].mean()) / merged_data['bdi_total'].std()
merged_data['phq9_normalized'] = (merged_data['phq9_total'] - merged_data['phq9_total'].mean()) / merged_data['phq9_total'].std()

# Consistency score (absolute difference between normalized scores)
merged_data['scale_consistency'] = abs(merged_data['bdi_normalized'] - merged_data['phq9_normalized'])

# Scale Awareness proxy (inverse of inconsistency)
merged_data['scale_awareness'] = 1 - merged_data['scale_consistency']

# Response Adjustment proxy: Extreme responding pattern
merged_data['response_adjustment'] = merged_data['extreme_response']

# Meta-Cognitive Monitoring proxy: Response variability
merged_data['meta_monitoring'] = merged_data['response_variability']

print("Meta-cognitive proxy variables:")
print("  - scale_awareness: Consistency between BDI and PHQ-9")
print("  - response_adjustment: Extreme responding pattern")
print("  - meta_monitoring: Response variability")

# =============================================================================
# 5. Descriptive Statistics
# =============================================================================

print("\n5. Descriptive Statistics:")
print("-" * 60)

# Summary statistics
vars_of_interest = ['bdi_total', 'phq9_total', 'scale_awareness', 'response_adjustment', 'meta_monitoring']
print(merged_data[vars_of_interest].describe().round(3))

# Correlation matrix
print("\nCorrelation Matrix:")
cor_matrix = merged_data[vars_of_interest].corr()
print(cor_matrix.round(3))

# =============================================================================
# 6. Test Meta-Prediction Correction Method
# =============================================================================

print("\n6. Testing Meta-Prediction Correction Method...")
print("-" * 60)

from sklearn.linear_model import LinearRegression

# Prepare data for regression
X = merged_data[['phq9_total', 'scale_awareness', 'response_adjustment', 'meta_monitoring']].dropna()
y = merged_data.loc[X.index, 'bdi_total']

# Model 1: Main effects
model1 = LinearRegression()
model1.fit(X, y)

print("\nModel 1: BDI predicted by PHQ-9 and Meta-Cognitive Proxies")
print(f"R-squared = {model1.score(X, y):.4f}")
print(f"Intercept = {model1.intercept_:.4f}")
print("Coefficients:")
for var, coef in zip(X.columns, model1.coef_):
    print(f"  {var}: {coef:.4f}")

# =============================================================================
# 7. Apply Meta-Prediction Correction
# =============================================================================

print("\n7. Applying Meta-Prediction Correction...")
print("-" * 60)

# Extract coefficients
beta_sa = model1.coef_[1]  # scale_awareness
beta_ra = model1.coef_[2]  # response_adjustment
beta_mcm = model1.coef_[3]  # meta_monitoring

print(f"Correction coefficients:")
print(f"  Scale Awareness (SA): {beta_sa:.4f}")
print(f"  Response Adjustment (RA): {beta_ra:.4f}")
print(f"  Meta-Cognitive Monitoring (MCM): {beta_mcm:.4f}")

# Apply correction
merged_data['bdi_corrected'] = merged_data['bdi_total'] - \
    beta_sa * merged_data['scale_awareness'] - \
    beta_ra * merged_data['response_adjustment'] - \
    beta_mcm * merged_data['meta_monitoring']

merged_data['phq9_corrected'] = merged_data['phq9_total'] - \
    beta_sa * merged_data['scale_awareness'] - \
    beta_ra * merged_data['response_adjustment'] - \
    beta_mcm * merged_data['meta_monitoring']

# =============================================================================
# 8. Validation
# =============================================================================

print("\n8. Validation Results...")
print("-" * 60)

# Compare raw and corrected scores
print(f"BDI Raw - Corrected Correlation: {merged_data['bdi_total'].corr(merged_data['bdi_corrected']):.4f}")
print(f"PHQ-9 Raw - Corrected Correlation: {merged_data['phq9_total'].corr(merged_data['phq9_corrected']):.4f}")

# Test if correction improves concordance between BDI and PHQ-9
print(f"\nConcordance Improvement:")
print(f"Raw BDI-PHQ9 Correlation: {merged_data['bdi_total'].corr(merged_data['phq9_total']):.4f}")
print(f"Corrected BDI-PHQ9 Correlation: {merged_data['bdi_corrected'].corr(merged_data['phq9_corrected']):.4f}")

# =============================================================================
# 9. Visualization
# =============================================================================

print("\n9. Creating visualizations...")

# Create output directory
import os
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Plot 1: Correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix_v2.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Raw vs Corrected scores
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BDI
axes[0].scatter(merged_data['bdi_total'], merged_data['bdi_corrected'], 
                alpha=0.5, s=20, color='steelblue')
axes[0].plot([0, merged_data['bdi_total'].max()], [0, merged_data['bdi_total'].max()], 
             'r--', linewidth=2, label='Perfect Agreement')
axes[0].set_xlabel('BDI Raw Score', fontsize=12)
axes[0].set_ylabel('BDI Corrected Score', fontsize=12)
axes[0].set_title('BDI: Raw vs Corrected', fontsize=14, fontweight='bold')
axes[0].legend()

# PHQ-9
axes[1].scatter(merged_data['phq9_total'], merged_data['phq9_corrected'], 
                alpha=0.5, s=20, color='steelblue')
axes[1].plot([0, merged_data['phq9_total'].max()], [0, merged_data['phq9_total'].max()], 
             'r--', linewidth=2, label='Perfect Agreement')
axes[1].set_xlabel('PHQ-9 Raw Score', fontsize=12)
axes[1].set_ylabel('PHQ-9 Corrected Score', fontsize=12)
axes[1].set_title('PHQ-9: Raw vs Corrected', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "raw_vs_corrected_v2.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# BDI
axes[0].hist(merged_data['bdi_total'], bins=20, alpha=0.5, color='steelblue', label='Raw')
axes[0].hist(merged_data['bdi_corrected'], bins=20, alpha=0.5, color='coral', label='Corrected')
axes[0].set_xlabel('BDI Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('BDI Score Distribution', fontsize=14, fontweight='bold')
axes[0].legend()

# PHQ-9
axes[1].hist(merged_data['phq9_total'], bins=20, alpha=0.5, color='steelblue', label='Raw')
axes[1].hist(merged_data['phq9_corrected'], bins=20, alpha=0.5, color='coral', label='Corrected')
axes[1].set_xlabel('PHQ-9 Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('PHQ-9 Score Distribution', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distribution_comparison_v2.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Meta-cognitive variables vs scores
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scale Awareness
axes[0].scatter(merged_data['scale_awareness'], merged_data['bdi_total'], 
                alpha=0.5, s=20, color='steelblue')
axes[0].set_xlabel('Scale Awareness', fontsize=12)
axes[0].set_ylabel('BDI Score', fontsize=12)
axes[0].set_title('Scale Awareness vs BDI', fontsize=14, fontweight='bold')

# Response Adjustment
axes[1].scatter(merged_data['response_adjustment'], merged_data['bdi_total'], 
                alpha=0.5, s=20, color='steelblue')
axes[1].set_xlabel('Response Adjustment', fontsize=12)
axes[1].set_ylabel('BDI Score', fontsize=12)
axes[1].set_title('Response Adjustment vs BDI', fontsize=14, fontweight='bold')

# Meta Monitoring
axes[2].scatter(merged_data['meta_monitoring'], merged_data['bdi_total'], 
                alpha=0.5, s=20, color='steelblue')
axes[2].set_xlabel('Meta Monitoring', fontsize=12)
axes[2].set_ylabel('BDI Score', fontsize=12)
axes[2].set_title('Meta Monitoring vs BDI', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "metacognitive_variables_v2.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations saved to: {OUTPUT_DIR}")

# =============================================================================
# 10. Save Results
# =============================================================================

print("\n10. Saving results...")

# Save analysis data
merged_data.to_csv(OUTPUT_DIR / "analysis_data_v2.csv", index=False)

# Save model results
with open(OUTPUT_DIR / "model_results_v2.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Meta-Prediction Correction Analysis Results V2\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now()}\n\n")
    
    f.write("=== Model 1: Main Effects ===\n")
    f.write(f"R² = {model1.score(X, y):.4f}\n")
    f.write(f"Intercept = {model1.intercept_:.4f}\n")
    f.write("Coefficients:\n")
    for var, coef in zip(X.columns, model1.coef_):
        f.write(f"  {var}: {coef:.4f}\n")
    
    f.write("\n=== Correction Coefficients ===\n")
    f.write(f"Scale Awareness (SA): {beta_sa:.4f}\n")
    f.write(f"Response Adjustment (RA): {beta_ra:.4f}\n")
    f.write(f"Meta-Cognitive Monitoring (MCM): {beta_mcm:.4f}\n")
    
    f.write("\n=== Validation Results ===\n")
    f.write(f"BDI Raw - Corrected Correlation: {merged_data['bdi_total'].corr(merged_data['bdi_corrected']):.4f}\n")
    f.write(f"PHQ-9 Raw - Corrected Correlation: {merged_data['phq9_total'].corr(merged_data['phq9_corrected']):.4f}\n")
    f.write(f"Raw BDI-PHQ9 Correlation: {merged_data['bdi_total'].corr(merged_data['phq9_total']):.4f}\n")
    f.write(f"Corrected BDI-PHQ9 Correlation: {merged_data['bdi_corrected'].corr(merged_data['phq9_corrected']):.4f}\n")

print(f"Results saved to: {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
