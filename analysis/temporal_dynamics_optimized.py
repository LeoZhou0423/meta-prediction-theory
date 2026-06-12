"""
Meta-Prediction Correction Analysis - Optimized Version
Using Advanced Response Time Patterns as Meta-Cognitive Proxies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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
print("Meta-Prediction Correction Analysis - Optimized")
print("Using Advanced Response Time Patterns")
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
# 2. Advanced Response Time Features
# =============================================================================

print("\n2. Creating advanced response time features...")

# PHQ-9 items and times
phq9_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7', 'question8', 'question9']
phq9_times = ['time1', 'time2', 'time3', 'time4', 'time5', 
              'time6', 'time7', 'time8', 'time9']

# Basic statistics
phq9['rt_mean'] = phq9[phq9_times].mean(axis=1)
phq9['rt_std'] = phq9[phq9_times].std(axis=1)
phq9['rt_median'] = phq9[phq9_times].median(axis=1)
phq9['rt_total'] = phq9[phq9_times].sum(axis=1)

# Advanced statistics
phq9['rt_cv'] = phq9['rt_std'] / phq9['rt_mean']  # Coefficient of variation
phq9['rt_skew'] = phq9[phq9_times].skew(axis=1)  # Skewness
phq9['rt_kurt'] = phq9[phq9_times].kurtosis(axis=1)  # Kurtosis
phq9['rt_range'] = phq9[phq9_times].max(axis=1) - phq9[phq9_times].min(axis=1)  # Range
phq9['rt_iqr'] = phq9[phq9_times].quantile(0.75, axis=1) - phq9[phq9_times].quantile(0.25, axis=1)  # IQR

# Sequential features (autocorrelation)
def calculate_autocorrelation(row, times):
    """Calculate autocorrelation of response times"""
    rt_values = row[times].values
    if len(rt_values) < 2:
        return np.nan
    return np.corrcoef(rt_values[:-1], rt_values[1:])[0, 1]

phq9['rt_autocorr'] = phq9.apply(lambda row: calculate_autocorrelation(row, phq9_times), axis=1)

# Response pattern features
phq9['extreme_response'] = phq9[phq9_items].apply(
    lambda row: np.mean([1 if x in [0, 3] else 0 for x in row]), axis=1
)
phq9['midpoint_response'] = phq9[phq9_items].apply(
    lambda row: np.mean([1 if x in [1, 2] else 0 for x in row]), axis=1
)

# Response variability (answer pattern)
phq9['answer_variability'] = phq9[phq9_items].std(axis=1)

# Time-answer correlation (within-person)
def calculate_time_answer_correlation(row, items, times):
    """Calculate correlation between response time and answer value"""
    answers = row[items].values
    times_values = row[times].values
    if len(answers) < 2:
        return np.nan
    return np.corrcoef(answers, times_values)[0, 1]

phq9['rt_answer_corr'] = phq9.apply(
    lambda row: calculate_time_answer_correlation(row, phq9_items, phq9_times), axis=1
)

# GAD-7 features
gad7_items = ['question1', 'question2', 'question3', 'question4', 'question5', 
              'question6', 'question7']
gad7_times = ['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 'time7']

gad7['gad7_total'] = gad7[gad7_items].sum(axis=1)
gad7['rt_mean'] = gad7[gad7_times].mean(axis=1)
gad7['rt_std'] = gad7[gad7_times].std(axis=1)
gad7['rt_cv'] = gad7['rt_std'] / gad7['rt_mean']

# =============================================================================
# 3. Create Meta-Cognitive Proxy Variables
# =============================================================================

print("\n3. Creating optimized meta-cognitive proxy variables...")

# Merge datasets
merged_data = pd.merge(
    phq9[['export_id', 'score', 'rt_mean', 'rt_std', 'rt_cv', 'rt_skew', 'rt_kurt',
          'rt_range', 'rt_iqr', 'rt_autocorr', 'extreme_response', 'midpoint_response',
          'answer_variability', 'rt_answer_corr']],
    gad7[['export_id', 'gad7_total', 'rt_mean', 'rt_std', 'rt_cv']],
    on='export_id',
    how='inner',
    suffixes=('_phq9', '_gad7')
)

print(f"Merged data: {merged_data.shape[0]} participants")

# Normalize scores
merged_data['phq9_normalized'] = (merged_data['score'] - merged_data['score'].mean()) / merged_data['score'].std()
merged_data['gad7_normalized'] = (merged_data['gad7_total'] - merged_data['gad7_total'].mean()) / merged_data['gad7_total'].std()

# Scale Awareness: Consistency between PHQ-9 and GAD-7
merged_data['scale_consistency'] = abs(merged_data['phq9_normalized'] - merged_data['gad7_normalized'])
merged_data['scale_awareness'] = 1 - merged_data['scale_consistency']

# Meta-Cognitive Monitoring: Response time variability patterns
merged_data['meta_monitoring'] = merged_data['rt_cv_phq9']  # Coefficient of variation

# Response Adjustment: Sequential patterns
merged_data['response_adjustment'] = merged_data['rt_autocorr']  # Autocorrelation

# Deliberation: Time-answer relationship
merged_data['deliberation'] = merged_data['rt_answer_corr']  # Correlation between time and answer

# Certainty: Inverse of response time variability
merged_data['certainty'] = 1 / (1 + merged_data['rt_std_phq9'])

# =============================================================================
# 4. Descriptive Statistics
# =============================================================================

print("\n4. Descriptive Statistics:")
print("-" * 60)

# Summary statistics
vars_of_interest = ['score', 'gad7_total', 'rt_mean_phq9', 'rt_cv_phq9', 'rt_autocorr',
                    'scale_awareness', 'meta_monitoring', 'response_adjustment', 'deliberation']
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

# Prepare data for regression
X_vars = ['gad7_total', 'scale_awareness', 'meta_monitoring', 'response_adjustment', 'deliberation']
X = merged_data[X_vars].dropna()
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

# Model 2: With interaction terms
merged_data['gad7_x_monitoring'] = merged_data['gad7_total'] * merged_data['meta_monitoring']
merged_data['gad7_x_adjustment'] = merged_data['gad7_total'] * merged_data['response_adjustment']

X2_vars = X_vars + ['gad7_x_monitoring', 'gad7_x_adjustment']
X2 = merged_data[X2_vars].dropna()
y2 = merged_data.loc[X2.index, 'score']

model2 = LinearRegression()
model2.fit(X2, y2)

print("\nModel 2: With Interaction Terms")
print(f"R-squared = {model2.score(X2, y2):.4f}")

# =============================================================================
# 6. Apply Meta-Prediction Correction
# =============================================================================

print("\n6. Applying Meta-Prediction Correction...")
print("-" * 60)

# Extract coefficients from Model 1
beta_gad7 = model1.coef_[0]
beta_sa = model1.coef_[1]
beta_mm = model1.coef_[2]
beta_ra = model1.coef_[3]
beta_del = model1.coef_[4]

print(f"Correction coefficients:")
print(f"  GAD-7: {beta_gad7:.4f}")
print(f"  Scale Awareness (SA): {beta_sa:.4f}")
print(f"  Meta-Cognitive Monitoring (MM): {beta_mm:.4f}")
print(f"  Response Adjustment (RA): {beta_ra:.4f}")
print(f"  Deliberation (DEL): {beta_del:.4f}")

# Apply correction
merged_data['phq9_corrected'] = merged_data['score'] - \
    beta_sa * merged_data['scale_awareness'] - \
    beta_mm * merged_data['meta_monitoring'] - \
    beta_ra * merged_data['response_adjustment'] - \
    beta_del * merged_data['deliberation']

merged_data['gad7_corrected'] = merged_data['gad7_total'] - \
    beta_sa * merged_data['scale_awareness'] - \
    beta_mm * merged_data['meta_monitoring'] - \
    beta_ra * merged_data['response_adjustment'] - \
    beta_del * merged_data['deliberation']

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

# Test response time patterns
print(f"\nResponse Time Patterns:")
print(f"PHQ-9 - RT Mean Correlation: {merged_data['score'].corr(merged_data['rt_mean_phq9']):.4f}")
print(f"PHQ-9 - RT CV Correlation: {merged_data['score'].corr(merged_data['rt_cv_phq9']):.4f}")
print(f"PHQ-9 - RT Autocorr Correlation: {merged_data['score'].corr(merged_data['rt_autocorr']):.4f}")
print(f"PHQ-9 - RT-Answer Correlation: {merged_data['score'].corr(merged_data['rt_answer_corr']):.4f}")

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
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=ax, fmt='.2f')
ax.set_title('Correlation Matrix of Key Variables (Optimized)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "optimized_correlation.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Response time patterns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# RT Mean vs Score
axes[0, 0].scatter(merged_data['rt_mean_phq9'], merged_data['score'], 
                   alpha=0.1, s=10, color='steelblue')
axes[0, 0].set_xlabel('Mean Response Time (seconds)', fontsize=12)
axes[0, 0].set_ylabel('PHQ-9 Score', fontsize=12)
axes[0, 0].set_title('RT Mean vs PHQ-9', fontsize=14, fontweight='bold')

# RT CV vs Score
axes[0, 1].scatter(merged_data['rt_cv_phq9'], merged_data['score'], 
                   alpha=0.1, s=10, color='steelblue')
axes[0, 1].set_xlabel('RT Coefficient of Variation', fontsize=12)
axes[0, 1].set_ylabel('PHQ-9 Score', fontsize=12)
axes[0, 1].set_title('RT CV vs PHQ-9', fontsize=14, fontweight='bold')

# RT Autocorrelation vs Score
axes[0, 2].scatter(merged_data['rt_autocorr'], merged_data['score'], 
                   alpha=0.1, s=10, color='steelblue')
axes[0, 2].set_xlabel('RT Autocorrelation', fontsize=12)
axes[0, 2].set_ylabel('PHQ-9 Score', fontsize=12)
axes[0, 2].set_title('RT Autocorrelation vs PHQ-9', fontsize=14, fontweight='bold')

# RT-Answer Correlation vs Score
axes[1, 0].scatter(merged_data['rt_answer_corr'], merged_data['score'], 
                   alpha=0.1, s=10, color='steelblue')
axes[1, 0].set_xlabel('RT-Answer Correlation', fontsize=12)
axes[1, 0].set_ylabel('PHQ-9 Score', fontsize=12)
axes[1, 0].set_title('RT-Answer Correlation vs PHQ-9', fontsize=14, fontweight='bold')

# Scale Awareness vs Score
axes[1, 1].scatter(merged_data['scale_awareness'], merged_data['score'], 
                   alpha=0.1, s=10, color='steelblue')
axes[1, 1].set_xlabel('Scale Awareness', fontsize=12)
axes[1, 1].set_ylabel('PHQ-9 Score', fontsize=12)
axes[1, 1].set_title('Scale Awareness vs PHQ-9', fontsize=14, fontweight='bold')

# Raw vs Corrected
axes[1, 2].scatter(merged_data['score'], merged_data['phq9_corrected'], 
                   alpha=0.1, s=10, color='steelblue')
axes[1, 2].plot([0, merged_data['score'].max()], [0, merged_data['score'].max()], 
                'r--', linewidth=2, label='Perfect Agreement')
axes[1, 2].set_xlabel('PHQ-9 Raw Score', fontsize=12)
axes[1, 2].set_ylabel('PHQ-9 Corrected Score', fontsize=12)
axes[1, 2].set_title('Raw vs Corrected', fontsize=14, fontweight='bold')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "optimized_patterns.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PHQ-9
axes[0].hist(merged_data['score'], bins=30, alpha=0.5, color='steelblue', label='Raw')
axes[0].hist(merged_data['phq9_corrected'], bins=30, alpha=0.5, color='coral', label='Corrected')
axes[0].set_xlabel('PHQ-9 Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('PHQ-9 Distribution Comparison', fontsize=14, fontweight='bold')
axes[0].legend()

# GAD-7
axes[1].hist(merged_data['gad7_total'], bins=30, alpha=0.5, color='steelblue', label='Raw')
axes[1].hist(merged_data['gad7_corrected'], bins=30, alpha=0.5, color='coral', label='Corrected')
axes[1].set_xlabel('GAD-7 Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('GAD-7 Distribution Comparison', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "optimized_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations saved to: {OUTPUT_DIR}")

# =============================================================================
# 9. Save Results
# =============================================================================

print("\n9. Saving results...")

# Save analysis data
merged_data.to_csv(OUTPUT_DIR / "optimized_analysis.csv", index=False)

# Save model results
with open(OUTPUT_DIR / "optimized_results.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("Meta-Prediction Correction Analysis Results - Optimized\n")
    f.write("Using Advanced Response Time Patterns\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now()}\n\n")
    
    f.write("=== Sample Information ===\n")
    f.write(f"Total participants: {merged_data.shape[0]}\n\n")
    
    f.write("=== Advanced Response Time Features ===\n")
    f.write("1. Basic Statistics: Mean, SD, Median, Total\n")
    f.write("2. Advanced Statistics: CV, Skewness, Kurtosis, Range, IQR\n")
    f.write("3. Sequential Features: Autocorrelation\n")
    f.write("4. Pattern Features: Extreme responding, Midpoint responding\n")
    f.write("5. Variability Features: Answer variability\n")
    f.write("6. Time-Answer Correlation: Within-person RT-answer relationship\n\n")
    
    f.write("=== Model 1: Main Effects ===\n")
    f.write(f"R-squared = {model1.score(X, y):.4f}\n")
    f.write(f"Intercept = {model1.intercept_:.4f}\n")
    f.write("Coefficients:\n")
    for var, coef in zip(X.columns, model1.coef_):
        f.write(f"  {var}: {coef:.4f}\n")
    
    f.write("\n=== Model 2: With Interactions ===\n")
    f.write(f"R-squared = {model2.score(X2, y2):.4f}\n")
    
    f.write("\n=== Correction Coefficients ===\n")
    f.write(f"GAD-7: {beta_gad7:.4f}\n")
    f.write(f"Scale Awareness (SA): {beta_sa:.4f}\n")
    f.write(f"Meta-Cognitive Monitoring (MM): {beta_mm:.4f}\n")
    f.write(f"Response Adjustment (RA): {beta_ra:.4f}\n")
    f.write(f"Deliberation (DEL): {beta_del:.4f}\n")
    
    f.write("\n=== Validation Results ===\n")
    f.write(f"PHQ-9 Raw - Corrected Correlation: {merged_data['score'].corr(merged_data['phq9_corrected']):.4f}\n")
    f.write(f"GAD-7 Raw - Corrected Correlation: {merged_data['gad7_total'].corr(merged_data['gad7_corrected']):.4f}\n")
    f.write(f"Raw PHQ9-GAD7 Correlation: {merged_data['score'].corr(merged_data['gad7_total']):.4f}\n")
    f.write(f"Corrected PHQ9-GAD7 Correlation: {merged_data['phq9_corrected'].corr(merged_data['gad7_corrected']):.4f}\n")
    
    f.write("\n=== Response Time Patterns ===\n")
    f.write(f"PHQ-9 - RT Mean Correlation: {merged_data['score'].corr(merged_data['rt_mean_phq9']):.4f}\n")
    f.write(f"PHQ-9 - RT CV Correlation: {merged_data['score'].corr(merged_data['rt_cv_phq9']):.4f}\n")
    f.write(f"PHQ-9 - RT Autocorr Correlation: {merged_data['score'].corr(merged_data['rt_autocorr']):.4f}\n")
    f.write(f"PHQ-9 - RT-Answer Correlation: {merged_data['score'].corr(merged_data['rt_answer_corr']):.4f}\n")

print(f"Results saved to: {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("Optimized Analysis Complete!")
print("=" * 60)
