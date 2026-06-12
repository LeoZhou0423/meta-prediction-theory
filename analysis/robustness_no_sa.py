"""
Robustness Analysis: Meta-Prediction Model WITHOUT Scale Awareness (SA)
Addresses reviewer M3 concern about SA circularity.
Reports model fit, coefficients, and comparison with full model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temporal_dynamics"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "output"

PHQ9_ITEMS = [f'question{i}' for i in range(1, 10)]
PHQ9_TIMES = [f'time{i}' for i in range(1, 10)]
GAD7_ITEMS = [f'question{i}' for i in range(1, 8)]

print("=" * 70)
print("ROBUSTNESS ANALYSIS: Model WITHOUT Scale Awareness (SA)")
print("=" * 70)

# =============================================================================
# 1. Data Loading (same as sem_analysis.py)
# =============================================================================
print("\n[1/3] Loading data...")

phq9 = pd.read_csv(DATA_PATH / "phq9.csv")
gad7 = pd.read_csv(DATA_PATH / "gad7.csv")

phq9['phq9_total'] = phq9[PHQ9_ITEMS].sum(axis=1)
phq9['rt_mean'] = phq9[PHQ9_TIMES].mean(axis=1)
phq9['rt_std'] = phq9[PHQ9_TIMES].std(axis=1)
phq9['rt_cv'] = phq9['rt_std'] / phq9['rt_mean']
phq9['extreme_response'] = phq9[PHQ9_ITEMS].apply(
    lambda row: np.mean([1 if x in [0, 3] else 0 for x in row]), axis=1
)

gad7['gad7_total'] = gad7[GAD7_ITEMS].sum(axis=1)

df = pd.merge(
    phq9[['export_id', 'phq9_total', 'rt_mean', 'rt_std', 'rt_cv', 'extreme_response'] +
         PHQ9_ITEMS + PHQ9_TIMES],
    gad7[['export_id', 'gad7_total'] + GAD7_ITEMS],
    on='export_id', how='inner'
)

# Meta-cognitive proxy variables
df['phq9_z'] = (df['phq9_total'] - df['phq9_total'].mean()) / df['phq9_total'].std()
df['gad7_z'] = (df['gad7_total'] - df['gad7_total'].mean()) / df['gad7_total'].std()
df['scale_awareness'] = 1 - abs(df['phq9_z'] - df['gad7_z'])
df['meta_monitoring'] = df['rt_cv']
df['response_adjustment'] = df['rt_mean']
df['deliberation'] = 1 - df['extreme_response']

df = df.dropna(subset=['phq9_total', 'gad7_total', 'scale_awareness',
                        'meta_monitoring', 'response_adjustment', 'deliberation'])
N = len(df)
print(f"  N = {N}")

# =============================================================================
# 2. Fit Models
# =============================================================================
print("\n[2/3] Fitting models...")

y = df['phq9_total']

# Null Model
X_null = sm.add_constant(pd.DataFrame(index=df.index).assign(intercept=1.0))
model_null = sm.OLS(y, X_null).fit()

# Concurrent-Validity Model: PHQ-9 ~ GAD-7
X_cv = sm.add_constant(df[['gad7_total']])
model_cv = sm.OLS(y, X_cv).fit()

# Full Meta-Prediction Model (with SA): PHQ-9 ~ GAD-7 + SA + MM + RA + DEL
X_full = sm.add_constant(df[['gad7_total', 'scale_awareness', 'meta_monitoring',
                              'response_adjustment', 'deliberation']])
model_full = sm.OLS(y, X_full).fit()

# NO-SA Meta-Prediction Model: PHQ-9 ~ GAD-7 + MM + RA + DEL
X_nosa = sm.add_constant(df[['gad7_total', 'meta_monitoring',
                              'response_adjustment', 'deliberation']])
model_nosa = sm.OLS(y, X_nosa).fit()

# =============================================================================
# 3. Report Results
# =============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON: Full Model vs. No-SA Model")
print("=" * 70)
print(f"\n{'Model':<40} {'AIC':>10} {'BIC':>10} {'R2':>8} {'dR2 vs CV':>10}")
print("-" * 78)

models = [
    ("Null Model", model_null, None),
    ("Concurrent-Validity Model", model_cv, None),
    ("Full Model (with SA)", model_full, model_cv),
    ("No-SA Model (GAD-7+MM+RA+DEL)", model_nosa, model_cv),
]

for name, model, base in models:
    delta_r2 = model.rsquared - base.rsquared if base else None
    delta_str = f"{delta_r2:.4f}" if delta_r2 is not None else "---"
    print(f"  {name:<38} {model.aic:>10.0f} {model.bic:>10.0f} {model.rsquared:>8.4f} {delta_str:>10}")

# =============================================================================
# 4. No-SA Model Coefficients
# =============================================================================
print("\n" + "=" * 70)
print("No-SA MODEL: Regression Coefficients")
print("=" * 70)
print(f"\n{'Variable':<30} {'B':>10} {'SE':>10} {'95% CI':>25}")
print("-" * 75)

for var in model_nosa.params.index:
    if var == 'const':
        continue
    b = model_nosa.params[var]
    se = model_nosa.bse[var]
    ci_lo = b - 1.96 * se
    ci_hi = b + 1.96 * se
    label = {
        'gad7_total': 'GAD-7',
        'meta_monitoring': 'MM',
        'response_adjustment': 'RA',
        'deliberation': 'DEL',
    }.get(var, var)
    print(f"  {label:<28} {b:>10.4f} {se:>10.4f} [{ci_lo:>8.4f}, {ci_hi:>8.4f}]")

print(f"\n  R2 = {model_nosa.rsquared:.4f}")
print(f"  Adj. R2 = {model_nosa.rsquared_adj:.4f}")

# =============================================================================
# 5. Key Comparison Metrics
# =============================================================================
print("\n" + "=" * 70)
print("KEY COMPARISON: Full Model vs. No-SA Model")
print("=" * 70)

delta_r2_full_vs_nosa = model_full.rsquared - model_nosa.rsquared
delta_aic_full_vs_nosa = model_nosa.aic - model_full.aic  # positive = full is better

print(f"  R2 (Full):    {model_full.rsquared:.4f}")
print(f"  R2 (No-SA):   {model_nosa.rsquared:.4f}")
print(f"  dR2:          {delta_r2_full_vs_nosa:.4f} ({delta_r2_full_vs_nosa/model_full.rsquared*100:.1f}% of Full model R2)")
print(f"  AIC (Full):   {model_full.aic:.0f}")
print(f"  AIC (No-SA):  {model_nosa.aic:.0f}")
print(f"  dAIC:         {delta_aic_full_vs_nosa:.0f} (positive = Full is better)")
print(f"  BIC penalty for SA: {model_nosa.bic - model_full.bic:.0f}")

# Direction of SA effect
sa_effect = model_full.params['scale_awareness']
print(f"\n  SA coefficient: B = {sa_effect:.4f}")
print(f"  Interpretation: SA has {'negative' if sa_effect < 0 else 'positive'} relationship with PHQ-9")
print(f"  Floor effect explanation: Low-symptom participants respond consistently → high SA, low PHQ-9")

# =============================================================================
# 6. Reduction effect without SA
# =============================================================================
print("\n" + "=" * 70)
print("RESIDUALIZATION: No-SA Model")
print("=" * 70)

raw_r = df['phq9_total'].corr(df['gad7_total'])
corrected_phq9_nosa = df['phq9_total'] \
    - model_nosa.params['meta_monitoring'] * df['meta_monitoring'] \
    - model_nosa.params['response_adjustment'] * df['response_adjustment'] \
    - model_nosa.params['deliberation'] * df['deliberation']
corrected_gad7_nosa = df['gad7_total'] \
    - model_nosa.params['meta_monitoring'] * df['meta_monitoring'] \
    - model_nosa.params['response_adjustment'] * df['response_adjustment'] \
    - model_nosa.params['deliberation'] * df['deliberation']
corrected_r_nosa = corrected_phq9_nosa.corr(corrected_gad7_nosa)
reduction_nosa = (raw_r - corrected_r_nosa) / raw_r * 100

print(f"  Raw PHQ9-GAD7 r:        {raw_r:.4f}")
print(f"  Residualized r (No-SA): {corrected_r_nosa:.4f}")
print(f"  Reduction (No-SA):      {reduction_nosa:.1f}%")

# For comparison, full model
corrected_phq9_full = df['phq9_total'] \
    - model_full.params['scale_awareness'] * df['scale_awareness'] \
    - model_full.params['meta_monitoring'] * df['meta_monitoring'] \
    - model_full.params['response_adjustment'] * df['response_adjustment'] \
    - model_full.params['deliberation'] * df['deliberation']
corrected_gad7_full = df['gad7_total'] \
    - model_full.params['scale_awareness'] * df['scale_awareness'] \
    - model_full.params['meta_monitoring'] * df['meta_monitoring'] \
    - model_full.params['response_adjustment'] * df['response_adjustment'] \
    - model_full.params['deliberation'] * df['deliberation']
corrected_r_full = corrected_phq9_full.corr(corrected_gad7_full)
reduction_full = (raw_r - corrected_r_full) / raw_r * 100

print(f"  Residualized r (Full):  {corrected_r_full:.4f}")
print(f"  Reduction (Full):       {reduction_full:.1f}%")
print(f"  SA contribution:        {reduction_full - reduction_nosa:.1f}% reduction points")

# =============================================================================
# 7. Save Results
# =============================================================================
with open(OUTPUT_PATH / "robustness_no_sa_results.txt", "w", encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("ROBUSTNESS ANALYSIS: Model WITHOUT Scale Awareness (SA)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"N = {N}\n\n")
    f.write("Model Comparison:\n")
    f.write(f"  Null Model:              AIC={model_null.aic:.0f}, BIC={model_null.bic:.0f}, R2={model_null.rsquared:.4f}\n")
    f.write(f"  Concurrent-Validity:     AIC={model_cv.aic:.0f}, BIC={model_cv.bic:.0f}, R2={model_cv.rsquared:.4f}\n")
    f.write(f"  Full Model (with SA):    AIC={model_full.aic:.0f}, BIC={model_full.bic:.0f}, R2={model_full.rsquared:.4f}\n")
    f.write(f"  No-SA Model:             AIC={model_nosa.aic:.0f}, BIC={model_nosa.bic:.0f}, R2={model_nosa.rsquared:.4f}\n\n")
    f.write(f"No-SA Model Coefficients:\n")
    for var in model_nosa.params.index:
        if var == 'const':
            continue
        b = model_nosa.params[var]
        se = model_nosa.bse[var]
        f.write(f"  {var}: B={b:.4f}, SE={se:.4f}\n")
    f.write(f"\nR2 comparison: Full={model_full.rsquared:.4f}, No-SA={model_nosa.rsquared:.4f}, d={delta_r2_full_vs_nosa:.4f}\n")
    f.write(f"AIC comparison: Full={model_full.aic:.0f}, No-SA={model_nosa.aic:.0f}, dAIC={delta_aic_full_vs_nosa:.0f}\n")
    f.write(f"Reduction: Full={reduction_full:.1f}%, No-SA={reduction_nosa:.1f}%\n")

print(f"\nResults saved to: {OUTPUT_PATH / 'robustness_no_sa_results.txt'}")
print("\nAnalysis Complete!")
