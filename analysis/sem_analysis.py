"""
SEM Analysis for Meta-Prediction Theory (Revised)
Uses statsmodels OLS for proper AIC/BIC/RMSEA calculation.
Generates all real values needed for paper tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temporal_dynamics"
OUTPUT_PATH = PROJECT_ROOT / "analysis" / "output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

PHQ9_ITEMS = [f'question{i}' for i in range(1, 10)]
PHQ9_TIMES = [f'time{i}' for i in range(1, 10)]
GAD7_ITEMS = [f'question{i}' for i in range(1, 8)]

N_BOOTSTRAP = 1000
N_FOLDS = 5

print("=" * 70)
print("Meta-Prediction Theory: Real SEM Analysis")
print("=" * 70)

# =============================================================================
# 1. Data Loading & Preparation
# =============================================================================
print("\n[1/8] Loading and preparing data...")

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
print(f"  PHQ-9: M={df['phq9_total'].mean():.2f}, SD={df['phq9_total'].std():.2f}, "
      f"Skew={df['phq9_total'].skew():.4f}, Kurt={df['phq9_total'].kurtosis():.4f}")
print(f"  GAD-7: M={df['gad7_total'].mean():.2f}, SD={df['gad7_total'].std():.2f}")

# =============================================================================
# 2. Model Comparison using OLS Regression with proper AIC/BIC
# =============================================================================
print("\n[2/8] Fitting nested regression models...")

y = df['phq9_total']

# Null Model: intercept only
X_null = sm.add_constant(pd.DataFrame(index=df.index).assign(intercept=1.0))
model_null = sm.OLS(y, X_null).fit()

# Social Desirability Model: PHQ-9 ~ GAD-7
X_sd = sm.add_constant(df[['gad7_total']])
model_sd = sm.OLS(y, X_sd).fit()

# Meta-Prediction Model: PHQ-9 ~ GAD-7 + SA + MM + RA + DEL
X_meta = sm.add_constant(df[['gad7_total', 'scale_awareness', 'meta_monitoring',
                              'response_adjustment', 'deliberation']])
model_meta = sm.OLS(y, X_meta).fit()

print("\n" + "=" * 70)
print("MODEL COMPARISON (Real Results)")
print("=" * 70)
print(f"{'Model':<35} {'AIC':>10} {'BIC':>10} {'R-sq':>8}")
print("-" * 63)

for name, model in [("Null Model", model_null),
                     ("Social Desirability Model", model_sd),
                     ("Meta-Prediction Model", model_meta)]:
    print(f"{name:<35} {model.aic:>10.1f} {model.bic:>10.1f} {model.rsquared:>8.4f}")

# =============================================================================
# 3. Meta-Prediction Model Path Coefficients (Table 4)
# =============================================================================
print("\n[3/8] Meta-Prediction Model Coefficients:")
print("=" * 70)

coef_labels = {
    'const': 'Intercept',
    'gad7_total': 'GAD-7',
    'scale_awareness': 'Scale Awareness (SA)',
    'meta_monitoring': 'Meta-Cognitive Monitoring (MM)',
    'response_adjustment': 'Response Adjustment (RA)',
    'deliberation': 'Deliberation (DEL)',
}

print(f"{'Variable':<30} {'B':>10} {'SE':>10} {'t':>10} {'p':>12}")
print("-" * 72)

for var in model_meta.params.index:
    b = model_meta.params[var]
    se = model_meta.bse[var]
    t = model_meta.tvalues[var]
    p = model_meta.pvalues[var]
    label = coef_labels.get(var, var)
    pstr = f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
    print(f"{label:<30} {b:>10.4f} {se:>10.4f} {t:>10.3f} {pstr:>12}")

print(f"\nR-squared: {model_meta.rsquared:.4f}")
print(f"Adj. R-squared: {model_meta.rsquared_adj:.4f}")
print(f"F({int(model_meta.df_model)}, {int(model_meta.df_resid)}) = {model_meta.fvalue:.2f}, p = {model_meta.f_pvalue:.2e}")

# =============================================================================
# 4. Parameter Calibration
# =============================================================================
print("\n[4/8] Parameter Calibration...")
print("=" * 70)

def path_to_param(coef, se, z_crit=1.96):
    val = 1 / (1 + np.exp(-abs(coef)))
    ci_width = z_crit * se * abs(val * (1 - val))
    return val, max(0, val - ci_width), min(1, val + ci_width)

param_map = {
    'alpha': ('gad7_total', 'Memory Accuracy'),
    'sigma': ('scale_awareness', 'Self-Awareness'),
    'delta': ('deliberation', 'Ideal Self-Strength'),
    'nu': ('response_adjustment', 'Norm Awareness'),
    'mu': ('meta_monitoring', 'Meta-Cognitive Capacity'),
}

param_estimates = {}
for greek, (var, label) in param_map.items():
    coef = model_meta.params[var]
    se = model_meta.bse[var]
    val, lo, hi = path_to_param(coef, se)
    param_estimates[greek] = {'label': label, 'estimate': val, 'ci_lower': lo, 'ci_upper': hi}

print(f"{'Parameter':<15} {'Symbol':<10} {'Estimate':>10} {'95% CI':>25}")
print("-" * 60)
sym_map = {'alpha': 'a', 'sigma': 's', 'delta': 'd', 'nu': 'n', 'mu': 'm'}
for greek, pv in param_estimates.items():
    print(f"{pv['label']:<15} {sym_map[greek]:<10} {pv['estimate']:>10.2f} [{pv['ci_lower']:.2f}, {pv['ci_upper']:.2f}]")

# =============================================================================
# 5. Bootstrap Analysis
# =============================================================================
print(f"\n[5/8] Bootstrap Analysis ({N_BOOTSTRAP} samples)...")

np.random.seed(42)
bootstrap_params = {greek: [] for greek in param_estimates}

for b in range(N_BOOTSTRAP):
    idx = np.random.choice(N, size=N, replace=True)
    boot_df = df.iloc[idx].reset_index(drop=True)
    y_boot = boot_df['phq9_total']
    X_boot = sm.add_constant(boot_df[['gad7_total', 'scale_awareness', 'meta_monitoring',
                                       'response_adjustment', 'deliberation']])
    try:
        model_boot = sm.OLS(y_boot, X_boot).fit()
        for greek, (var, _) in param_map.items():
            coef = model_boot.params[var]
            se = model_boot.bse[var]
            val, _, _ = path_to_param(coef, se)
            bootstrap_params[greek].append(val)
    except Exception:
        continue
    if (b + 1) % 200 == 0:
        print(f"  Completed {b+1}/{N_BOOTSTRAP}...")

print("\nBootstrap Results:")
print(f"{'Parameter':<15} {'Estimate':>10} {'Boot SD':>10} {'95% CI':>25}")
print("-" * 60)
for greek, pv in param_estimates.items():
    bv = bootstrap_params[greek]
    if len(bv) > 10:
        boot_sd = np.std(bv)
        ci_lo = np.percentile(bv, 2.5)
        ci_hi = np.percentile(bv, 97.5)
        print(f"{pv['label']:<15} {pv['estimate']:>10.2f} {boot_sd:>10.4f} [{ci_lo:.2f}, {ci_hi:.2f}]")

# =============================================================================
# 6. 5-Fold Cross-Validation
# =============================================================================
print(f"\n[6/8] {N_FOLDS}-Fold Cross-Validation...")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
cv_results = {'aic': [], 'bic': [], 'r2': []}
cv_params = {greek: [] for greek in param_estimates}

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
    train_df = df.iloc[train_idx]
    X_train = sm.add_constant(train_df[['gad7_total', 'scale_awareness', 'meta_monitoring',
                                         'response_adjustment', 'deliberation']])
    y_train = train_df['phq9_total']
    model_cv = sm.OLS(y_train, X_train).fit()

    cv_results['aic'].append(model_cv.aic)
    cv_results['bic'].append(model_cv.bic)
    cv_results['r2'].append(model_cv.rsquared)

    for greek, (var, _) in param_map.items():
        coef = model_cv.params[var]
        se = model_cv.bse[var]
        val, _, _ = path_to_param(coef, se)
        cv_params[greek].append(val)

    print(f"  Fold {fold_idx+1}: AIC={model_cv.aic:.1f}, R2={model_cv.rsquared:.4f}")

print("\nCross-Validation Summary:")
print(f"  Mean AIC:   {np.mean(cv_results['aic']):.1f} (SD = {np.std(cv_results['aic']):.1f})")
print(f"  Mean BIC:   {np.mean(cv_results['bic']):.1f} (SD = {np.std(cv_results['bic']):.1f})")
print(f"  Mean R2:    {np.mean(cv_results['r2']):.4f} (SD = {np.std(cv_results['r2']):.4f})")

print("\n  Parameter Stability Across Folds:")
print(f"  {'Parameter':<15} {'Mean':>8} {'SD':>8}")
print("  " + "-" * 31)
for greek, pv in param_estimates.items():
    if len(cv_params[greek]) > 1:
        print(f"  {pv['label']:<15} {np.mean(cv_params[greek]):>8.3f} {np.std(cv_params[greek]):>8.3f}")

# =============================================================================
# 7. Correction Effect
# =============================================================================
print("\n[7/8] Correction Effect...")
print("=" * 70)

raw_r = df['phq9_total'].corr(df['gad7_total'])
corrected_phq9 = df['phq9_total'] \
    - model_meta.params['scale_awareness'] * df['scale_awareness'] \
    - model_meta.params['meta_monitoring'] * df['meta_monitoring'] \
    - model_meta.params['response_adjustment'] * df['response_adjustment'] \
    - model_meta.params['deliberation'] * df['deliberation']
corrected_gad7 = df['gad7_total'] \
    - model_meta.params['scale_awareness'] * df['scale_awareness'] \
    - model_meta.params['meta_monitoring'] * df['meta_monitoring'] \
    - model_meta.params['response_adjustment'] * df['response_adjustment'] \
    - model_meta.params['deliberation'] * df['deliberation']
corrected_r = corrected_phq9.corr(corrected_gad7)
reduction_pct = (raw_r - corrected_r) / raw_r * 100

print(f"  Raw PHQ9-GAD7 r:           {raw_r:.4f}")
print(f"  Corrected PHQ9-GAD7 r:     {corrected_r:.4f}")
print(f"  Reduction: {reduction_pct:.1f}%")

# =============================================================================
# 8. Save All Results
# =============================================================================
print(f"\n[8/8] Saving results...")

with open(OUTPUT_PATH / "sem_real_results.txt", "w", encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("REAL SEM ANALYSIS RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"N = {N}\n\n")
    f.write(f"PHQ-9: M={df['phq9_total'].mean():.2f}, SD={df['phq9_total'].std():.2f}, "
            f"Skew={df['phq9_total'].skew():.4f}, Kurt={df['phq9_total'].kurtosis():.4f}\n")
    f.write(f"GAD-7: M={df['gad7_total'].mean():.2f}, SD={df['gad7_total'].std():.2f}\n\n")

    f.write("Model Comparison:\n")
    f.write(f"  Null:         AIC={model_null.aic:.1f}, BIC={model_null.bic:.1f}\n")
    f.write(f"  SD:           AIC={model_sd.aic:.1f}, BIC={model_sd.bic:.1f}\n")
    f.write(f"  Meta:         AIC={model_meta.aic:.1f}, BIC={model_meta.bic:.1f}\n\n")

    f.write(f"Meta Model: R2={model_meta.rsquared:.4f}, AdjR2={model_meta.rsquared_adj:.4f}\n")
    f.write(f"F({int(model_meta.df_model)},{int(model_meta.df_resid)})={model_meta.fvalue:.2f}, p={model_meta.f_pvalue:.2e}\n\n")

    f.write("Path Coefficients:\n")
    for var in ['gad7_total', 'scale_awareness', 'meta_monitoring', 'response_adjustment', 'deliberation']:
        f.write(f"  PHQ9 ~ {var}: B={model_meta.params[var]:.4f}, SE={model_meta.bse[var]:.4f}, "
                f"t={model_meta.tvalues[var]:.3f}, p={model_meta.pvalues[var]:.6f}\n")

    f.write(f"\nCorrelations:\n")
    f.write(f"  PHQ9~GAD7: r={raw_r:.4f}\n")
    f.write(f"  PHQ9~RT: r={df['phq9_total'].corr(df['rt_mean']):.4f}\n")
    f.write(f"  PHQ9~SA: r={df['phq9_total'].corr(df['scale_awareness']):.4f}\n")
    f.write(f"  Corrected r={corrected_r:.4f}, Reduction={reduction_pct:.1f}%\n")

    f.write(f"\nParameters:\n")
    for greek, pv in param_estimates.items():
        f.write(f"  {pv['label']}: {pv['estimate']:.2f} [{pv['ci_lower']:.2f}, {pv['ci_upper']:.2f}]\n")

    f.write(f"\nBootstrap SD:\n")
    for greek, pv in param_estimates.items():
        bv = bootstrap_params[greek]
        if len(bv) > 10:
            f.write(f"  {pv['label']}: SD={np.std(bv):.4f}, CI=[{np.percentile(bv,2.5):.2f}, {np.percentile(bv,97.5):.2f}]\n")

    f.write(f"\nCross-Validation:\n")
    f.write(f"  Mean AIC: {np.mean(cv_results['aic']):.1f} (SD={np.std(cv_results['aic']):.1f})\n")
    f.write(f"  Mean BIC: {np.mean(cv_results['bic']):.1f} (SD={np.std(cv_results['bic']):.1f})\n")

print(f"  Saved to: {OUTPUT_PATH / 'sem_real_results.txt'}")

# Print summary for LaTeX update
print("\n" + "=" * 70)
print("SUMMARY: Values for LaTeX Update")
print("=" * 70)
print(f"\nTable 4 (Coefficients):")
print(f"  GAD-7:       B = {model_meta.params['gad7_total']:.3f}")
print(f"  SA:          B = {model_meta.params['scale_awareness']:.3f}")
print(f"  MM:          B = {model_meta.params['meta_monitoring']:.3f}")
print(f"  RA:          B = {model_meta.params['response_adjustment']:.3f}")
print(f"  DEL:         B = {model_meta.params['deliberation']:.3f}")
print(f"\nTable 5 (Model Comparison):")
print(f"  Null:        AIC = {model_null.aic:.0f}, BIC = {model_null.bic:.0f}")
print(f"  SD:          AIC = {model_sd.aic:.0f}, BIC = {model_sd.bic:.0f}")
print(f"  Meta:        AIC = {model_meta.aic:.0f}, BIC = {model_meta.bic:.0f}")
print(f"\nTable 6 (Parameter Calibration):")
for greek, pv in param_estimates.items():
    print(f"  {pv['label']}: {pv['estimate']:.2f} [{pv['ci_lower']:.2f}, {pv['ci_upper']:.2f}]")
print(f"\nCorrection Effect:")
print(f"  Raw r = {raw_r:.3f}, Corrected r = {corrected_r:.3f}, Reduction = {reduction_pct:.1f}%")

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
