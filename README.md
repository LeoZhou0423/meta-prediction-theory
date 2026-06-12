# Meta-Prediction Theory

Code and paper for "The Meta-Prediction Theory: A Recursive Meta-Cognitive Framework for Understanding Self-Report Bias in Psychological Assessment"

## Repository Structure

```
├── analysis/                  # Analysis scripts
│   ├── sem_analysis.py        # Main OLS regression analysis (Table 4, 7, 8)
│   ├── robustness_no_sa.py    # No-SA robustness analysis (Table 8, 9)
│   ├── proposition_verification.py  # Proposition testing (Table 2)
│   ├── temporal_dynamics_analysis.py  # RT patterns and correlations
│   └── supplementary_analysis.py  # Additional analyses
├── paper/                     # LaTeX source
│   ├── meta_prediction_theory_complete.tex  # Main paper
│   ├── parameter_calibration.tex  # Parameter calibration section
│   ├── references_social_desirability_en.bib  # Bibliography
│   ├── create_figures.py  # Figure generation script
│   └── figures/  # Generated figures
└── README.md
```

## Data

The analysis uses the "Temporal Dynamics in Psychological Assessments" dataset:
- **Source**: https://zenodo.org/records/10423537
- **Citation**: Su et al. (2024). Scientific Data, 11, 1046.

Place the CSV files (`phq9.csv`, `gad7.csv`) in `data/temporal_dynamics/` before running analyses.

## Reproducing Results

1. Install dependencies: `pip install pandas numpy statsmodels scikit-learn matplotlib`
2. Place data files in `data/temporal_dynamics/`
3. Run main analysis: `python analysis/sem_analysis.py`
4. Run robustness analysis: `python analysis/robustness_no_sa.py`
5. Generate figures: `python paper/create_figures.py`

## Key Results

| Model | AIC | BIC | R² |
|-------|-----|-----|----|
| Null Model | 132,614 | 132,622 | .000 |
| Concurrent-Validity Model | 111,131 | 111,147 | .587 |
| Full Model (with SA) | 92,143 | 92,191 | .811 |
| No-SA Model | 94,096 | 94,136 | .795 |

## License

MIT
