# Agricultural Experiment Analysis — Two-Way ANOVA

A Python-based statistical analysis pipeline for agricultural experimental data using two-way factorial ANOVA, Tukey HSD post-hoc testing, diagnostic plots, and statistical visualizations.

## Overview

This project analyzes agricultural experimental data to investigate how different treatments and measurement types affect a response variable.

The analysis includes:

- Two-way factorial ANOVA
- Treatment × Measurement interaction analysis
- Tukey HSD post-hoc comparisons
- Treatment mean separation
- Residual diagnostics
- Q-Q plot for normality assessment
- Residuals vs fitted values plot
- Treatment-wise response distributions
- Treatment means with standard error bars
- Treatment × Measurement heatmaps
- Measurement correlation analysis
- Summary statistics

## Project Structure

```text
anova-agricultural-analysis/
│
├── data/
│   └── dummy data.csv
│
├── outputs/
│   ├── anova_diagnostic_dashboard.png
│   └── comprehensive_heatmaps.png
│
├── anova_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
