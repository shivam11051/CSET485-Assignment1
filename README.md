# CSET485 – AI and Society: Assignment #1

**Name:** Shivam Mishra
**Roll Number:** S24CSEU1470
**Batch:** 56

## Project Overview
This repository contains the completed Jupyter Notebook for Assignment 1 of the CSET485 AI and Society course. The assignment explores the critical issues of algorithmic fairness, bias, and evaluation trade-offs using the COMPAS Recidivism Risk Score dataset.

## Contents
- **`Assignment1_470.ipynb`**: The primary Jupyter Notebook containing the full analysis, reproducible code, and written responses.
- **`Assignment1_470.pdf`**: The exported PDF version of the final notebook.
- **`data/compas-scores-two-years.csv`**: The underlying COMPAS dataset used for all modeling and analysis (Seed: 470).

## Methodology
The assignment conducts a rigorous evaluation of machine learning risk assessment tools, specifically covering:
1. **Model Comparison:** Evaluating differences between threshold-based linear regression and direct logistic regression predictions.
2. **Simpson's Paradox:** Investigating subgroup-specific effects based on gender.
3. **Omitted Variable Bias:** Stress-testing the model by removing critical behavioral confounds and observing the mathematical shift onto demographic proxies.
4. **Adversarial Subsets:** Identifying high-confidence failure modes in specific demographic overlaps.
5. **Fairness Trade-offs:** Sweeping decision thresholds to demonstrate the mathematical impossibility of satisfying Demographic Parity, Equalized Odds, and Predictive Parity simultaneously.

## Reproducibility
All numbers, charts, and statistics cited within the reflection essay and answers can be generated dynamically. To verify the results, simply run the Jupyter Notebook from top to bottom.
