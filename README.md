# An Explainable and Fair Machine-Learning Framework for Credit-Risk Assessment and Dynamic Expected-Loss Optimization

## Project Overview
This repository contains the methodology and codebase for an end-to-end credit risk classification and dynamic Expected Loss (EL) optimization pipeline. It addresses the accuracy-interpretability trade-off in financial machine learning by combining gradient-boosted decision trees (LightGBM) with Shapley Additive Explanations (SHAP), rigorous Out-of-Time (OOT) validation, and cost-sensitive probability calibration.

## Data Access & Setup
This project uses the historical Lending Club accepted loans dataset (2007–2018Q4). Due to GitHub's file size constraints, the raw data (~1.6GB) is not hosted in this repository.

To reproduce the environment:
1. Download `accepted_2007_to_2018Q4.csv.gz` from [Kaggle: wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
2. Place the compressed file directly into the `data/raw/` directory.
3. Run the notebooks in the `notebooks-&-codes/` directory sequentially.

## Repository Structure
* **`data/`**: Directory for raw inputs and processed `.parquet` cohorts.
* **`models/checkpoints/`**: Contains training-fitted preprocessing pipelines (`.joblib`).
* **`notebooks-&-codes/`**: Sequential Jupyter Notebooks for data ingestion, exploratory data analysis, and feature engineering.
* **`src/features/`**: Reusable Python modules containing deterministic row-level rules and chronological split definitions.

## Current Progress (Module 3)
* **Cohort Engineering:** Filtered the raw 2.2 million records to isolate a highly comparable, fully resolved cohort of 603,589 36-month loans (519,362 Fully Paid; 84,227 Charged Off) issued between 2011 and 2015.
* **Leakage Prevention:** Removed all post-origination repayment artifacts from the Probability of Default (PD) feature space.
* **Out-of-Time (OOT) Validation:** Designed a chronological split to evaluate macroeconomic drift (Train: 2011–2013, Validation: 2014, Held-out Test: 2015).
* **Preprocessing:** Fitted baseline and lender-pricing median imputers, standard scalers, and one-hot encoders exclusively on the training partition to prevent data leakage.

## Tech Stack
* **Language:** Python 3.13+
* **Libraries:** Pandas, NumPy, Scikit-Learn, LightGBM, Imbalanced-Learn, SHAP, Matplotlib, Seaborn