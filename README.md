# Depression Prediction System

A production-grade machine learning system for binary classification of depression based on demographic, lifestyle, academic, and psychosocial factors. This system implements rigorous data leakage prevention, automated hyperparameter optimization, and comprehensive model interpretability.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Output Artifacts](#output-artifacts)
- [Performance Metrics](#performance-metrics)

## Overview

This system provides an end-to-end pipeline for depression prediction using ensemble machine learning techniques. Key features include:

- **Intelligent Data Preprocessing**: Context-aware missing value imputation with strict leakage prevention
- **Automated Feature Engineering**: Domain-specific feature creation with 35 engineered features
- **Ensemble Modeling**: Five algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- **Hyperparameter Optimization**: Optuna-based Bayesian optimization with cross-validation
- **Learning Rate Tuning**: Automatic optimization for gradient boosting models
- **Threshold Optimization**: Model-specific probability threshold calibration
- **Model Interpretability**: SHAP-based feature importance and explanation analysis
- **GPU Acceleration**: Automatic GPU detection and utilization when available

## System Requirements

### Minimum Requirements

- Python 3.8 or higher
- 8GB RAM (16GB recommended for full pipeline with optimization)
- 2GB disk space for datasets and model artifacts

### Optional

- NVIDIA GPU with CUDA support (for accelerated training)
- 16GB+ RAM (for interpretability analysis with large sample sizes)

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /home/michaellee/Desktop/alex/0.9429
```

### Step 2: Create Virtual Environment

Using Conda (Recommended):

```bash
conda create -n alex python=3.10
conda activate alex
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Full Pipeline Execution

Run the complete pipeline with all steps:

```bash
cd src
python main.py --full
```

This executes:

1. Exploratory Data Analysis
2. Data Preprocessing with leak prevention
3. Feature Engineering (35 features)
4. Model Training (5 models)
5. Model Evaluation
6. Interpretability Analysis (SHAP)
7. Prediction Generation

### With Hyperparameter Optimization

Enable Optuna-based hyperparameter tuning:

```bash
python main.py --full --optimize --trials 50
```

### Skip EDA (if already done)

```bash
python main.py --full --skip-eda
```

### Execution Time

- Basic pipeline (no optimization): 15-25 minutes
- With hyperparameter optimization (50 trials): 1-2 hours

## Methodology

This system follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology with four primary phases.

### Data Understanding

**Implementation:** `eda_analysis.py`

#### Dataset Profiling

- **Training set:** 140,700 samples, 18 features + ID + target
- **Test set:** 93,800 samples, 18 features + ID
- **Target variable:** Binary classification (Depression: 0/1)

#### Target Distribution

- Class 0 (No Depression): 115,133 samples (81.8%)
- Class 1 (Depression): 25,567 samples (18.2%)
- Imbalance ratio: 4.5:1 (requires SMOTE for balanced training)

#### Missing Value Patterns

**Structural Missing Values** (expected based on student vs. professional status):
- Students lack work-related features (Work Pressure, Job Satisfaction, Profession)
- Professionals lack academic features (CGPA, Academic Pressure, Study Satisfaction)

**Anomalous Missing Values** (data quality issues, 6.24% of records):
- Students without CGPA
- Professionals without Profession
- Missing universal features (Financial Stress, Dietary Habits)

#### Feature Categories

| Category | Features |
|----------|----------|
| Demographic | Age, Gender, City, Working Professional or Student |
| Academic | CGPA, Academic Pressure, Study Satisfaction, Degree |
| Occupational | Profession, Work Pressure, Job Satisfaction |
| Lifestyle | Work/Study Hours, Sleep Duration, Dietary Habits |
| Psychosocial | Suicidal Thoughts, Financial Stress, Family History of Mental Illness |

### Data Preparation

**Implementation:** `new_data_preprocessing.py`

**Critical Design Principle:** All transformation statistics (imputation values, encoding mappings, scaling parameters) are computed exclusively from the training partition after the train-validation split. These learned parameters are then applied to transform validation and test sets, ensuring no data leakage.

#### Phase 1: Data Cleaning

Applied to the complete dataset before splitting (deterministic rules, no learned statistics):

- Remove records with >= 50% missing values
- Standardize Sleep Duration categories (5 levels)
- Validate Dietary Habits (Healthy/Moderate/Unhealthy)
- Clean invalid Degree and Profession values

#### Phase 2: Train-Validation Split

Performed **before** any statistical transformations to establish strict separation:

- 80% training, 20% validation (stratified)
- Random state: 42 for reproducibility
- Stratification preserves class distribution (4.5:1 imbalance ratio in both partitions)

**Why split early?** This ensures all subsequent transformation steps compute statistics exclusively from training data, preserving validation set independence for unbiased evaluation.

#### Phase 3: Missing Value Imputation

**Context-Aware Strategy (fit on training data only):**

Missing values are handled differently based on their nature:

**Structural Missing Values** (kept as NaN for tree-based models to learn):

| Population | Feature | Handling |
|------------|---------|----------|
| Students | Work Pressure | Keep as NaN (not applicable) |
| Students | Job Satisfaction | Keep as NaN (not applicable) |
| Professionals | CGPA | Keep as NaN (not applicable) |
| Professionals | Academic Pressure | Keep as NaN (not applicable) |
| Professionals | Study Satisfaction | Keep as NaN (not applicable) |

Tree-based models (XGBoost, LightGBM, CatBoost) handle NaN values natively and learn optimal split directions for missing values.

**Anomalous Missing Values** (statistics computed from training partition only, then applied to all partitions):

| Population | Feature | Imputation |
|------------|---------|------------|
| Students | CGPA (if missing) | Median from student training data |
| Students | Academic Pressure (if missing) | Median from student training data |
| Professionals | Work Pressure (if missing) | Median from professional training data |
| Professionals | Profession (if missing) | Mode from professional training data |
| Students | Profession (if missing) | "Student" |
| All | Other numerical features | Median from training set |
| All | Other categorical features | Mode from training set |

#### Phase 4: Feature Encoding

Encoding mappings learned from training partition, applied to all partitions:

**Ordinal Encoding** (predefined mappings, no data-dependent learning):
- Sleep Duration: Encoded 1-5 based on sleep quality (7-8 hours = optimal = 5)
- Dietary Habits: Encoded 0-3 based on health risk (Healthy=0, Unhealthy=3)

**Label Encoding** (default for high-cardinality features):
- Profession: Simple integer encoding (64 unique categories)
- Gender, Working Professional or Student: Binary integer encoding

**Target Encoding** (evaluated but not default):
- Uses target mean with Bayesian smoothing for each category
- Tested but did not show significant improvement over label encoding
- Available via `--encoding-method target` flag

**One-Hot Encoding** (not used):
- Would create 64+ sparse columns for Profession
- Not practical for high-cardinality features

#### Phase 5: Feature Engineering

**Implementation:** `new_feature_engineering.py`

Creates 35 engineered features from 18 preprocessed features. Any statistics required for feature construction are computed from training partition only.

| Category | Features |
|----------|----------|
| Stress (7) | Total_Stress_Index, Max_Stress, Stress_Variance, Stress_Saturated, Academic_Stress_per_CGPA, Work_Stress_per_Hour, Combined_Stress_Score |
| Lifestyle (8) | Work_Life_Balance, Overwork_Flag, Hours_Saturated, Sleep_Optimal, Sleep_Poor, Diet_Healthy, Diet_Unhealthy, Lifestyle_Risk_Score |
| Psychosocial (6) | Suicidal_Financial_Stress, Family_History_Financial, Suicidal_Family_History, High_Risk_Profile, Financial_Stress_Severity, Mental_Health_Risk_Index |
| Satisfaction (4) | Job_Study_Satisfaction_Avg, Satisfaction_Gap, Low_Satisfaction_Flag, Satisfaction_Stress_Ratio |
| Academic (3) | CGPA_Academic_Pressure_Interaction, Study_Hours_CGPA_Ratio, Academic_Performance_Index |
| Interaction (5) | Sleep_Work_Interaction, Financial_Work_Interaction, Age_Stress_Interaction, CGPA_Satisfaction_Product, Pressure_Hours_Product |
| Statistical (2) | Feature_Mean, Feature_Std |

#### Phase 6: Feature Scaling

Fit on training partition only, transform applied to all partitions:

- **Method:** RobustScaler (resistant to outliers)
- **Outlier handling:** Clip to 5th and 95th percentiles (computed from training data)
- **Leakage prevention:** Scaler fit exclusively on training data; learned parameters applied to validation and test sets

#### Phase 7: SMOTE (Class Imbalance Handling)

Applied **exclusively to training partition** after all other transformations:

- Method: SMOTE with k=5 nearest neighbors
- Result: Balanced class distribution in training set only
- Validation set remains untouched (original imbalanced distribution preserved for realistic evaluation)

**Why SMOTE last and training-only?**
1. Synthetic samples should be generated in the fully transformed feature space (after scaling)
2. Validation set must contain only genuine samples for unbiased performance estimation
3. SMOTE on validation data would cause data leakage and overly optimistic metrics

### Data Modeling

**Implementation:** `new_model_training.py`

**Models:**

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| Logistic Regression | Linear baseline | Default sklearn parameters |
| Random Forest | Ensemble of decision trees | 500 trees, max_depth=10 |
| XGBoost | Gradient boosting with regularization | GPU-accelerated, early stopping |
| LightGBM | Fast gradient boosting | Leaf-wise growth, GPU-accelerated |
| CatBoost | Symmetric trees with ordered boosting | Built-in categorical handling, GPU-accelerated |

**Hyperparameter Optimization:**

- Framework: Optuna with TPE sampler
- Validation: Stratified 5-fold cross-validation
- Objective: F1-score maximization
- Trials: Configurable (default: 30)

**Early Stopping:**

- Patience: 50 iterations without improvement
- Applied to: XGBoost, LightGBM, CatBoost

**Ensemble Model:**

- Combines: XGBoost + LightGBM + CatBoost
- Method: Average probability predictions
- Threshold: Average of individual optimal thresholds

### Evaluation

**Implementation:** `evaluation_metrics.py`

#### Metrics

- Accuracy
- AUC-ROC
- F1-Score
- Precision
- Recall

#### Threshold Optimization

- Tests 81 threshold values (0.10 to 0.90, step 0.01)
- Selects threshold with highest validation accuracy
- Stored in `trained_models/optimal_thresholds.json`

#### Model Interpretability

**Implementation:** `model_interpretability.py`

- Method: SHAP (SHapley Additive exPlanations)
- Sample size: 2,000 instances
- Output: Feature importance rankings, summary plots

## Usage Guide

### Individual Pipeline Stages

Run specific stages independently:

```bash
# Exploratory Data Analysis
python main.py --eda

# Data Preprocessing
python main.py --preprocess

# Feature Engineering
python main.py --engineer

# Feature Selection (optional)
python main.py --feature-selection

# Model Training
python main.py --train

# Model Interpretability
python main.py --interpret

# Prediction Generation
python main.py --predict
```

### Training with Optimization

```bash
# Hyperparameter optimization with 50 trials
python main.py --train --optimize --trials 50

# Learning rate optimization
python main.py --train --optimize-lr

# Both optimizations
python main.py --train --optimize --trials 50 --optimize-lr
```

## Configuration

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--full` | Run complete pipeline | - |
| `--eda` | Run EDA only | - |
| `--preprocess` | Run preprocessing only | - |
| `--engineer` | Run feature engineering only | - |
| `--feature-selection` | Run feature selection only | - |
| `--train` | Run model training only | - |
| `--interpret` | Run interpretability analysis only | - |
| `--predict` | Run prediction only | - |
| `--train-path` | Path to training data | `data/train.csv` |
| `--test-path` | Path to test data | `data/test.csv` |
| `--encoding-method` | Encoding for Profession (`label` or `target`) | `label` |
| `--val-size` | Validation set proportion | `0.2` |
| `--no-smote` | Disable SMOTE | - |
| `--no-scaling` | Disable feature scaling | - |
| `--optimize` | Enable hyperparameter optimization | - |
| `--optimize-lr` | Enable learning rate optimization | - |
| `--trials` | Number of optimization trials | `30` |
| `--skip-eda` | Skip EDA in full pipeline | - |

### Configuration Examples

```bash
# Full pipeline with label encoding (default)
python main.py --full

# Full pipeline with target encoding (alternative)
python main.py --full --encoding-method target

# Full pipeline with hyperparameter optimization
python main.py --full --optimize --trials 50

# Training only with custom validation size
python main.py --train --val-size 0.15

# Full pipeline without SMOTE (ablation study)
python main.py --full --no-smote
```

## Project Structure

```
0.9429/
├── README.md
├── final.md
├── requirements.txt
├── src/
│   ├── main.py                      # Pipeline orchestrator
│   ├── eda_analysis.py              # Exploratory data analysis
│   ├── new_data_preprocessing.py    # Data preprocessing
│   ├── new_feature_engineering.py   # Feature engineering
│   ├── feature_selection.py         # Feature selection
│   ├── new_model_training.py        # Model training
│   ├── evaluation_metrics.py        # Evaluation metrics
│   ├── model_interpretability.py    # SHAP analysis
│   ├── prediction_pipeline.py       # Prediction generation
│   ├── threshold_tuning.py          # Threshold optimization
│   ├── learning_rate_tuning.py      # Learning rate optimization
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   ├── processed_data/              # Preprocessed data
│   ├── engineered_data/             # Feature-engineered data
│   ├── trained_models/              # Saved models
│   ├── evaluation_results/          # Evaluation reports
│   ├── interpretability_results/    # SHAP analysis results
│   ├── eda_outputs/                 # EDA reports and plots
│   └── submission_*.csv             # Prediction files
```

## Output Artifacts

### Submission Files

Generated in `src/` directory:

| File | Description |
|------|-------------|
| `submission_logistic_regression.csv` | Logistic Regression predictions |
| `submission_random_forest.csv` | Random Forest predictions |
| `submission_xgboost.csv` | XGBoost predictions |
| `submission_lightgbm.csv` | LightGBM predictions |
| `submission_catboost.csv` | CatBoost predictions |
| `submission_ensemble.csv` | Ensemble predictions (Recommended) |

### Model Artifacts

Stored in `trained_models/`:

- `logistic_regression.pkl`
- `random_forest.pkl`
- `xgboost.pkl`
- `lightgbm.pkl`
- `catboost.pkl`
- `optimal_thresholds.json`
- `training_metadata.json`

### Interpretability Results

Stored in `interpretability_results/`:

- Individual model reports: `{model_name}/interpretability_report.txt`
- Feature importance: `{model_name}/feature_importance.csv`
- Ensemble reports: `ensemble_all_models/`, `ensemble_tree_models/`

## Performance Metrics

### Validation Set Results

| Model | Accuracy | AUC | F1-Score | Precision | Recall |
|-------|----------|-----|----------|-----------|--------|
| LightGBM | 0.9403 | 0.9752 | 0.8348 | 0.8394 | 0.8302 |
| CatBoost | 0.9402 | 0.9754 | 0.8344 | - | - |
| XGBoost | 0.9390 | 0.9750 | 0.8320 | - | - |
| **Ensemble** | **0.9395** | **0.9755** | **0.8329** | - | - |
| Random Forest | 0.9290 | 0.9724 | 0.8188 | - | - |
| Logistic Regression | 0.9184 | 0.9741 | 0.8034 | - | - |

### Top Feature Importance (SHAP)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Age | 1.474 |
| 2 | Suicidal_Thoughts | 0.820 |
| 3 | Work_Study_Hours | 0.554 |
| 4 | Total_Stress_Index | 0.439 |
| 5 | Profession | 0.393 |
| 6 | Work_Life_Balance | 0.339 |

## Technical Notes

### Data Leakage Prevention

The pipeline enforces strict separation between training and validation data:

1. **Train-Validation Split First**: Performed before any statistical transformations
2. **Fit-on-Train Principle**: All transformation statistics (imputation values, encoding mappings, scaling parameters) are computed exclusively from training partition
3. **Transform-on-All**: Learned parameters are applied to transform both training and validation/test sets
4. **SMOTE Training-Only**: Synthetic oversampling applied exclusively to training partition; validation set contains only genuine samples

This design ensures validation metrics reflect true generalization performance on unseen data.

### GPU Acceleration

- XGBoost: `device="cuda"` with histogram tree method
- LightGBM: `device="gpu"`
- CatBoost: `task_type="GPU"`
- Automatic CPU fallback if GPU unavailable

### Reproducibility

- Data splitting: `random_state=42`
- Model training: `random_state=55`
- All configurations logged in JSON format
