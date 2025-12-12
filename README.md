# Depression Prediction System

A production-grade machine learning system for binary classification of depression based on demographic, lifestyle, academic, and psychosocial factors. This system implements rigorous data leakage prevention, automated hyperparameter optimization, and comprehensive model interpretability.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology](#methodology)
  - [Data Understanding](#data-understanding)
  - [Data Preparation](#data-preparation)
  - [Data Modeling](#data-modeling)
  - [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Output Artifacts](#output-artifacts)
- [Performance Metrics](#performance-metrics)
- [Technical Implementation](#technical-implementation)

## Overview

This system provides an end-to-end pipeline for depression prediction using ensemble machine learning techniques. Key features include:

- **Intelligent Data Preprocessing**: Context-aware missing value imputation with strict leakage prevention
- **Automated Feature Engineering**: Domain-specific feature creation with 35 engineered features
- **Ensemble Modeling**: Five algorithms (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression)
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

Run complete pipeline with all steps:

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

### Individual Steps

Execute specific pipeline stages:

```bash
# Exploratory Data Analysis only
python main.py --eda

# Data Preprocessing only
python main.py --preprocess

# Feature Engineering only
python main.py --engineer

# Model Training only
python main.py --train

# Prediction Generation only
python main.py --predict

# Skip EDA in full pipeline
python main.py --full --skip-eda
```

### Hyperparameter Optimization

Enable Optuna-based hyperparameter tuning:

```bash
python main.py --train --optimize --trials 50
```

### Execution Time

- Basic pipeline (no hyperparameter optimization): 15-25 minutes
- With hyperparameter optimization (--optimize --trials 50): 1-2 hours

## Methodology

This system follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology with four primary phases.

### Data Understanding

**Implementation:** `eda_analysis.py`

The exploratory data analysis phase provides comprehensive insights into the dataset structure, quality, and characteristics.

#### Dataset Profiling

- **Training set:** 140,700 samples, 20 columns (18 features + ID + target)
- **Test set:** 93,800 samples, 19 columns (18 features + ID)
- **Target variable:** Binary classification (Depression: 0/1)

#### Target Distribution Analysis

- Class 0 (No Depression): 115,133 samples (81.8%)
- Class 1 (Depression): 25,567 samples (18.2%)
- Imbalance ratio: 4.5:1 (requires SMOTE for balanced training)

#### Missing Value Pattern Analysis

**Structural Missing Values** (expected based on student vs. professional status):
- Students lack work-related features (Work Pressure, Job Satisfaction, Profession)
- Professionals lack academic features (CGPA, Academic Pressure, Study Satisfaction)

**Anomalous Missing Values** (data quality issues, 6.24% of records):
- Students without CGPA
- Professionals without Profession
- Missing universal features (Financial Stress, Dietary Habits)

#### Feature Categories

**Demographic Features:**
- Age, Gender, City
- Working Professional or Student (binary indicator)

**Academic Features (Students):**
- CGPA: Grade point average (0-10 scale)
- Academic Pressure: Stress level (0-5 scale)
- Study Satisfaction: Satisfaction level (0-5 scale)
- Degree: Educational qualification

**Work Features (Professionals):**
- Profession: Job category
- Work Pressure: Job-related stress (0-5 scale)
- Job Satisfaction: Satisfaction level (0-5 scale)

**Universal Features:**
- Work/Study Hours: Daily hours spent
- Sleep Duration: Categorical sleep hours
- Dietary Habits: Healthy/Moderate/Unhealthy
- Suicidal Thoughts: Yes/No (critical indicator)
- Financial Stress: Financial burden (0-5 scale)
- Family History of Mental Illness: Yes/No

#### Output Artifacts

- `eda_outputs/eda_summary_report.txt`: Comprehensive text report
- `eda_outputs/target_distribution.png`: Class distribution visualization
- `eda_outputs/missing_values.png`: Missing value heatmap
- `eda_outputs/correlation_matrix.png`: Feature correlation analysis

### Data Preparation

**Implementation:** `new_data_preprocessing.py`

The data preparation phase implements rigorous preprocessing with strict data leakage prevention. All transformations use training set statistics exclusively.

#### Phase 1: Data Cleaning

**Anomalous Record Removal:**
- Remove records with >= 50% missing values (extremely corrupted data)
- Conservative strategy: Preserve maximum data for imputation
- Typical removal: < 1% of dataset

**Data Quality Cleaning:**
- Sleep Duration: Standardize to valid categories (5 levels)
- Dietary Habits: Validate categorical values (Healthy/Moderate/Unhealthy)
- Degree: Remove obvious non-degree values (city names, genders)
- Profession: Clean obviously incorrect professions

#### Phase 2: Missing Value Imputation

**Context-Aware Strategy:**

**Structural Missing Values (Expected):**
- Students missing Work Pressure: Impute with 0 (no work pressure)
- Students missing Job Satisfaction: Impute with 3 (neutral)
- Students missing Profession: Impute with 'Student' category
- Professionals missing CGPA: Impute with median professional CGPA
- Professionals missing Academic Pressure: Impute with 0 (no academic pressure)
- Professionals missing Study Satisfaction: Impute with 3 (neutral)

**Universal Feature Imputation:**
- Numerical features: Median imputation (computed from training set only)
- Categorical features: Mode imputation (computed from training set only)
- Training statistics stored and reused for validation/test sets

**Data Leakage Prevention:**
- All imputation values derived exclusively from training set
- Validation and test sets never influence imputation parameters
- `is_training` flag ensures proper separation

#### Phase 3: Feature Encoding

**Categorical Encoding:**
- Binary features: Yes/No mapped to 1/0
- Ordinal features: Sleep Duration encoded 1-5 (Less than 5h to More than 8h)
- Nominal features: Label encoding for Degree, Profession, City, Gender
- Label encoders fit on training set, applied consistently to all sets

**Numerical Features:**
- Preserved as-is after imputation
- No binning or discretization (maintains information)

#### Phase 4: Train-Validation Split

**Stratified Splitting:**
- 80% training, 20% validation (default)
- Stratification preserves class distribution
- Random state 42 for reproducibility

**SMOTE Application:**
- Applied exclusively to training set after split
- Synthetic sample generation for minority class (Depression=1)
- Validation and test sets remain unchanged (real data only)
- Methods: SMOTE (default) or SMOTETomek (optional)

#### Phase 5: Feature Scaling

**Scaling Strategy:**
- RobustScaler (default): Resistant to outliers using median and IQR
- StandardScaler (alternative): Z-score normalization using mean and std

**Outlier Handling:**
- Clip extreme values to 5th and 95th percentiles
- Percentile bounds computed from training set only
- Applied consistently to train/validation/test

**Data Leakage Prevention:**
- Scaler fit exclusively on training set
- Same scaler applied to validation and test sets
- No information from validation/test influences scaling parameters

####Output Artifacts

- `processed_data/X_train.npy`: Training features (after SMOTE)
- `processed_data/y_train.npy`: Training labels (after SMOTE)
- `processed_data/X_val.npy`: Validation features (original data, scaled)
- `processed_data/y_val.npy`: Validation labels
- `processed_data/X_test.npy`: Test features (scaled using training statistics)
- `processed_data/test_ids.npy`: Test sample IDs
- `processed_data/feature_names.txt`: List of 18 preprocessed features

### Data Modeling

The modeling phase implements multiple algorithms, automated optimization, and ensemble techniques.

#### Feature Engineering

**Implementation:** `new_feature_engineering.py`

Creates 35 engineered features from 18 preprocessed features.

**Stress Features (7 features):**
- Total_Stress_Index: Average stress across all domains
- Max_Stress: Maximum stress factor
- Stress_Variance: Consistency of stress levels
- Stress_Saturated: Log-transformed stress (non-linear relationship)
- Academic_Stress_per_CGPA: Academic pressure normalized by performance
- Work_Stress_per_Hour: Work pressure per hour worked
- Combined_Stress_Score: Weighted combination of stress factors

**Lifestyle Features (8 features):**
- Work_Life_Balance: Inverse relationship with work hours
- Overwork_Flag: Binary indicator for excessive hours (>= 10)
- Hours_Saturated: Log-transformed work hours
- Sleep_Optimal: Binary flag for optimal sleep
- Sleep_Poor: Binary flag for insufficient sleep
- Diet_Healthy: Binary flag for healthy diet
- Diet_Unhealthy: Binary flag for unhealthy diet
- Lifestyle_Risk_Score: Combined lifestyle risk factors

**Psychosocial Features (6 features):**
- Suicidal_Financial_Stress: Interaction between suicidal thoughts and financial stress
- Family_History_Financial: Interaction between family history and financial stress
- Suicidal_Family_History: Combined risk from both factors
- High_Risk_Profile: Binary flag for multiple high-risk factors
- Financial_Stress_Severity: Financial stress with family history weighting
- Mental_Health_Risk_Index: Comprehensive mental health risk score

**Satisfaction Features (4 features):**
- Job_Study_Satisfaction_Avg: Average satisfaction across domains
- Satisfaction_Gap: Difference between work and academic satisfaction
- Low_Satisfaction_Flag: Binary indicator for low satisfaction
- Satisfaction_Stress_Ratio: Satisfaction relative to stress

**Academic Features (3 features):**
- CGPA_Academic_Pressure_Interaction: Performance under pressure
- Study_Hours_CGPA_Ratio: Study efficiency metric
- Academic_Performance_Index: Combined academic health indicator

**Interaction Features (5 features):**
- Sleep_Work_Interaction: Sleep quality and work hours relationship
- Financial_Work_Interaction: Financial stress and work hours
- Age_Stress_Interaction: Age and stress level relationship
- CGPA_Satisfaction_Product: Academic performance and satisfaction
- Pressure_Hours_Product: Combined pressure and time commitment

**Statistical Features (2 features):**
- Feature_Mean: Average across normalized key features
- Feature_Std: Standard deviation across key features

**Data Leakage Prevention:**
- All statistics computed from training set exclusively
- Training statistics stored and reused for validation/test
- `is_training` parameter ensures proper separation

**Output Artifacts:**
- `engineered_data/X_train_eng.npy`: Training features (35 features)
- `engineered_data/X_val_eng.npy`: Validation features (35 features)
- `engineered_data/X_test_eng.npy`: Test features (35 features)
- `engineered_data/y_train.npy`: Training labels
- `engineered_data/y_val.npy`: Validation labels
- `engineered_data/test_ids.npy`: Test IDs
- `engineered_data/feature_names_eng.txt`: List of all 35 feature names

#### Model Training

**Implementation:** `new_model_training.py`

Trains five models with comprehensive optimization and evaluation.

**Model Selection:**

Five algorithms with complementary strengths:

1. **Logistic Regression**: Linear baseline model
   - Fast training and inference
   - Interpretable coefficients
   - Good for linearly separable data

2. **Random Forest**: Robust ensemble method
   - 500 trees with max depth 10
   - Handles non-linear relationships
   - Built-in feature importance

3. **XGBoost**: Gradient boosting with regularization
   - Optimized implementation
   - Strong regularization options
   - Excellent accuracy-speed tradeoff

4. **LightGBM**: Fast gradient boosting
   - Leaf-wise tree growth
   - Low memory usage
   - Efficient for large datasets

5. **CatBoost**: Symmetric trees with categorical handling
   - Built-in overfitting protection
   - Robust to parameter choices
   - Superior categorical feature handling

**GPU Acceleration:**
- Automatic GPU detection using PyTorch CUDA and nvidia-smi
- GPU acceleration for XGBoost (device=cuda), LightGBM (device=gpu), CatBoost (task_type=GPU)
- CPU fallback if GPU unavailable

**Learning Rate Optimization (`learning_rate_tuning.py`):**
- Automatic learning rate optimization for tree-based models
- Tests 7 learning rates: [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
- Uses 20,000 sample subset for faster optimization
- Evaluates on full validation set
- Selects learning rate with best validation accuracy
- Saves optimal learning rates to `trained_models/optimal_learning_rates.json`
- Automatically applied during final model training

**Hyperparameter Optimization (Optional):**

Uses Optuna framework with TPE (Tree-structured Parzen Estimator) sampler.

Optimizes key hyperparameters for each tree-based model with 5-fold stratified cross-validation. Configurable number of trials (default: 30). Best parameters selected based on cross-validation F1-score.

**Threshold Optimization (`threshold_tuning.py`):**
- Optimizes classification thresholds for each model
- Tests 81 threshold values from 0.1 to 0.9 (step 0.01)
- Evaluates accuracy for each threshold on validation set
- Selects threshold with highest validation accuracy
- Saves optimal thresholds to `trained_models/optimal_thresholds.json`
- Automatically applied during prediction generation

**Training Process:**

For each model:
1. Initialize with optimal learning rate (if available)
2. Apply hyperparameter optimization (if requested)
3. Train on full training set with early stopping
4. Evaluate on validation set
5. Save model to disk
6. Record training history

**Ensemble Model:**
- Averages probability predictions from XGBoost, LightGBM, and CatBoost
- Uses average of individual optimal thresholds
- Improved robustness and generalization

**Output Artifacts:**
- `trained_models/logistic_regression.pkl`
- `trained_models/random_forest.pkl`
- `trained_models/xgboost.pkl`
- `trained_models/lightgbm.pkl`
- `trained_models/catboost.pkl`
- `trained_models/optimal_learning_rates.json`
- `trained_models/optimal_thresholds.json`
- `training_curves/*.png`: Learning curves for each model
- `training_history/*.csv`: Detailed training logs

#### Model Interpretability

**Implementation:** `model_interpretability.py`

SHAP (SHapley Additive exPlanations) analysis for model understanding.

**Analysis Components:**

1. **Feature Importance Ranking**
   - Global importance across all predictions
   - Ranked by mean absolute SHAP value
   - Identifies most influential features

2. **SHAP Summary Plot**
   - Distribution of SHAP values for each feature
   - Shows both magnitude and direction of effects
   - Color represents feature value (red=high, blue=low)

3. **SHAP Dependence Plots**
   - Relationship between feature value and SHAP value
   - Shows non-linear effects and interactions
   - Generated for top 10 most important features

4. **Individual Predictions**
   - Force plots explaining specific predictions
   - Shows contribution of each feature to final prediction
   - Useful for debugging and understanding edge cases

5. **Ensemble Analysis**
   - Combined analysis for all 5 models
   - Separate analysis for tree models only (XGBoost, LightGBM, CatBoost)
   - Identifies consensus features across models

**Output Artifacts:**
- `interpretability_results/<model_name>/shap_summary.png`
- `interpretability_results/<model_name>/shap_dependence.png`
- `interpretability_results/<model_name>/interpretability_report.txt`
- `interpretability_results/ensemble_all_models/`: Combined analysis
- `interpretability_results/ensemble_tree_models/`: Tree model consensus

### Evaluation

**Implementation:** `evaluation_metrics.py`

Comprehensive model evaluation with multiple metrics and visualizations.

#### Metrics Calculated

**Classification Metrics:**
- Accuracy: Overall correctness
- Precision: Positive predictive value (minimize false positives)
- Recall: Sensitivity (minimize false negatives)
- F1-Score: Harmonic mean of precision and recall
- AUC-ROC: Area under receiver operating characteristic curve

**Confusion Matrix:**
- True Positives (TP): Correctly predicted depression
- True Negatives (TN): Correctly predicted no depression
- False Positives (FP): Incorrectly predicted depression
- False Negatives (FN): Missed depression cases

**Per-Class Metrics:**
- Precision, Recall, F1-Score for each class
- Support: Number of samples per class

#### Visualization Outputs

**ROC Curves:**
- TPR (True Positive Rate) vs FPR (False Positive Rate)
- Diagonal line represents random classifier
- Higher AUC indicates better discrimination

**Confusion Matrix Heatmaps:**
- Visual representation of prediction quality
- Normalized and absolute count versions
- Color-coded for easy interpretation

**Model Comparison Charts:**
- Side-by-side metric comparison
- Identifies best performing model
- Highlights strengths and weaknesses

#### Output Artifacts

- `evaluation_results/evaluation_summary.txt`: Comprehensive text report
- `evaluation_results/evaluation_results.csv`: Tabular metrics for all models
- `evaluation_results/confusion_matrices/`: Confusion matrix plots per model
- `evaluation_results/roc_curves/`: ROC curve plots per model

#### Prediction Generation

**Implementation:** `prediction_pipeline.py`

Generates final predictions on test set.

**Process:**

1. **Model Loading**
   - Load trained model from pickle file
   - Load optimal threshold from JSON

2. **Prediction Generation**
   - Predict probabilities on test features
   - Apply model-specific optimal threshold
   - Generate binary predictions (0/1)

3. **Ensemble Prediction**
   - Average probabilities from XGBoost, LightGBM, CatBoost
   - Apply average of individual optimal thresholds
   - Recommended for best performance

4. **Output Formatting**
   - Two-column format: id, Depression
   - 93,800 rows (one per test sample)
   - Ready for competition submission

**Output Artifacts:**
- `submission_ensemble.csv`: Ensemble predictions (recommended)
- Optionally: `submission_<model_name>.csv` for individual models

## Project Structure

```
0.9429/
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
│
├── src/                                # Source code
│   ├── main.py                        # Pipeline orchestrator
│   ├── eda_analysis.py                # Exploratory data analysis
│   ├── new_data_preprocessing.py      # Data preprocessing with leak prevention
│   ├── new_feature_engineering.py     # Feature engineering
│   ├── new_model_training.py          # Model training and optimization
│   ├── learning_rate_tuning.py        # Learning rate optimization
│   ├── threshold_tuning.py            # Probability threshold optimization
│   ├── evaluation_metrics.py          # Model evaluation
│   ├── model_interpretability.py      # SHAP analysis
│   ├── prediction_pipeline.py         # Prediction generation
│   ├── validate_submission.py         # Submission file validator
│   │
│   └── data/                          # Data directory
│       ├── train.csv                  # Training dataset (140,700 samples)
│       ├── test.csv                   # Test dataset (93,800 samples)
│       └── sample_submission.csv      # Submission format template
│
├── eda_outputs/                       # Generated by EDA phase
│   ├── eda_summary_report.txt
│   ├── target_distribution.png
│   ├── missing_values.png
│   └── correlation_matrix.png
│
├── processed_data/                    # Generated by preprocessing phase
│   ├── X_train.npy                   # Training features (18 features)
│   ├── y_train.npy                   # Training labels
│   ├── X_val.npy                     # Validation features
│   ├── y_val.npy                     # Validation labels
│   ├── X_test.npy                    # Test features
│   ├── test_ids.npy                  # Test IDs
│   └── feature_names.txt             # Feature names
│
├── engineered_data/                   # Generated by feature engineering phase
│   ├── X_train_eng.npy               # Training features (35 features)
│   ├── y_train.npy                   # Training labels
│   ├── X_val_eng.npy                 # Validation features
│   ├── y_val.npy                     # Validation labels
│   ├── X_test_eng.npy                # Test features
│   ├── test_ids.npy                  # Test IDs
│   └── feature_names_eng.txt         # Feature names
│
├── trained_models/                    # Generated by training phase
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── catboost.pkl
│   ├── optimal_learning_rates.json   # Optimized learning rates
│   └── optimal_thresholds.json       # Optimized thresholds
│
├── training_curves/                   # Training visualizations
│   ├── xgboost_learning_curve.png
│   ├── lightgbm_learning_curve.png
│   └── catboost_learning_curve.png
│
├── training_history/                  # Training logs
│   ├── training_summary.csv
│   ├── xgboost_training_history.csv
│   ├── lightgbm_training_history.csv
│   └── catboost_training_history.csv
│
├── evaluation_results/                # Evaluation metrics and plots
│   ├── evaluation_summary.txt
│   ├── evaluation_results.csv
│   ├── confusion_matrices/
│   └── roc_curves/
│
├── interpretability_results/          # SHAP analysis
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── xgboost/
│   ├── lightgbm/
│   ├── catboost/
│   ├── ensemble_all_models/
│   └── ensemble_tree_models/
│
└── submission_ensemble.csv            # Final predictions
```

## Usage Guide

### Basic Usage

**Full Pipeline:**

```bash
cd src
python main.py --full
```

**Individual Phases:**

```bash
# Exploratory Data Analysis only
python main.py --eda

# Data Preprocessing only
python main.py --preprocess

# Feature Engineering only
python main.py --engineer

# Model Training only
python main.py --train

# Model Interpretability only
python main.py --interpret

# Prediction Generation only
python main.py --predict
```

### Advanced Usage

**With Hyperparameter Optimization:**

```bash
# Full pipeline with hyperparameter tuning (30 trials)
python main.py --full --optimize --trials 30

# Training only with extensive optimization (100 trials)
python main.py --train --optimize --trials 100
```

**With Learning Rate Optimization:**

```bash
# Full pipeline with learning rate tuning
python main.py --full --optimize-lr

# Training with both hyperparameter and learning rate optimization
python main.py --train --optimize --optimize-lr --trials 50
```

**Skip EDA (if already completed):**

```bash
python main.py --full --skip-eda
```

**Custom Configuration:**

```bash
# Disable SMOTE
python main.py --train --no-smote

# Custom validation split (30%)
python main.py --train --val-size 0.3

# Use different model for single predictions
python main.py --predict --model xgboost
```

## Configuration

### Default Configuration

The system uses the following default configuration (can be modified in `main.py`):

```python
config = {
    'train_path': 'data/train.csv',           # Training data path
    'test_path': 'data/test.csv',             # Test data path
    'use_smote': True,                        # Enable SMOTE for class imbalance
    'smote_method': 'smote',                  # SMOTE variant: 'smote' or 'smotetomek'
    'val_size': 0.2,                          # Validation set proportion
    'scaling_method': 'robust',               # Scaler: 'robust' or 'standard'
    'optimize_hyperparams': False,            # Enable hyperparameter optimization
    'optimize_learning_rate': False,          # Enable learning rate optimization
    'n_trials': 30,                           # Optuna optimization trials
    'best_model': 'catboost',                 # Model for single predictions
    'ensemble_models': ['xgboost', 'lightgbm', 'catboost'],  # Ensemble components
    'shap_samples': 2000                      # Sample size for SHAP analysis
}
```

### Command-Line Arguments

```
--full              Run complete pipeline
--eda               Run EDA only
--preprocess        Run preprocessing only
--engineer          Run feature engineering only
--train             Run model training only
--interpret         Run interpretability analysis only
--predict           Run prediction generation only

--train-path PATH   Path to training CSV (default: data/train.csv)
--test-path PATH    Path to test CSV (default: data/test.csv)
--no-smote          Disable SMOTE
--val-size FLOAT    Validation set proportion (default: 0.2)
--optimize          Enable hyperparameter optimization
--optimize-lr       Enable learning rate optimization
--trials INT        Number of optimization trials (default: 30)
--model NAME        Model name for single predictions (default: catboost)
--skip-eda          Skip EDA in full pipeline
```

## Output Artifacts

### Data Outputs

**Preprocessed Data:**
- Numpy arrays (.npy): Efficient storage for numerical data
- Feature names (.txt): Human-readable feature list
- Separate files for train/validation/test sets

**Engineered Data:**
- Extended feature set (35 features)
- Same format as preprocessed data
- Maintains data separation

### Model Outputs

**Trained Models:**
- Pickle files (.pkl): Serialized scikit-learn compatible models
- One file per model
- Can be loaded for inference or further analysis

**Optimization Results:**
- `optimal_learning_rates.json`: Best learning rate for each tree model
- `optimal_thresholds.json`: Best classification threshold for each model
- JSON format for easy parsing

### Visualization Outputs

**EDA Visualizations:**
- PNG format at 300 DPI
- Target distribution, missing values, correlations

**Training Curves:**
- Learning curves showing training progress
- Validation performance over epochs/iterations

**Evaluation Plots:**
- ROC curves with AUC scores
- Confusion matrix heatmaps
- Model comparison charts

**SHAP Analysis:**
- Summary plots (feature importance)
- Dependence plots (feature relationships)
- Force plots (individual predictions)

### Reports

**Text Reports:**
- `eda_summary_report.txt`: Comprehensive EDA findings
- `evaluation_summary.txt`: Model performance metrics
- `interpretability_report.txt`: SHAP analysis insights (per model)

**CSV Reports:**
- `evaluation_results.csv`: Tabular metrics for all models
- `training_summary.csv`: Training statistics
- Individual model training history files


### GPU Acceleration

**Automatic Detection:**
- Checks PyTorch CUDA availability
- Falls back to nvidia-smi if PyTorch unavailable
- Sets appropriate device parameters for each model

**Model-Specific Configuration:**
- XGBoost: `device='cuda'`, `tree_method='hist'`
- LightGBM: `device='gpu'`
- CatBoost: `task_type='GPU'`
- CPU fallback: Automatic if GPU unavailable

### Memory Optimization

**Efficient Data Handling:**
- Numpy arrays for numerical data (lower memory footprint)
- Batch processing in SHAP analysis
- Configurable sample sizes for interpretability

### Error Handling

**Robust Design:**
- Try-except blocks for optional GPU features
- Graceful degradation if components unavailable
- Comprehensive error messages


## License

This project is provided for educational and research purposes.

---

For questions or issues, refer to the inline documentation in each source file or check the generated output logs and reports.
