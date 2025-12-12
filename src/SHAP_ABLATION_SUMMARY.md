# SHAP Ablation Study Summary

## Overview
Created a proper ablation study comparing feature importance between students and working professionals for depression prediction.

## Methodology

### Correct Approach (shap_ablation_separate_training.py)
The new implementation follows the proper ablation study methodology:

1. **Data Split First**: Split raw training data by student status BEFORE any preprocessing
   - Students: 27,901 samples (19.83%) 
   - Professionals: 112,799 samples (80.17%)
   - Note: The 'Working Professional or Student' feature is kept during preprocessing for context-aware imputation, then removed before modeling

2. **Separate Preprocessing**: Each group gets its own preprocessing pipeline
   - Separate train/validation split
   - Group-specific outlier clipping bounds
   - Group-specific imputation statistics
   - Group-specific feature scaling
   - SMOTE applied separately to balance each group

3. **Separate Feature Engineering**: Each group generates features independently
   - 16 original features -> 34 engineered features per group
   - Feature engineering statistics computed separately

4. **Separate Model Training**: Train XGBoost, LightGBM, and CatBoost for each group
   - Models learn group-specific patterns
   - Students: Higher depression rate (58.56% in validation)
   - Professionals: Lower depression rate (8.18% in validation)

5. **SHAP Analysis**: Compute SHAP values for each group's models
   - Used 1,000 samples per group for robust results
   - Averaged importance across LightGBM and CatBoost (XGBoost has SHAP compatibility issue)
   - Generate comparison visualizations and report

## Key Findings

### Most Important Features for Students:
1. Have you ever had suicidal thoughts? (0.988)
2. Total_Stress_Index (0.636)
3. Academic Pressure (0.408)
4. Age (0.408)
5. Stress_Saturated (0.381)

### Most Important Features for Professionals:
1. Age (2.037)
2. Have you ever had suicidal thoughts? (1.220)
3. Work/Study Hours (0.626)
4. Sleep Duration (0.445)
5. Job Satisfaction (0.383)

### Key Differences:
**More Important for Students:**
- Academic Pressure: +0.408
- Total_Stress_Index: +0.258
- Study Satisfaction: +0.126
- CGPA: +0.047

**More Important for Professionals:**
- Age: -1.629
- Job Satisfaction: -0.383
- Work/Study Hours: -0.327
- Sleep Duration: -0.299
- Work-Life Balance: -0.195

## Results Location
All results saved to: `shap_ablation_separate_results/`

Generated files:
- `feature_importance_students.csv` - Full importance ranking for students
- `feature_importance_professionals.csv` - Full importance ranking for professionals
- `feature_importance_comparison.csv` - Side-by-side comparison with differences
- `comparison_side_by_side.png` - Visualization comparing top features
- `comparison_difference.png` - Visualization showing which features differ most
- `ablation_study_report.txt` - Detailed text report with statistics

## Usage

Run the analysis:
```bash
cd /home/michaellee/Desktop/alex/0.9429/src
conda activate alex

# Quick test with 200 samples per group
python shap_ablation_separate_training.py --max-samples 200

# Full analysis with 1000 samples per group (recommended)
python shap_ablation_separate_training.py --max-samples 1000

# Use all validation samples (takes longer)
python shap_ablation_separate_training.py
```

Options:
- `--max-samples N`: Limit SHAP computation to N samples per group (default: use all)
- `--no-smote`: Disable SMOTE balancing
- `--output-dir DIR`: Specify output directory (default: shap_ablation_separate_results)

## Why This Approach is Correct

The previous approach (`shap_ablation_student_vs_professional.py`) had a logical flaw:
- It used models trained on ALL data (including the identifying feature)
- Then tried to separate groups and compute SHAP values
- This doesn't truly isolate group-specific patterns

The new approach:
- Splits data FIRST by student status
- Trains completely separate models for each group
- Each model learns only from its respective group
- SHAP values reflect true group-specific feature importance
- The identifying feature is not used in modeling at all

This is a proper ablation study where we isolate the effect of being a student vs professional by training separate models.

## Technical Details

### Models Performance
**Students:**
- Training samples: 26,136 (after SMOTE balancing)
- Validation samples: 5,581
- Validation accuracy: ~76-77% (all three models)

**Professionals:**
- Training samples: 165,708 (after SMOTE balancing)
- Validation samples: 22,560
- Validation accuracy: ~96-97% (all three models)

Note: Higher accuracy for professionals due to lower base depression rate (8.18% vs 58.56%)

### SHAP Computation
- XGBoost: Has compatibility issue with SHAP TreeExplainer
- LightGBM: Works correctly
- CatBoost: Works correctly
- Final importance: Average of LightGBM and CatBoost SHAP values

## Interpretation

The results reveal meaningful differences:

1. **Age is much more critical for professionals** (1.6x more important), suggesting depression in working professionals is more age-related

2. **Academic pressure dominates for students**, which makes intuitive sense

3. **Work-life balance matters more for professionals**, while students are more affected by pure stress levels

4. **Sleep and diet** are important for both groups, but slightly more critical for professionals

5. **Job satisfaction** is obviously only relevant for professionals (students don't have this feature populated)

These findings can inform targeted interventions for each population.
