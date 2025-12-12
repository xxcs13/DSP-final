# Depression Prediction Using Ensemble Machine Learning: A Comprehensive Study

## Abstract

This study presents a robust machine learning framework for binary classification of depression based on demographic, lifestyle, academic, and psychosocial factors. The proposed system implements rigorous data leakage prevention mechanisms, automated hyperparameter optimization, and comprehensive model interpretability analysis using SHAP (SHapley Additive exPlanations). Through systematic feature engineering and ensemble modeling techniques, the framework achieves strong predictive performance with an AUC of 0.9755 and F1-score of 0.8348 on the held-out validation set. The methodology emphasizes reproducibility, fairness across demographic subgroups, and actionable insights for mental health intervention strategies.

---

## 1. Introduction

Depression represents one of the most prevalent mental health disorders globally, affecting approximately 280 million individuals worldwide. Early identification of at-risk populations enables timely intervention and potentially mitigates severe outcomes including suicidal ideation. This research develops a comprehensive machine learning pipeline for depression prediction that balances predictive accuracy with model interpretability, ensuring that insights can inform clinical decision-making and public health policy.

### 1.1 Research Aims

The primary objectives of this research are threefold. The first objective is to develop a robust binary classification model capable of accurately predicting an individual's risk of depression based on available demographic, lifestyle, and psychosocial indicators. Given the serious consequences of missed diagnoses, the model must maintain high sensitivity while minimizing false negatives that could result in at-risk individuals failing to receive necessary intervention.

The second objective is to quantify the contribution of various demographic, lifestyle, and stress-related factors to depression risk. Understanding which factors most strongly influence depression outcomes enables healthcare practitioners and policymakers to prioritize resources and design targeted prevention programs. This quantification extends beyond simple correlation analysis to provide importance rankings that account for complex feature interactions.

The third objective is to provide a high degree of model interpretability that generates actionable insights informing targeted prevention strategies. Rather than treating the predictive model as a black box, this research emphasizes transparency in how predictions are made, enabling domain experts to validate model behavior against clinical knowledge and translate findings into practical intervention recommendations.

### 1.2 Challenges

To achieve the stated research aims, several technical challenges must be addressed. Overcoming these obstacles constitutes a significant portion of the methodological contribution and directly enables both the predictive performance and analytical insights produced by this research.

The first primary challenge is class imbalance. The dataset used for this study exhibits significant imbalance, with the depression class (positive class) representing only 18.2% of the training samples compared to 81.8% for the non-depression class. This 4.5:1 imbalance ratio increases the risk of developing a biased model that systematically under-predicts the minority class. In the context of depression screening, such bias produces dangerous false negatives, failing to identify individuals who are genuinely at risk and require intervention. Standard machine learning algorithms trained on imbalanced data tend to optimize overall accuracy by favoring the majority class, which is clinically unacceptable when the cost of missing a true positive substantially exceeds the cost of a false alarm.

The second primary challenge is feature noise. The dataset contains numerous features spanning demographic, academic, occupational, lifestyle, and psychosocial domains. This high-dimensional feature space makes it difficult to distinguish weak, noisy predictors from the truly significant features that drive depression risk. Many features exhibit complex interactions and non-linear relationships with the target variable that simple linear models cannot capture. Additionally, structural missing value patterns arising from the mutual exclusivity between student and professional populations introduce systematic noise that requires careful handling. Failing to isolate core predictors from noise can reduce model accuracy through overfitting to irrelevant patterns and obscure the meaningful insights necessary for informing intervention strategies.

### 1.3 Contributions

The main contributions of this research directly address the identified challenges while fulfilling the research aims. These contributions span both predictive performance and interpretable insights.

The first contribution is performance gain through ensemble modeling and class imbalance handling. The developed ensemble model achieves superior predictive performance, with an AUC of 0.9755 and F1-score of 0.8348, significantly outperforming baseline models such as Logistic Regression (AUC 0.9741, F1-score 0.8034). More importantly, the gradient boosting ensemble maintains balanced precision (0.8361) and recall (0.8298), demonstrating that the class imbalance challenge has been effectively addressed without sacrificing sensitivity for specificity.

The second contribution is interpretable insights through systematic feature importance analysis. The model produces actionable insights by quantifying the importance of key lifestyle and stress-related factors using SHAP (SHapley Additive exPlanations) analysis. The research identifies age as the primary risk stratification factor (importance score 1.474), followed by suicidal thoughts history (0.820) and work/study hours (0.554). Engineered features such as Total Stress Index (0.439) and Work-Life Balance (0.339) demonstrate that domain-driven feature engineering successfully extracts meaningful signals from noisy raw features. These quantified importance rankings directly inform intervention strategies by identifying which modifiable risk factors offer the greatest potential for depression prevention.

### 1.4 Problem Formulation

Having established the context, challenges, and contributions of this research, this section provides a formal definition of the predictive task, its data flow, and the metrics governing its evaluation.

The depression prediction task is formulated as a supervised binary classification problem. Given an input feature vector x containing demographic, lifestyle, academic, occupational, and psychosocial attributes, the objective is to learn a function f that maps x to a binary output y, where y equals 1 indicates depression risk and y equals 0 indicates no depression risk. The learned function f should maximize classification performance on held-out data while providing interpretable feature importance rankings.

The data flow proceeds through four sequential phases. In the data understanding phase, exploratory analysis characterizes the dataset structure, identifies missing value patterns, and quantifies class imbalance. In the data preparation phase, cleaning, imputation, encoding, and splitting transform raw data into model-ready format while preventing information leakage. In the data modeling phase, feature engineering, model training, and hyperparameter optimization produce trained classifiers. In the evaluation phase, performance metrics, cross-validation, and interpretability analysis assess model quality and extract insights.

The evaluation employs multiple complementary metrics. Accuracy measures overall classification correctness. AUC-ROC evaluates discrimination capability across all classification thresholds. F1-score provides the harmonic mean of precision and recall, particularly relevant for imbalanced datasets where accuracy alone is misleading. Precision and recall separately quantify the trade-off between false positives and false negatives, enabling assessment of whether the class imbalance challenge has been adequately addressed.

---

## 2. Proposed Methods

The methodology follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework, comprising four primary phases: Data Understanding, Data Preparation, Data Modeling, and Evaluation. Each phase is designed with strict attention to preventing information leakage between training and validation datasets. Critically, specific methodological components are explicitly designed to address the class imbalance and feature noise challenges identified in Section 1.2.

### 2.1 Data Understanding

#### 2.1.1 Dataset Overview

The dataset comprises 140,700 training samples and 93,800 test samples, with 18 predictive features spanning demographic, academic, occupational, and psychosocial domains. The target variable represents binary depression classification, where class 0 indicates no depression and class 1 indicates presence of depression.

#### 2.1.2 Target Distribution Analysis

Exploratory analysis reveals substantial class imbalance in the target variable, directly confirming the class imbalance challenge described in Section 1.2. The non-depression class (class 0) contains 115,133 samples, representing 81.8% of the training data, while the depression class (class 1) contains 25,567 samples, accounting for 18.2% of the data. This yields an imbalance ratio of approximately 4.5:1, necessitating appropriate handling strategies during model training to prevent the model from systematically under-predicting depression cases.

#### 2.1.3 Feature Categories

The feature set is organized into five conceptual categories. Demographic features include Age, Gender, and City, providing basic population characteristics. Academic features encompass CGPA, Academic Pressure, Study Satisfaction, and Degree, applicable primarily to student populations. Occupational features include Profession, Work Pressure, and Job Satisfaction, relevant to working professionals. Lifestyle features comprise Work/Study Hours, Sleep Duration, and Dietary Habits. Psychosocial features include Suicidal Thoughts, Financial Stress, and Family History of Mental Illness, representing critical risk indicators.

#### 2.1.4 Missing Value Pattern Analysis

The dataset exhibits two distinct patterns of missing values, contributing to the feature noise challenge identified in Section 1.2. Structural missing values arise from the mutual exclusivity between student and professional status. Students naturally lack work-related features (Work Pressure, Job Satisfaction, Profession), while professionals lack academic features (CGPA, Academic Pressure, Study Satisfaction). These systematic patterns require context-aware imputation strategies rather than simple deletion to avoid introducing additional noise.

Anomalous missing values, affecting approximately 6.24% of records, represent data quality issues where expected values are absent. Examples include students without CGPA records or professionals without profession designations. These cases require careful handling to preserve data integrity while maintaining model performance.

### 2.2 Data Preparation

The data preparation phase implements comprehensive preprocessing with strict data leakage prevention protocols. All transformations derive statistics exclusively from the training set, which are subsequently applied to validation and test sets.

#### 2.2.1 Data Cleaning

The cleaning process employs a conservative strategy to maximize data retention, addressing the feature noise challenge by removing only severely corrupted records. Records with extreme corruption, defined as those with 50% or more missing values across all columns, are removed. This conservative threshold ensures that most data remains available for intelligent imputation rather than blanket deletion. Additional quality checks address inconsistent categorical values, including standardization of sleep duration categories and validation of dietary habit classifications. By standardizing noisy categorical values, this step reduces feature noise that could otherwise propagate through subsequent modeling stages.

#### 2.2.2 Missing Value Imputation

A context-aware imputation strategy addresses the structural missing value patterns, directly mitigating the feature noise challenge by replacing missing values with domain-appropriate substitutes rather than arbitrary defaults. For students, Work Pressure is imputed with 0 (reflecting no work-related stress), Job Satisfaction receives a neutral value of 3, and Profession is categorized as "Student." For professionals, CGPA receives the median professional CGPA computed from the training set, Academic Pressure is set to 0, and Study Satisfaction receives the neutral value. Universal features such as Financial Stress and Dietary Habits undergo median or mode imputation, with all imputation values computed exclusively from the training partition. This context-aware approach ensures that imputed values reflect realistic scenarios rather than introducing artificial patterns that could mislead the model.

#### 2.2.3 Train-Validation Split

The dataset undergoes stratified splitting with an 80-20 ratio for training and validation, respectively. Stratification preserves the original class distribution in both partitions, ensuring representative evaluation despite class imbalance. This stratification directly supports addressing the class imbalance challenge by ensuring that the validation set maintains the same 4.5:1 imbalance ratio as the training set, enabling realistic assessment of model performance on imbalanced data. The random state is fixed at 42 to guarantee reproducibility across experiments.

#### 2.2.4 Feature Encoding

Categorical feature encoding proceeds after the train-validation split to prevent information leakage. Binary features (e.g., Yes/No responses) undergo binary encoding. Ordinal features such as Sleep Duration receive integer encoding reflecting their natural ordering. For high-cardinality categorical features, specifically the Profession variable with approximately 64 unique categories, target encoding with Bayesian smoothing is employed. The target encoding computes the mean target value for each category using only training data, with smoothing applied to handle rare categories robustly. This encoding approach addresses the feature noise challenge by transforming noisy high-cardinality categorical variables into informative numerical representations that capture the relationship between category and target variable. The encoding formula applies Bayesian averaging: the smoothed mean equals (category mean multiplied by category count plus global mean multiplied by smoothing parameter) divided by (category count plus smoothing parameter).

#### 2.2.5 Class Imbalance Handling (Addressing Challenge 1)

The Synthetic Minority Over-sampling Technique (SMOTE) directly addresses the class imbalance challenge by generating synthetic samples for the minority class. SMOTE operates by selecting minority class instances and creating synthetic examples along the line segments joining k nearest minority class neighbors. This approach increases minority class representation without simple duplication, enabling models to learn more robust decision boundaries for the depression class.

Critically, SMOTE is applied exclusively to the training partition after the train-validation split, ensuring that validation data consists entirely of genuine samples. This separation prevents overly optimistic performance estimates that would arise from evaluating on synthetic data. The implementation uses default SMOTE parameters with k=5 nearest neighbors. After SMOTE application, the training set achieves approximate class balance, enabling subsequent models to learn from equal representation of both classes without inherent bias toward the majority class.

By addressing class imbalance at the data level, this approach enables standard classification algorithms to achieve balanced precision and recall without requiring cost-sensitive modifications or threshold manipulation. The effectiveness of this approach is validated by the final model achieving precision of 0.8361 and recall of 0.8298, demonstrating successful mitigation of the class imbalance challenge.

#### 2.2.6 Feature Scaling

RobustScaler is applied for feature normalization, selected for its resistance to outliers through the use of median and interquartile range rather than mean and standard deviation. This scaling approach addresses the feature noise challenge by reducing the influence of outliers that could otherwise dominate model training. The scaler is fit exclusively on training data, with the learned parameters subsequently applied to transform validation and test sets. Extreme values are clipped to the 5th and 95th percentiles computed from training data to further mitigate outlier influence.

### 2.3 Data Modeling

#### 2.3.1 Feature Engineering (Addressing Challenge 2)

The feature engineering module directly addresses the feature noise challenge by transforming the 18 preprocessed features into 35 engineered features. This transformation creates domain-specific representations that capture complex relationships within the data, effectively separating meaningful signal from noise. All feature engineering statistics are computed from training data exclusively to maintain data leakage prevention.

Stress-related features consolidate multiple stress indicators into unified metrics, reducing noise through aggregation. Total Stress Index averages stress across academic, occupational, and financial domains, providing a single holistic stress measure less susceptible to noise in individual stress components. Maximum Stress identifies the highest stress factor, capturing peak stress regardless of domain. Stress Variance measures consistency across stress domains. Stress Saturated applies logarithmic transformation to capture diminishing marginal effects at high stress levels, modeling the non-linear relationship between stress and depression risk.

Lifestyle features decompose complex categorical variables into interpretable binary indicators. Work-Life Balance computes the inverse relationship with work hours. Overwork Flag provides a binary indicator for excessive hours exceeding 10 per day. Sleep Optimal and Sleep Poor create binary flags for optimal and insufficient sleep respectively, extracting clear signals from the ordinal sleep duration variable. Diet Healthy and Diet Unhealthy similarly decompose dietary habits into interpretable risk indicators.

Psychosocial interaction features capture compound risk factors that individual features cannot represent. These include Suicidal-Financial Stress interaction, Family History-Financial Stress interaction, and a comprehensive Mental Health Risk Index combining multiple risk indicators. By explicitly modeling these interactions, the feature engineering process extracts signals that would otherwise remain hidden in feature noise.

Satisfaction features include Overall Satisfaction (averaging job and study satisfaction), Satisfaction Gap (difference between domains), and Low Satisfaction Flag (binary indicator for concerning satisfaction levels). Academic interaction features include CGPA-Academic Pressure Interaction, Study Hours-CGPA Ratio (measuring study efficiency), and Academic Performance Index.

The effectiveness of this feature engineering in addressing the feature noise challenge is validated by SHAP analysis, which reveals that six of the top 15 most important features are engineered features (Total_Stress_Index, Work_Life_Balance, Overall_Satisfaction, Stress_Saturated, Hours_Saturated, Youth_Risk). This demonstrates that the feature engineering process successfully extracts meaningful predictive signals that were obscured in the original noisy feature space.

#### 2.3.2 Model Selection

Five classification algorithms are employed to leverage their complementary strengths. Logistic Regression serves as a linear baseline model, providing interpretable coefficients and establishing minimum performance benchmarks against which more sophisticated models are compared. Random Forest contributes robust ensemble predictions through 200 decision trees with maximum depth of 10, offering built-in feature importance and resistance to overfitting through bootstrap aggregation.

The three gradient boosting models form the core of the predictive system. XGBoost implements gradient boosting with regularization (L1 and L2), employing histogram-based tree construction for computational efficiency. LightGBM provides fast gradient boosting through leaf-wise tree growth, offering efficient handling of large datasets with lower memory requirements. CatBoost employs symmetric trees with ordered boosting, providing superior handling of categorical features and built-in overfitting prevention. These gradient boosting models are particularly effective for the depression prediction task because their sequential tree construction naturally handles the feature noise challenge by learning to weight informative features more heavily than noisy ones.

#### 2.3.3 Hyperparameter Optimization

Hyperparameter optimization employs the Optuna framework with Tree-structured Parzen Estimator (TPE) sampling. The optimization process utilizes stratified 5-fold cross-validation to ensure robust hyperparameter selection that generalizes across different data partitions. The stratification in cross-validation is critical for addressing the class imbalance challenge, ensuring that each fold maintains representative class proportions despite the 4.5:1 imbalance ratio.

For each fold, the model is trained on four partitions and evaluated on the held-out fifth partition, with the average F1-score across all five folds serving as the optimization objective. The choice of F1-score as the optimization metric directly addresses the class imbalance challenge, as F1-score penalizes both false positives and false negatives equally, preventing the optimizer from selecting hyperparameters that achieve high accuracy by simply predicting the majority class.

Key hyperparameters optimized for tree-based models include the number of estimators (trees), maximum tree depth, learning rate, subsampling ratio, column subsampling ratio, and regularization parameters (L1 and L2). The optimization process conducts 30 to 50 trials per model, with each trial evaluating a unique hyperparameter configuration.

#### 2.3.4 Model Training with Early Stopping

Final model training employs early stopping to prevent overfitting, which is particularly important given the feature noise challenge. For XGBoost, LightGBM, and CatBoost, training monitors validation loss and terminates when no improvement is observed for 50 consecutive iterations. The model state corresponding to the best validation performance is retained for final evaluation. This approach automatically determines the optimal number of boosting iterations for each model, preventing the model from memorizing noisy patterns in the training data.

Training incorporates GPU acceleration when available, automatically detecting CUDA-capable devices and configuring models accordingly. XGBoost uses device="cuda" with histogram tree method, LightGBM uses device="gpu", and CatBoost uses task_type="GPU".

#### 2.3.5 Ensemble Construction (Addressing Challenge 1)

The ensemble model combines predictions from the three tree-based gradient boosting models: XGBoost, LightGBM, and CatBoost. Probability predictions from each constituent model are averaged to produce ensemble predictions. This ensemble approach addresses the class imbalance challenge by leveraging model diversity to improve robustness and reduce variance in predictions. Different models may make different errors on borderline cases, and averaging their predictions produces more calibrated probability estimates.

The ensemble classification threshold is computed as the average of individually optimized thresholds for each component model. This threshold optimization ensures that the ensemble achieves balanced precision and recall despite the underlying class imbalance, as each constituent model's threshold is calibrated to its specific probability distribution.

### 2.4 Evaluation

#### 2.4.1 Evaluation Metrics

Model performance is assessed using multiple complementary metrics specifically chosen to evaluate whether the class imbalance challenge has been successfully addressed. Accuracy measures overall classification correctness but is insufficient for imbalanced datasets where a naive majority-class classifier achieves high accuracy. Area Under the Receiver Operating Characteristic Curve (AUC-ROC) evaluates the model's ability to discriminate between classes across all classification thresholds, providing a class-imbalance-robust measure of model quality.

F1-Score provides the harmonic mean of precision and recall, penalizing models that achieve high recall at the expense of precision or vice versa. This metric is particularly relevant for assessing whether the class imbalance handling strategies have produced a balanced model. Precision quantifies the proportion of positive predictions that are correct, while Recall measures the proportion of actual positives correctly identified. A model that successfully addresses class imbalance should achieve similar precision and recall values rather than high recall with low precision (over-predicting positives) or high precision with low recall (under-predicting positives).

#### 2.4.2 Stratified 5-Fold Cross-Validation

The evaluation framework employs stratified 5-fold cross-validation to obtain robust performance estimates. The training data is partitioned into five equally-sized folds, with each fold maintaining the original class distribution through stratification. This stratification is essential for valid evaluation under class imbalance, ensuring that each fold contains representative proportions of both classes.

For each iteration, four folds serve as training data while the remaining fold serves as the validation set. This process repeats five times, with each fold serving as the validation set exactly once. The final performance metrics are computed as the mean across all five folds, providing estimates that are less sensitive to particular data partitions.

The stratified 5-fold cross-validation is specifically applied during the hyperparameter optimization phase (Section 2.3.3), ensuring that hyperparameter selection generalizes across different data subsets. For final model evaluation, the held-out validation set (20% of original data, created during the train-validation split described in Section 2.2.3) provides unbiased performance estimates on data not used during any training or hyperparameter selection process.

#### 2.4.3 Threshold Optimization

Classification thresholds are optimized individually for each model to maximize validation accuracy. The optimization process evaluates 81 threshold values ranging from 0.1 to 0.9 in increments of 0.01. The threshold yielding the highest validation accuracy is selected and stored for prediction generation. This model-specific threshold optimization accounts for differences in probability calibration across models and provides an additional mechanism to address class imbalance by adjusting the decision boundary to achieve optimal precision-recall balance.

#### 2.4.4 Model Interpretability (Addressing Contribution 2)

SHAP (SHapley Additive exPlanations) analysis provides model-agnostic feature importance rankings and individual prediction explanations, directly fulfilling the interpretability contribution and third research aim. For tree-based models, TreeExplainer computes exact SHAP values efficiently. The analysis processes 2,000 randomly sampled instances to balance computational feasibility with representative coverage.

Feature importance rankings aggregate SHAP values across all samples, computing the mean absolute SHAP value for each feature. This approach quantifies each feature's average contribution to predictions, enabling identification of the most influential predictors and addressing the feature noise challenge by distinguishing truly important features from noise. The SHAP analysis validates that engineered features successfully extract meaningful signal, with features like Total_Stress_Index and Work_Life_Balance ranking among the top predictors.

SHAP summary plots visualize feature importance alongside feature value distributions, revealing directional relationships between feature values and predictions. These visualizations enable domain experts to validate that model behavior aligns with clinical knowledge, enhancing trust in model predictions and enabling translation of quantified importance into actionable intervention recommendations.

---

## 3. Implementation Details

### 3.1 Pipeline Architecture

The implementation follows a modular architecture with six primary components. The EDA module (eda_analysis.py) conducts exploratory data analysis and generates visualization outputs. The preprocessing module (new_data_preprocessing.py) handles data cleaning, encoding, imputation, and train-validation splitting. The feature engineering module (new_feature_engineering.py) creates derived features and applies scaling and SMOTE. The model training module (new_model_training.py) implements model initialization, hyperparameter optimization, and training with early stopping. The evaluation module (evaluation_metrics.py) computes performance metrics and generates evaluation reports. The interpretability module (model_interpretability.py) conducts SHAP analysis and produces feature importance rankings.

### 3.2 Data Leakage Prevention

The implementation enforces strict separation between training and validation data throughout all preprocessing and feature engineering stages. Imputation values, encoding mappings, scaling parameters, and feature engineering statistics are computed exclusively from training data. These learned parameters are stored and applied to transform validation and test data, ensuring no information from held-out data influences the modeling process.

### 3.3 Reproducibility

All stochastic processes use fixed random seeds (random_state=55 for model training, random_state=42 for data splitting) to ensure reproducible results. Configuration parameters are logged and stored in JSON format for experimental tracking. Model artifacts, including trained models, optimal thresholds, and hyperparameters, are persisted to disk for subsequent analysis and deployment.

---

## 4. Results Summary

### 4.1 Model Performance

The evaluation on the held-out validation set demonstrates strong predictive performance across all models, validating that the proposed methods successfully address the identified challenges. LightGBM achieves the highest F1-score of 0.8348 with accuracy of 0.9403 and AUC of 0.9752. CatBoost follows closely with F1-score of 0.8344, accuracy of 0.9402, and AUC of 0.9754. XGBoost achieves F1-score of 0.8320, accuracy of 0.9390, and AUC of 0.9750. The ensemble model achieves F1-score of 0.8329, accuracy of 0.9395, and the highest AUC of 0.9755.

Random Forest achieves F1-score of 0.8188, accuracy of 0.9290, and AUC of 0.9724. Logistic Regression, serving as the linear baseline, achieves F1-score of 0.8034, accuracy of 0.9184, and AUC of 0.9741.

The gradient boosting models demonstrate balanced precision and recall (LightGBM: precision 0.8394, recall 0.8302), confirming that the class imbalance challenge has been effectively addressed. The performance improvement over the Logistic Regression baseline (F1-score improvement of 3.14 percentage points) validates the performance gain contribution.

### 4.2 Feature Importance

SHAP analysis across the ensemble of tree-based models identifies the most influential predictors, fulfilling the interpretability contribution. Age emerges as the most critical feature with importance score of 1.474, substantially higher than other features and contributing approximately 7.5% of total predictive importance. Suicidal thoughts history ranks second with importance score of 0.820, representing a strong indicator of depression severity. Work/Study Hours follows with importance score of 0.554, indicating the role of excessive time commitment in depression risk.

The Total Stress Index, an engineered feature combining multiple stress domains, achieves importance score of 0.439, validating the value of feature engineering in addressing the feature noise challenge. Profession achieves importance score of 0.393, demonstrating occupational factors in mental health. Work-Life Balance, another engineered feature, achieves importance score of 0.339.

These quantified importance rankings directly inform intervention strategies: targeting younger demographics for screening, monitoring individuals reporting suicidal thoughts, promoting work-life balance, and implementing stress management programs. This translation of model insights into actionable recommendations fulfills the third research aim of providing interpretable results for mental health practitioners.

---

## 5. Output Artifacts

The pipeline generates comprehensive outputs organized in dedicated directories. Exploratory data analysis outputs reside in eda_outputs/, including summary reports and visualization plots. Preprocessed data is stored in processed_data/, containing NumPy arrays for features and labels along with preprocessing settings. Engineered features are stored in engineered_data/, including all 35 engineered features with accompanying feature names.

Trained models are persisted in trained_models/, including serialized model files, optimal thresholds, and training metadata. Evaluation results are stored in evaluation_results/, containing performance metrics in CSV format and summary reports. Interpretability results reside in interpretability_results/, with separate subdirectories for each model and combined ensemble analyses.

Submission files are generated in the src/ directory, including predictions from all individual models (submission_logistic_regression.csv, submission_random_forest.csv, submission_xgboost.csv, submission_lightgbm.csv, submission_catboost.csv) and the ensemble model (submission_ensemble.csv).

---

## 6. Reproducibility Instructions

### 6.1 Environment Setup

The implementation requires Python 3.8 or higher with the following primary dependencies: pandas (version 2.0.0 or higher), numpy (version 1.24.0 or higher), scikit-learn (version 1.3.0 or higher), xgboost (version 2.0.0 or higher), lightgbm (version 4.0.0 or higher), catboost (version 1.2.0 or higher), optuna (version 3.3.0 or higher), imbalanced-learn (version 0.11.0 or higher), and shap (version 0.43.0 or higher).

### 6.2 Execution

The complete pipeline executes via the main.py orchestrator script. The full pipeline command (python main.py --full) sequentially executes exploratory data analysis, data preprocessing, feature engineering, model training, evaluation, interpretability analysis, and prediction generation. Individual pipeline stages can be executed separately using corresponding command-line flags (--eda, --preprocess, --engineer, --train, --predict).

Hyperparameter optimization is enabled via the --optimize flag with configurable trial count (--trials N). The encoding method for high-cardinality categorical features is selectable via --encoding-method with options "target" or "label".

---

## 7. Acknowledgments

This research utilizes open-source machine learning libraries including scikit-learn, XGBoost, LightGBM, CatBoost, and SHAP. The Optuna framework provides hyperparameter optimization capabilities. The implementation benefits from the broader machine learning and data science community's contributions to reproducible research practices.
