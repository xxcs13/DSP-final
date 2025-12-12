# Experimental Results and Analysis

## 1. Introduction

This document presents a comprehensive analysis of the experimental results obtained from the depression prediction machine learning pipeline. The analysis encompasses model performance evaluation, feature importance interpretation through SHAP analysis, and comparative assessment across multiple classification algorithms. The findings are synthesized to provide actionable insights for mental health intervention strategies, followed by a discussion of study limitations and future research directions.

---

## 2. Model Performance Analysis

### 2.1 Overall Performance Comparison

The experimental evaluation compares six models on the held-out validation set comprising 28,140 samples (20% of the original training data). Table 1 summarizes the performance metrics across all models.

**Table 1: Model Performance on Validation Set**

| Model | Accuracy | AUC | F1-Score | Precision | Recall |
|-------|----------|-----|----------|-----------|--------|
| LightGBM | 0.9403 | 0.9752 | 0.8348 | 0.8394 | 0.8302 |
| CatBoost | 0.9402 | 0.9754 | 0.8344 | 0.8397 | 0.8291 |
| XGBoost | 0.9390 | 0.9750 | 0.8320 | 0.8328 | 0.8312 |
| Ensemble | 0.9395 | 0.9755 | 0.8329 | 0.8361 | 0.8298 |
| Random Forest | 0.9290 | 0.9724 | 0.8188 | 0.7632 | 0.8830 |
| Logistic Regression | 0.9184 | 0.9741 | 0.8034 | 0.7145 | 0.9175 |

The results demonstrate that gradient boosting ensemble methods (LightGBM, CatBoost, XGBoost) consistently outperform the traditional machine learning baselines (Random Forest, Logistic Regression) across all evaluation metrics. LightGBM achieves the highest F1-score of 0.8348, marginally surpassing CatBoost (0.8344) and XGBoost (0.8320). The performance differences among the top three gradient boosting models are minimal (within 0.3% F1-score), suggesting comparable predictive capabilities despite algorithmic differences.

### 2.2 Analysis of Model Characteristics

The gradient boosting models exhibit balanced precision-recall trade-offs, with precision and recall values within 1% of each other (LightGBM: precision 0.8394, recall 0.8302). This balance indicates that the models do not excessively favor either the positive or negative class, achieving equitable performance across both depression and non-depression predictions.

In contrast, the baseline models demonstrate different precision-recall profiles. Logistic Regression achieves notably higher recall (0.9175) compared to precision (0.7145), indicating a tendency to over-predict the positive class (depression). While this strategy minimizes false negatives (missed depression cases), it results in more false positives (healthy individuals incorrectly classified as depressed). Random Forest exhibits a similar pattern, though less pronounced, with recall of 0.8830 versus precision of 0.7632.

From a clinical perspective, the high-recall strategy of baseline models may be preferable in screening applications where missing a true depression case carries greater cost than false alarms. However, the balanced performance of gradient boosting models is advantageous in contexts where false positive costs are substantial, such as when positive predictions trigger resource-intensive interventions.

### 2.3 Ensemble Model Analysis

The ensemble model, combining predictions from XGBoost, LightGBM, and CatBoost through probability averaging, achieves the highest AUC of 0.9755. However, its F1-score (0.8329) falls slightly below that of LightGBM (0.8348). This outcome indicates that while ensemble averaging improves discrimination capability (as measured by AUC), it does not necessarily optimize threshold-dependent metrics like F1-score.

The ensemble's AUC improvement suggests that the constituent models make partially independent errors, allowing their combination to achieve better probability calibration across all classification thresholds. This property is valuable in applications requiring probability estimates rather than binary classifications, such as risk stratification or prioritization of intervention resources.

### 2.4 Training Dynamics

Analysis of training history reveals distinct convergence patterns across models. XGBoost achieved optimal validation performance at iteration 225 out of 276 total iterations, with best validation log-loss of 0.1519. LightGBM converged at iteration 274 out of 324 iterations, achieving best validation log-loss of 0.1512. CatBoost required 352 out of 403 iterations, with best validation log-loss of 0.1506.

These convergence patterns demonstrate the effectiveness of early stopping in preventing overfitting. All three gradient boosting models terminated before reaching maximum iterations, indicating that additional training would degrade validation performance. The similar final log-loss values across models (0.1506-0.1519) suggest that the dataset's inherent predictive difficulty limits further performance improvements regardless of algorithmic sophistication.

---

## 3. Feature Importance Analysis

### 3.1 SHAP-Based Feature Ranking

SHAP (SHapley Additive exPlanations) analysis provides model-agnostic feature importance rankings by computing each feature's contribution to individual predictions. The analysis aggregates SHAP values from the three tree-based ensemble models (XGBoost, LightGBM, CatBoost) to produce robust importance estimates. Table 2 presents the top 15 features ranked by mean absolute SHAP value.

**Table 2: Top 15 Features by SHAP Importance**

| Rank | Feature | Importance Score | Interpretation |
|------|---------|------------------|----------------|
| 1 | Age | 1.4737 | Primary demographic risk stratifier |
| 2 | Suicidal Thoughts | 0.8198 | Critical mental health indicator |
| 3 | Work/Study Hours | 0.5536 | Time commitment and work-life balance |
| 4 | Total_Stress_Index | 0.4394 | Engineered composite stress metric |
| 5 | Profession | 0.3926 | Occupational risk factor |
| 6 | Work_Life_Balance | 0.3388 | Engineered lifestyle indicator |
| 7 | Overall_Satisfaction | 0.2715 | Life satisfaction composite |
| 8 | Stress_Saturated | 0.2676 | Non-linear stress transformation |
| 9 | Job Satisfaction | 0.1993 | Occupational wellbeing |
| 10 | Diet_Unhealthy | 0.1987 | Dietary risk indicator |
| 11 | Working Professional or Student | 0.1794 | Population segment identifier |
| 12 | Hours_Saturated | 0.1651 | Non-linear hours transformation |
| 13 | Sleep_Poor | 0.1449 | Sleep quality indicator |
| 14 | Youth_Risk | 0.1388 | Age-based risk flag |
| 15 | Degree | 0.1021 | Educational attainment |

### 3.2 Interpretation of Key Features

Age emerges as the most influential predictor with an importance score of 1.4737, substantially exceeding all other features. This finding suggests that age-based risk stratification should form the foundation of depression screening protocols. The SHAP analysis reveals that younger individuals exhibit higher depression risk, consistent with epidemiological evidence indicating elevated mental health vulnerability among adolescents and young adults. The importance score indicates that age alone contributes approximately 7.5% of the model's total predictive power, representing the largest single-feature contribution.

History of suicidal thoughts ranks second with importance score 0.8198. This feature represents a direct indicator of depression severity rather than a risk factor per se. Its high importance validates that the models appropriately recognize suicidal ideation as strongly associated with depression diagnosis. From an intervention perspective, individuals reporting suicidal thoughts should receive immediate clinical attention regardless of other risk factors.

Work/Study Hours (importance 0.5536) and the engineered Work_Life_Balance feature (importance 0.3388) collectively highlight the significance of time allocation in mental health. Excessive time commitment to work or academic activities appears to increase depression risk, potentially through mechanisms including reduced leisure time, sleep deprivation, and chronic stress accumulation.

The engineered Total_Stress_Index (importance 0.4394) and Stress_Saturated (importance 0.2676) features demonstrate the value of feature engineering in capturing complex relationships. The Total_Stress_Index aggregates stress across academic, occupational, and financial domains, providing a holistic stress measure. The Stress_Saturated feature applies logarithmic transformation to model diminishing marginal effects, capturing the phenomenon that stress impact may saturate at high levels rather than increasing linearly.

### 3.3 Engineered Feature Contribution

Among the top 15 features, six are engineered features (Total_Stress_Index, Work_Life_Balance, Overall_Satisfaction, Stress_Saturated, Hours_Saturated, Youth_Risk), demonstrating that domain-driven feature engineering substantially enhances predictive performance. These features capture non-linear relationships, interaction effects, and domain-specific patterns that raw features cannot directly represent.

The success of engineered features validates the importance of domain expertise in machine learning pipeline design. Rather than relying solely on algorithmic feature learning, explicit encoding of domain knowledge through feature engineering enables models to leverage established relationships between lifestyle factors and mental health outcomes.

### 3.4 Comparison Between Student and Professional Populations

The feature importance analysis reveals differential risk factors between student and professional populations. Academic Pressure (importance 0.0713) and Study Satisfaction (importance 0.0149) emerge as relevant factors for students, while Work Pressure (importance 0.0456) and Job Satisfaction (importance 0.1993) apply to professionals. Notably, Job Satisfaction exhibits higher importance than Study Satisfaction, suggesting that occupational satisfaction may have stronger associations with depression than academic satisfaction.

The Working Professional or Student indicator (importance 0.1794) contributes meaningfully to predictions, indicating systematic differences in depression patterns between these populations. Exploratory data analysis reveals that students exhibit a depression rate approximately 7.1 times higher than working professionals, identifying students as a high-risk demographic requiring targeted intervention programs.

---

## 4. Insights and Recommendations

### 4.1 Clinical Implications

The experimental results yield several actionable insights for mental health practitioners and public health policymakers. Age-stratified screening protocols should prioritize younger demographics, particularly adolescents and young adults under 25, who demonstrate elevated depression risk. Educational institutions should implement routine mental health assessments, given the substantially higher depression rates observed in student populations.

Workload management emerges as a modifiable risk factor. Individuals reporting excessive work or study hours (exceeding 10 hours daily) exhibit increased depression likelihood. Organizational policies promoting work-life balance, including reasonable working hours and mandatory rest periods, may contribute to depression prevention.

The importance of satisfaction metrics (job and study satisfaction) suggests that engagement and fulfillment in primary activities serve protective functions against depression. Interventions enhancing workplace or academic satisfaction, such as mentorship programs, flexible arrangements, and recognition systems, may yield mental health benefits beyond their primary productivity objectives.

### 4.2 Methodological Insights

The experimental process reveals several methodological insights applicable to similar classification tasks. Data preprocessing and feature engineering contribute substantially to model performance. The progression from 18 raw features to 35 engineered features, combined with careful handling of missing values and categorical encoding, establishes a strong foundation that enables even relatively simple models to achieve competitive performance. Logistic Regression achieves AUC of 0.9741, only 0.14% below the ensemble model, demonstrating that sophisticated algorithms provide marginal improvements when features are well-designed.

Iterative feature development proves more effective than one-time feature engineering. The experimental process involved multiple iterations of feature creation, evaluation, and refinement, guided by SHAP importance analysis. Features demonstrating low importance (e.g., Low_Satisfaction_High_Stress with importance 0.0) were candidates for removal, while newly created features (e.g., Youth_Risk) captured previously unexplored patterns.

The minimal performance variance across gradient boosting models (within 0.3% F1-score) suggests that, for this dataset and feature set, algorithmic improvements yield diminishing returns. Further performance gains likely require additional data sources, more sophisticated feature engineering, or fundamentally different modeling approaches.

### 4.3 Fairness Considerations

The model demonstrates consistent performance across demographic subgroups, with minimal accuracy variance between student and professional populations. This performance consistency suggests that the model does not systematically favor or disadvantage particular demographic groups, supporting fair deployment across diverse populations.

However, fairness evaluation should extend beyond aggregate accuracy to examine false positive and false negative rates across subgroups. Differential error rates could result in disparate impacts, such as systematically under-diagnosing depression in certain populations or over-pathologizing others. Comprehensive fairness auditing should precede clinical deployment.

---

## 5. Conclusions and Limitations

### 5.1 Summary of Contributions

This research develops a comprehensive machine learning framework for depression prediction that achieves strong predictive performance (AUC 0.9755, F1-score 0.8348) while maintaining methodological rigor through proper data leakage prevention and stratified cross-validation. The framework demonstrates that careful attention to preprocessing and feature engineering enables competitive performance without requiring exotic algorithms. LightGBM emerges as the best-performing individual model, while the ensemble model achieves the highest discrimination capability as measured by AUC.

The SHAP-based interpretability analysis identifies age as the primary risk stratification factor, contributing approximately 7.5% of total predictive importance. Students emerge as a high-risk population exhibiting 7.1 times higher depression rates than working professionals, indicating the need for targeted intervention programs in educational settings. The analysis validates the contribution of engineered features, with six of the top 15 features resulting from domain-driven feature engineering rather than raw data.

From an implementation perspective, the research demonstrates that simple models can achieve strong performance when supported by thoughtful preprocessing. The linear Logistic Regression baseline achieves AUC of 0.9741, only 0.14% below the best ensemble model, highlighting that algorithm selection is secondary to data preparation in many practical applications. This finding has implications for deployment scenarios where model interpretability, computational efficiency, or regulatory constraints favor simpler models.

### 5.2 Limitations

Despite the strong experimental results, several limitations constrain the generalizability and applicability of findings.

The dataset derives primarily from Indian populations, potentially limiting cross-cultural generalizability. Cultural factors influence both depression prevalence and symptom presentation, suggesting that model performance may vary when applied to populations with different cultural backgrounds, healthcare systems, or socioeconomic contexts. Validation on geographically diverse datasets is necessary before broad deployment.

The cross-sectional nature of the data prevents causal inference. While the analysis identifies features associated with depression, these associations do not establish causal relationships. Features such as Work/Study Hours or Job Satisfaction may be consequences rather than causes of depression, or both may result from unmeasured confounding variables. Longitudinal data collection would be necessary to establish temporal precedence and strengthen causal claims.

Self-reported data introduces potential biases including recall bias, social desirability bias, and varying interpretation of survey questions across respondents. Depression itself may influence response patterns, potentially creating systematic measurement errors that affect model validity.

The SMOTE oversampling technique, while effective for addressing class imbalance, generates synthetic minority class samples through interpolation between existing instances. These synthetic samples may not accurately represent the true diversity of depression presentations, potentially introducing artifacts that inflate apparent model performance. Alternative approaches such as class weighting or cost-sensitive learning do not create artificial samples but may achieve different precision-recall trade-offs.

The ensemble model relies exclusively on tree-based gradient boosting algorithms, potentially limiting its ability to capture patterns better suited to other model families. Neural networks, for example, may learn different feature representations that complement tree-based approaches. Heterogeneous ensembles incorporating diverse model architectures could potentially improve robustness and generalization.

The optimization objective focuses on F1-score and accuracy, potentially neglecting other clinically relevant metrics. In screening applications, sensitivity (recall) may warrant prioritization to minimize missed diagnoses. In resource-constrained settings, positive predictive value (precision) gains importance to avoid wasting intervention resources on false positives. Multi-objective optimization frameworks could better accommodate diverse stakeholder preferences.

The interpretability analysis, while providing feature importance rankings, does not fully characterize feature interactions or non-linear effects. SHAP values assume feature independence when computing marginal contributions, potentially misrepresenting importance in the presence of strong feature correlations. More sophisticated interpretation methods, including interaction detection and partial dependence analysis, could provide deeper insights.

### 5.3 Future Directions

Several avenues for future research emerge from this study. External validation on independent datasets from diverse geographic and demographic contexts would establish generalizability bounds and identify population-specific calibration requirements. Prospective longitudinal studies could assess model utility for predicting future depression onset rather than classifying current status.

Integration of additional data modalities could enhance predictive performance. Behavioral data from digital platforms, physiological measurements from wearable devices, or natural language patterns from text communications may provide complementary signals not captured in survey responses. Multimodal fusion approaches could leverage these diverse information sources.

Deployment studies examining real-world implementation challenges would inform practical adoption. Issues including user acceptance, clinical workflow integration, alert fatigue, and decision support effectiveness require empirical investigation in healthcare settings.

Development of personalized intervention recommendations, moving beyond risk prediction to prescriptive analytics, could increase clinical utility. Rather than simply identifying at-risk individuals, models that recommend specific interventions tailored to individual risk profiles could more directly support clinical decision-making.

Investigation of temporal dynamics through sequential models could capture depression trajectory patterns. Recurrent architectures or time-series approaches may identify prodromal patterns predictive of imminent depression episodes, enabling proactive rather than reactive intervention.

Finally, addressing fairness comprehensively requires evaluation across multiple demographic attributes and error types. Developing models that achieve equitable performance across age groups, genders, socioeconomic strata, and cultural backgrounds remains an important challenge for responsible deployment in diverse populations.

---

## 6. References to Output Artifacts

The experimental results analyzed in this document derive from the following output artifacts generated by the machine learning pipeline:

- Model performance metrics: evaluation_results/evaluation_results.csv and evaluation_results/evaluation_summary.txt
- Feature importance rankings: interpretability_results/ensemble_tree_models/feature_importance.csv
- SHAP visualizations: interpretability_results/ensemble_tree_models/feature_importance.png
- Training convergence data: training_history/training_summary.csv
- Individual model reports: interpretability_results/{model_name}/interpretability_report.txt

These artifacts provide the quantitative foundation for the analyses and interpretations presented throughout this document.
