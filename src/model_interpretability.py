"""
Model Interpretability Module using SHAP

Provides actionable insights through:
1. SHAP feature importance
2. Feature importance rankings
3. SHAP summary plots
4. SHAP dependence plots
5. Individual prediction explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelInterpreter:
    def __init__(self, model, X_data, feature_names, output_dir='interpretability_results'):
        """
        Initialize model interpreter
        
        Args:
            model: Trained model
            X_data: Feature data (sample for SHAP computation)
            feature_names: List of feature names
            output_dir: Directory to save results
        """
        self.model = model
        self.X_data = X_data
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.explainer = None
        self.shap_values = None
        
    def compute_shap_values(self, max_samples=2000):
        """
        Compute SHAP values for the model
        
        Args:
            max_samples: Maximum number of samples to use for SHAP computation
        """
        print(f"Computing SHAP values (using {max_samples} samples)...")
        
        # Sample data if too large
        if len(self.X_data) > max_samples:
            sample_indices = np.random.choice(len(self.X_data), max_samples, replace=False)
            X_sample = self.X_data[sample_indices]
        else:
            X_sample = self.X_data
        
        # Create SHAP explainer
        try:
            # Try TreeExplainer for tree-based models (faster)
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)
            print("Using TreeExplainer")
        except:
            # Fall back to KernelExplainer for other models
            print("TreeExplainer not available, using KernelExplainer (slower)...")
            background = shap.kmeans(self.X_data, 50)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multi-output SHAP values (for binary classification)
        if isinstance(self.shap_values, list):
            # Use positive class (index 1) for binary classification
            self.shap_values = self.shap_values[1]
        
        # Convert to numpy array and ensure correct shape
        self.shap_values = np.array(self.shap_values)
        
        # For binary classification, some models return shape (n_samples, n_features, 2)
        # We want shape (n_samples, n_features)
        if len(self.shap_values.shape) == 3:
            # Take the positive class predictions (last dimension, index 1)
            self.shap_values = self.shap_values[:, :, 1]
        
        print(f"SHAP values computed successfully! Shape: {self.shap_values.shape}")
        
        return self.shap_values
        
    def get_feature_importance(self):
        """
        Get feature importance from SHAP values
        
        Returns:
            DataFrame with feature importance rankings
        """
        if self.shap_values is None:
            print("SHAP values not computed yet. Call compute_shap_values() first.")
            return None
        
        # Calculate mean absolute SHAP values
        # Handle potential multi-dimensional arrays
        shap_array = np.array(self.shap_values)
        
        # If 3D or higher dimensional, reduce to 2D (samples x features)
        while len(shap_array.shape) > 2:
            shap_array = shap_array[..., 0]  # Take first element of last dimension
        
        mean_shap = np.abs(shap_array).mean(axis=0)
        
        # Ensure mean_shap is 1-dimensional
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.flatten()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': mean_shap
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
        
    def plot_feature_importance(self, top_n=20, save_name='feature_importance.png'):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to display
            save_name: Filename to save plot
        """
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            return
        
        plt.figure(figsize=(10, max(8, top_n * 0.3)))
        
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {self.output_dir / save_name}")
        
    def plot_shap_summary(self, max_display=20, save_name='shap_summary.png'):
        """
        Create SHAP summary plot
        
        Args:
            max_display: Maximum number of features to display
            save_name: Filename to save plot
        """
        if self.shap_values is None:
            print("SHAP values not computed yet. Call compute_shap_values() first.")
            return
        
        # Sample data if used during SHAP computation
        X_sample = self.X_data[:len(self.shap_values)]
        
        plt.figure(figsize=(10, max(8, max_display * 0.3)))
        shap.summary_plot(self.shap_values, X_sample, 
                         feature_names=self.feature_names,
                         max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP summary plot saved to: {self.output_dir / save_name}")
        
    def plot_shap_bar(self, max_display=20, save_name='shap_bar.png'):
        """
        Create SHAP bar plot
        
        Args:
            max_display: Maximum number of features to display
            save_name: Filename to save plot
        """
        if self.shap_values is None:
            print("SHAP values not computed yet. Call compute_shap_values() first.")
            return
        
        # Sample data if used during SHAP computation
        X_sample = self.X_data[:len(self.shap_values)]
        
        plt.figure(figsize=(10, max(8, max_display * 0.3)))
        shap.summary_plot(self.shap_values, X_sample,
                         feature_names=self.feature_names,
                         plot_type='bar', max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP bar plot saved to: {self.output_dir / save_name}")
        
    def plot_shap_dependence(self, feature_name, save_name=None):
        """
        Create SHAP dependence plot for a specific feature
        
        Args:
            feature_name: Name of the feature
            save_name: Filename to save plot
        """
        if self.shap_values is None:
            print("SHAP values not computed yet. Call compute_shap_values() first.")
            return
        
        if feature_name not in self.feature_names:
            print(f"Feature {feature_name} not found")
            return
        
        feature_idx = self.feature_names.index(feature_name)
        X_sample = self.X_data[:len(self.shap_values)]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, self.shap_values, X_sample,
                            feature_names=self.feature_names, show=False)
        
        if save_name is None:
            safe_name = feature_name.replace(' ', '_').replace('/', '_').replace('?', '')
            save_name = f'shap_dependence_{safe_name}.png'
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP dependence plot saved to: {self.output_dir / save_name}")
        
    def generate_insights_report(self, top_n=10, save_name='interpretability_report.txt'):
        """
        Generate actionable insights report
        
        Args:
            top_n: Number of top features to include
            save_name: Filename to save report
        """
        importance_df = self.get_feature_importance()
        
        if importance_df is None:
            return
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MODEL INTERPRETABILITY REPORT")
        report_lines.append("Actionable Insights for Depression Intervention Strategies")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Use "FEATURE IMPORTANCE RANKING" instead of "TOP X" to be generic
        report_lines.append("FEATURE IMPORTANCE RANKING:")
        report_lines.append("-"*80)
        
        top_features = importance_df.head(top_n)
        
        for idx, row in top_features.iterrows():
            rank = row['Rank']
            feature = row['Feature']
            importance = row['Importance']
            report_lines.append(f"{rank}. {feature}")
            report_lines.append(f"   Importance Score: {importance:.4f}")
            
            # Add interpretation
            interpretation = self._interpret_feature(feature)
            if interpretation:
                report_lines.append(f"   Interpretation: {interpretation}")
            report_lines.append("")
        
        report_lines.append("="*80)
        report_lines.append("KEY INSIGHTS AND RECOMMENDATIONS:")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Generate recommendations based on top features
        recommendations = self._generate_recommendations(top_features)
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        report_text = "\n".join(report_lines)
        
        # Print report
        print("\n" + report_text)
        
        # Save report
        with open(self.output_dir / save_name, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {self.output_dir / save_name}")
        
        return importance_df
        
    def _interpret_feature(self, feature_name):
        """
        Provide interpretation for a feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Interpretation string
        """
        interpretations = {
            'Age': 'Younger individuals show higher depression risk',
            'Academic Pressure': 'Higher academic pressure strongly correlates with depression',
            'Work Pressure': 'Work-related stress is a significant depression factor',
            'Financial Stress': 'Financial difficulties contribute to depression risk',
            'Work/Study Hours': 'Excessive work/study hours increase depression likelihood',
            'Have you ever had suicidal thoughts ?': 'Strong indicator of depression severity',
            'Total_Stress_Index': 'Overall stress level is critical predictor',
            'Sleep Duration': 'Poor sleep quality/duration linked to depression',
            'Dietary Habits': 'Unhealthy diet associated with depression',
            'Family History of Mental Illness': 'Genetic/environmental risk factor',
            'Job Satisfaction': 'Low job satisfaction correlates with depression',
            'Study Satisfaction': 'Academic dissatisfaction indicates depression risk',
            'Overall_Satisfaction': 'Life satisfaction is protective factor',
            'Youth_Risk': 'Young age group at higher risk',
            'Overwork_Flag': 'Overworking is significant risk factor',
            'Total_Risk_Factors': 'Cumulative risk factors strongly predict depression'
        }
        
        return interpretations.get(feature_name, '')
        
    def _generate_recommendations(self, top_features_df):
        """
        Generate intervention recommendations based on top features
        
        Args:
            top_features_df: DataFrame of top features
            
        Returns:
            List of recommendations
        """
        recommendations = []
        top_features = top_features_df['Feature'].tolist()
        
        if any('Stress' in f for f in top_features):
            recommendations.append(
                "Implement stress management programs (counseling, mindfulness, relaxation techniques)"
            )
        
        if any('Pressure' in f for f in top_features):
            recommendations.append(
                "Reduce academic/work pressure through workload assessment and adjustment"
            )
        
        if 'Age' in top_features or 'Youth_Risk' in top_features:
            recommendations.append(
                "Target intervention programs for younger age groups (18-30 years)"
            )
        
        if 'Sleep Duration' in top_features:
            recommendations.append(
                "Promote healthy sleep habits and address sleep disorders"
            )
        
        if 'Financial Stress' in top_features:
            recommendations.append(
                "Provide financial counseling and support services"
            )
        
        if 'Work/Study Hours' in top_features or 'Overwork_Flag' in top_features:
            recommendations.append(
                "Monitor and limit excessive work/study hours, encourage work-life balance"
            )
        
        if any('Satisfaction' in f for f in top_features):
            recommendations.append(
                "Improve job/academic satisfaction through engagement and support"
            )
        
        if 'Have you ever had suicidal thoughts ?' in top_features:
            recommendations.append(
                "Immediate intervention for individuals with suicidal ideation"
            )
        
        if 'Family History of Mental Illness' in top_features:
            recommendations.append(
                "Screen individuals with family history of mental illness for early intervention"
            )
        
        if 'Dietary Habits' in top_features:
            recommendations.append(
                "Promote healthy nutrition and dietary counseling"
            )
        
        return recommendations
        
    def analyze_full_model(self, max_samples=2000, top_n=20):
        """
        Perform complete interpretability analysis
        
        Args:
            max_samples: Maximum samples for SHAP computation
            top_n: Number of top features to analyze
        """
        print("="*80)
        print("MODEL INTERPRETABILITY ANALYSIS")
        print("="*80 + "\n")
        
        # Compute SHAP values
        self.compute_shap_values(max_samples)
        
        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_feature_importance(top_n)
        self.plot_shap_summary(top_n)
        self.plot_shap_bar(top_n)
        
        # Generate report
        print("\nGenerating insights report...")
        self.generate_insights_report(top_n)
        
        print("\n" + "="*80)
        print("INTERPRETABILITY ANALYSIS COMPLETE")
        print("="*80)


def interpret_model(model_name=None, data_dir='engineered_data',
                   model_dir='trained_models', max_samples=2000):
    """
    Complete interpretability pipeline for trained models
    
    Args:
        model_name: Name of the model(s) to interpret. Can be:
                   - None: interpret all models
                   - str: interpret single model
                   - list: interpret specified models
        data_dir: Directory containing data
        model_dir: Directory containing trained models
        max_samples: Maximum samples for SHAP computation
    """
    # Define all available models
    all_models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
    
    # Determine which models to interpret
    if model_name is None:
        models_to_interpret = all_models
    elif isinstance(model_name, str):
        models_to_interpret = [model_name]
    elif isinstance(model_name, list):
        models_to_interpret = model_name
    else:
        raise ValueError("model_name must be None, str, or list")
    
    # Check training metadata to determine which features to use
    import json
    metadata_path = Path(model_dir) / 'training_metadata.json'
    use_selected_features = False
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            use_selected_features = metadata.get('use_feature_selection', False)
        print(f"Training metadata found: using {'SELECTED' if use_selected_features else 'ALL'} features")
    else:
        print("Warning: No training metadata found. Using ALL features by default.")
    
    # Load data based on training metadata
    print("Loading data...")
    if use_selected_features:
        X_val = np.load(Path(data_dir) / 'X_val_selected.npy')
        with open(Path(data_dir) / 'feature_names_selected.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
    else:
        X_val = np.load(Path(data_dir) / 'X_val_eng.npy')
        with open(Path(data_dir) / 'feature_names_eng.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
    
    print(f"Data shape: {X_val.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Interpret each model
    interpreters = {}
    for current_model in models_to_interpret:
        print(f"\n{'='*80}")
        print(f"Interpreting model: {current_model}")
        print(f"{'='*80}")
        
        # Load model
        model_path = Path(model_dir) / f"{current_model}.pkl"
        if not model_path.exists():
            print(f"Warning: Model file not found: {model_path}")
            print(f"Skipping {current_model}")
            continue
            
        print(f"Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Create interpreter
        interpreter = ModelInterpreter(
            model=model,
            X_data=X_val,
            feature_names=feature_names,
            output_dir=f'interpretability_results/{current_model}'
        )
        
        # Run analysis (use number of features instead of fixed 20)
        n_features = len(feature_names)
        interpreter.analyze_full_model(max_samples=max_samples, top_n=n_features)
        
        interpreters[current_model] = interpreter
        print(f"Completed interpretation for {current_model}")
    
    # Generate ensemble reports if multiple models were interpreted
    if len(interpreters) > 1:
        print(f"\n{'='*80}")
        print("GENERATING ENSEMBLE INTERPRETABILITY REPORTS")
        print(f"{'='*80}")
        
        # 1. All models ensemble report
        print("\n1. Generating comprehensive report for ALL models...")
        generate_ensemble_interpretability_report(
            interpreters=interpreters,
            output_name='ensemble_all_models',
            output_dir='interpretability_results'
        )
        
        # 2. Tree models ensemble report (XGBoost, LightGBM, CatBoost)
        tree_models = ['xgboost', 'lightgbm', 'catboost']
        tree_interpreters = {k: v for k, v in interpreters.items() if k in tree_models}
        
        if len(tree_interpreters) >= 2:
            print("\n2. Generating report for TREE MODELS (XGBoost, LightGBM, CatBoost)...")
            generate_ensemble_interpretability_report(
                interpreters=tree_interpreters,
                output_name='ensemble_tree_models',
                output_dir='interpretability_results'
            )
        else:
            print(f"\nSkipping tree ensemble report (only {len(tree_interpreters)} tree models available)")
    
    return interpreters if len(interpreters) > 1 else interpreters.get(models_to_interpret[0])


def generate_ensemble_interpretability_report(interpreters, output_name='all_models',
                                              output_dir='interpretability_results'):
    """
    Generate ensemble interpretability report by averaging SHAP values across multiple models
    
    Args:
        interpreters: Dictionary of ModelInterpreter objects {model_name: interpreter}
        output_name: Name for the ensemble report (e.g., 'all_models' or 'tree_ensemble')
        output_dir: Base directory for interpretability results
    """
    print(f"\n{'='*80}")
    print(f"GENERATING ENSEMBLE INTERPRETABILITY REPORT: {output_name}")
    print(f"{'='*80}")
    
    if len(interpreters) == 0:
        print("No interpreters provided. Skipping ensemble report.")
        return
    
    model_names = list(interpreters.keys())
    print(f"Combining SHAP values from models: {model_names}")
    
    # Create output directory
    ensemble_output_dir = Path(output_dir) / output_name
    ensemble_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature names from first interpreter
    feature_names = list(interpreters.values())[0].feature_names
    n_features = len(feature_names)
    
    # Collect SHAP values from all models
    all_shap_values = []
    valid_models = []
    
    for model_name, interpreter in interpreters.items():
        if interpreter.shap_values is not None:
            # Ensure SHAP values are 2D (samples x features)
            shap_array = np.array(interpreter.shap_values)
            while len(shap_array.shape) > 2:
                shap_array = shap_array[..., 0]
            all_shap_values.append(shap_array)
            valid_models.append(model_name)
        else:
            print(f"Warning: No SHAP values for {model_name}, skipping")
    
    if len(all_shap_values) == 0:
        print("No valid SHAP values found. Cannot generate ensemble report.")
        return
    
    print(f"Valid models for ensemble: {valid_models}")
    
    # Average SHAP values across models
    # Each model may have different number of samples, so we average the importance
    ensemble_importance = np.zeros(n_features)
    
    for shap_vals in all_shap_values:
        # Calculate mean absolute SHAP for this model
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        ensemble_importance += mean_abs_shap
    
    # Average across models
    ensemble_importance = ensemble_importance / len(all_shap_values)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': ensemble_importance
    })
    
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    # Save importance to CSV
    importance_csv_path = ensemble_output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"Feature importance saved to: {importance_csv_path}")
    
    # Generate bar plot
    plt.figure(figsize=(10, max(8, n_features * 0.3)))
    plt.barh(range(len(importance_df)), importance_df['Importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Mean Absolute SHAP Value (Ensemble Average)')
    plt.title(f'Feature Importance - Ensemble of {len(valid_models)} Models')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = ensemble_output_dir / 'feature_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to: {plot_path}")
    
    # Generate text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("ENSEMBLE MODEL INTERPRETABILITY REPORT")
    report_lines.append("Actionable Insights for Depression Intervention Strategies")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Models included in ensemble: {', '.join(valid_models)}")
    report_lines.append(f"Number of models: {len(valid_models)}")
    report_lines.append("")
    report_lines.append("FEATURE IMPORTANCE RANKING:")
    report_lines.append("-"*80)
    
    # Get interpretation function from first interpreter
    first_interpreter = list(interpreters.values())[0]
    
    for idx, row in importance_df.iterrows():
        rank = row['Rank']
        feature = row['Feature']
        importance = row['Importance']
        report_lines.append(f"{rank}. {feature}")
        report_lines.append(f"   Importance Score: {importance:.4f}")
        
        # Add interpretation if available
        interpretation = first_interpreter._interpret_feature(feature)
        if interpretation:
            report_lines.append(f"   Interpretation: {interpretation}")
        report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("KEY INSIGHTS AND RECOMMENDATIONS:")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Generate recommendations based on top features
    top_features = importance_df.head(10)
    recommendations = first_interpreter._generate_recommendations(top_features)
    for rec in recommendations:
        report_lines.append(f"- {rec}")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("NOTE: This report combines SHAP values from multiple models")
    report_lines.append("to provide a robust feature importance ranking.")
    report_lines.append("="*80)
    
    # Save report
    report_path = ensemble_output_dir / 'interpretability_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Ensemble interpretability report saved to: {report_path}")
    print(f"\nEnsemble report generation complete for: {output_name}")
    
    return importance_df


if __name__ == "__main__":
    # Interpret all models
    print("Interpreting all models\n")
    interpreters = interpret_model(
        model_name=None,  # None means all models
        max_samples=2000
    )
    
    print("\nInterpretability analysis complete!")
    print("Check the 'interpretability_results/' directory for outputs.")
    print("Results generated for each model in separate subdirectories.")

