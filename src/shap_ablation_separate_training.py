"""
SHAP Feature Importance Ablation Study: Students vs Working Professionals
Proper Implementation with Separate Training Pipelines

This script performs a proper ablation study by:
1. Loading RAW training data
2. Splitting data by student status FIRST (before any processing)
3. Applying separate preprocessing to each group
4. Applying separate feature engineering to each group
5. Training separate models (XGBoost, LightGBM, CatBoost) for each group
6. Computing SHAP values for each group's models
7. Averaging SHAP importance across the three models per group
8. Comparing feature importance between groups

This ensures that:
- The "Working Professional or Student" feature is NOT included in modeling
- Each group has its own trained models reflecting group-specific patterns
- Feature importance comparison is valid and interpretable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
from pathlib import Path
import warnings
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Import preprocessing and feature engineering
import sys
sys.path.append(str(Path(__file__).parent))
from new_data_preprocessing import DataPreprocessor
from new_feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')


class SHAPAblationSeparateTraining:
    def __init__(self, output_dir='shap_ablation_separate_results', 
                 random_state=42, use_smote=True):
        """
        Initialize SHAP ablation study with separate training
        
        Args:
            output_dir: Directory to save results
            random_state: Random seed for reproducibility
            use_smote: Whether to use SMOTE for balancing
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.use_smote = use_smote
        
        # Data containers
        self.df_students = None
        self.df_professionals = None
        
        # Processed data for students
        self.X_train_students = None
        self.X_val_students = None
        self.y_train_students = None
        self.y_val_students = None
        self.feature_names_students = None
        
        # Processed data for professionals
        self.X_train_professionals = None
        self.X_val_professionals = None
        self.y_train_professionals = None
        self.y_val_professionals = None
        self.feature_names_professionals = None
        
        # Models
        self.models_students = {}
        self.models_professionals = {}
        
        # SHAP values
        self.shap_values_students = {}
        self.shap_values_professionals = {}
        self.importance_students = None
        self.importance_professionals = None
        
    def load_and_split_data(self):
        """
        Load raw training data and split by student status
        """
        print("="*80)
        print("STEP 1: Loading and splitting raw data by student status")
        print("="*80)
        
        # Load raw training data
        df = pd.read_csv('data/train.csv')
        print(f"Total training samples: {len(df)}")
        
        # Check target distribution
        if 'Depression' in df.columns:
            print(f"Overall depression rate: {df['Depression'].mean():.2%}")
        
        # Split by student status (BEFORE any processing)
        if 'Working Professional or Student' not in df.columns:
            raise ValueError("Column 'Working Professional or Student' not found")
        
        self.df_students = df[df['Working Professional or Student'] == 'Student'].copy()
        self.df_professionals = df[df['Working Professional or Student'] == 'Working Professional'].copy()
        
        print(f"\nStudents: {len(self.df_students)} samples ({len(self.df_students)/len(df)*100:.2f}%)")
        if 'Depression' in self.df_students.columns:
            print(f"  Depression rate: {self.df_students['Depression'].mean():.2%}")
        
        print(f"Professionals: {len(self.df_professionals)} samples ({len(self.df_professionals)/len(df)*100:.2f}%)")
        if 'Depression' in self.df_professionals.columns:
            print(f"  Depression rate: {self.df_professionals['Depression'].mean():.2%}")
        
        print("\nSuccessfully split data into two groups")
        print("Note: 'Working Professional or Student' column will be kept for preprocessing context")
        
    def preprocess_group(self, df, group_name):
        """
        Apply preprocessing to a specific group
        
        Args:
            df: Raw dataframe for the group
            group_name: 'students' or 'professionals'
            
        Returns:
            X_train, X_val, y_train, y_val, feature_names
        """
        print(f"\n{'='*80}")
        print(f"STEP 2: Preprocessing {group_name}")
        print(f"{'='*80}")
        
        preprocessor = DataPreprocessor(random_state=self.random_state)
        
        # Use the existing preprocess_train method
        result = preprocessor.preprocess_train(
            df, 
            use_smote=self.use_smote,
            smote_method='smote',
            val_size=0.2,
            scaling_method='robust'
        )
        
        X_train = result['X_train']
        X_val = result['X_val']
        y_train = result['y_train']
        y_val = result['y_val']
        feature_names = preprocessor.feature_names
        
        # Remove 'Working Professional or Student' from features if present
        if 'Working Professional or Student' in feature_names:
            ws_idx = feature_names.index('Working Professional or Student')
            feature_names = [f for i, f in enumerate(feature_names) if i != ws_idx]
            X_train = np.delete(X_train, ws_idx, axis=1)
            X_val = np.delete(X_val, ws_idx, axis=1)
            print(f"\nRemoved 'Working Professional or Student' feature from modeling")
        
        print(f"\nFinal preprocessed data:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Features: {len(feature_names)}")
        
        return X_train, X_val, y_train, y_val, feature_names
    
    def engineer_features_group(self, X_train, X_val, feature_names, group_name):
        """
        Apply feature engineering to a specific group
        
        Args:
            X_train: Training features (numpy array)
            X_val: Validation features (numpy array)
            feature_names: List of feature names
            group_name: 'students' or 'professionals'
            
        Returns:
            X_train_eng, X_val_eng, feature_names_eng
        """
        print(f"\n{'='*80}")
        print(f"STEP 3: Feature Engineering for {group_name}")
        print(f"{'='*80}")
        
        # Import the engineer_features function
        from new_feature_engineering import engineer_features
        
        # Engineer training features
        print("Engineering training features...")
        X_train_eng, feature_names_eng, engineer = engineer_features(
            X_train, feature_names, is_training=True, engineer=None
        )
        
        # Engineer validation features
        print("Engineering validation features...")
        X_val_eng, _, _ = engineer_features(
            X_val, feature_names, is_training=False, engineer=engineer
        )
        
        print(f"\nFeatures after engineering: {len(feature_names_eng)}")
        print(f"New features created: {len(feature_names_eng) - len(feature_names)}")
        
        return X_train_eng, X_val_eng, feature_names_eng
    
    def train_models_group(self, X_train, y_train, X_val, y_val, group_name):
        """
        Train XGBoost, LightGBM, and CatBoost for a specific group
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            group_name: 'students' or 'professionals'
            
        Returns:
            Dictionary of trained models
        """
        print(f"\n{'='*80}")
        print(f"STEP 4: Training Models for {group_name}")
        print(f"{'='*80}")
        
        models = {}
        
        # XGBoost
        print("\nTraining XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss',
            early_stopping_rounds=20,
            verbosity=0
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models['xgboost'] = xgb
        
        train_score = xgb.score(X_train, y_train)
        val_score = xgb.score(X_val, y_val)
        print(f"  Training accuracy: {train_score:.4f}")
        print(f"  Validation accuracy: {val_score:.4f}")
        
        # LightGBM
        print("\nTraining LightGBM...")
        lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=-1
        )
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        models['lightgbm'] = lgbm
        
        train_score = lgbm.score(X_train, y_train)
        val_score = lgbm.score(X_val, y_val)
        print(f"  Training accuracy: {train_score:.4f}")
        print(f"  Validation accuracy: {val_score:.4f}")
        
        # CatBoost
        print("\nTraining CatBoost...")
        cat = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=False
        )
        cat.fit(X_train, y_train, eval_set=(X_val, y_val))
        models['catboost'] = cat
        
        train_score = cat.score(X_train, y_train)
        val_score = cat.score(X_val, y_val)
        print(f"  Training accuracy: {train_score:.4f}")
        print(f"  Validation accuracy: {val_score:.4f}")
        
        return models
    
    def compute_shap_values_group(self, models, X_val, feature_names, group_name, max_samples=None):
        """
        Compute SHAP values for all models of a specific group
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            feature_names: List of feature names
            group_name: 'students' or 'professionals'
            max_samples: Maximum samples to use for SHAP (None = use all)
            
        Returns:
            Dictionary of SHAP values per model
        """
        print(f"\n{'='*80}")
        print(f"STEP 5: Computing SHAP Values for {group_name}")
        print(f"{'='*80}")
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(X_val):
            indices = np.random.choice(len(X_val), max_samples, replace=False)
            X_val_sample = X_val[indices]
            print(f"Using {max_samples} samples for SHAP computation")
        else:
            X_val_sample = X_val
            print(f"Using all {len(X_val)} samples for SHAP computation")
        
        shap_values_dict = {}
        
        for model_name, model in models.items():
            print(f"\nComputing SHAP for {model_name}...")
            
            try:
                # Create explainer
                explainer = shap.TreeExplainer(model)
                
                # Compute SHAP values
                shap_values = explainer.shap_values(X_val_sample)
                
                # Handle binary classification (take positive class)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Store absolute mean SHAP values per feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_values_dict[model_name] = mean_abs_shap
                
                print(f"  Successfully computed SHAP values")
                print(f"  Shape: {shap_values.shape}")
                
            except Exception as e:
                print(f"  ERROR computing SHAP for {model_name}: {str(e)}")
                print(f"  This model will be excluded from analysis")
        
        if len(shap_values_dict) == 0:
            raise RuntimeError(f"Failed to compute SHAP values for all models in {group_name}")
        
        print(f"\nSuccessfully computed SHAP for {len(shap_values_dict)}/3 models")
        
        return shap_values_dict
    
    def average_shap_importance(self, shap_values_dict, feature_names):
        """
        Average SHAP importance across all models
        
        Args:
            shap_values_dict: Dictionary of SHAP values per model
            feature_names: List of feature names
            
        Returns:
            DataFrame with averaged importance
        """
        # Stack SHAP values from all models
        shap_arrays = [values for values in shap_values_dict.values()]
        
        # Average across models
        avg_importance = np.mean(shap_arrays, axis=0)
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        })
        
        # Sort by importance
        df_importance = df_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return df_importance
    
    def plot_comparison(self):
        """
        Generate comparison visualizations
        """
        print(f"\n{'='*80}")
        print(f"STEP 6: Generating Comparison Visualizations")
        print(f"{'='*80}")
        
        # Create comparison DataFrame
        df_students = self.importance_students.copy()
        df_students.columns = ['feature', 'importance_students']
        
        df_professionals = self.importance_professionals.copy()
        df_professionals.columns = ['feature', 'importance_professionals']
        
        # Merge
        df_comparison = pd.merge(df_students, df_professionals, on='feature', how='outer').fillna(0)
        df_comparison['difference'] = df_comparison['importance_students'] - df_comparison['importance_professionals']
        
        # Get top features from both groups
        top_features = set(self.importance_students.head(15)['feature'].tolist() + 
                          self.importance_professionals.head(15)['feature'].tolist())
        df_plot = df_comparison[df_comparison['feature'].isin(top_features)].copy()
        
        # Sort by average importance
        df_plot['avg_importance'] = (df_plot['importance_students'] + df_plot['importance_professionals']) / 2
        df_plot = df_plot.sort_values('avg_importance', ascending=True)
        
        # Plot 1: Side-by-side comparison
        fig, ax = plt.subplots(figsize=(12, 10))
        
        y_pos = np.arange(len(df_plot))
        width = 0.35
        
        ax.barh(y_pos - width/2, df_plot['importance_students'], width, 
                label='Students', alpha=0.8, color='#2ecc71')
        ax.barh(y_pos + width/2, df_plot['importance_professionals'], width,
                label='Professionals', alpha=0.8, color='#3498db')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['feature'])
        ax.set_xlabel('Average SHAP Importance', fontsize=12)
        ax.set_title('Feature Importance Comparison: Students vs Professionals', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_side_by_side.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved: comparison_side_by_side.png")
        
        # Plot 2: Difference plot
        df_diff = df_comparison.sort_values('difference', ascending=True).tail(20)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df_diff['difference']]
        ax.barh(range(len(df_diff)), df_diff['difference'], color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(df_diff)))
        ax.set_yticklabels(df_diff['feature'])
        ax.set_xlabel('Importance Difference (Students - Professionals)', fontsize=12)
        ax.set_title('Feature Importance Differences', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_difference.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved: comparison_difference.png")
        
        # Save comparison data
        df_comparison.to_csv(self.output_dir / 'feature_importance_comparison.csv', index=False)
        print("Saved: feature_importance_comparison.csv")
        
    def generate_report(self):
        """
        Generate detailed text report
        """
        print(f"\n{'='*80}")
        print(f"STEP 7: Generating Report")
        print(f"{'='*80}")
        
        report_path = self.output_dir / 'ablation_study_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SHAP ABLATION STUDY: STUDENTS VS WORKING PROFESSIONALS\n")
            f.write("Separate Training Pipeline Implementation\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Study Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Used: XGBoost, LightGBM, CatBoost\n")
            f.write(f"Methodology: Separate preprocessing, feature engineering, and training for each group\n\n")
            
            # Group statistics
            f.write("GROUP STATISTICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Students:\n")
            f.write(f"  Training samples: {len(self.y_train_students)}\n")
            f.write(f"  Validation samples: {len(self.y_val_students)}\n")
            f.write(f"  Features: {len(self.feature_names_students)}\n")
            f.write(f"  Depression rate (train): {self.y_train_students.mean():.2%}\n")
            f.write(f"  Depression rate (val): {self.y_val_students.mean():.2%}\n\n")
            
            f.write(f"Working Professionals:\n")
            f.write(f"  Training samples: {len(self.y_train_professionals)}\n")
            f.write(f"  Validation samples: {len(self.y_val_professionals)}\n")
            f.write(f"  Features: {len(self.feature_names_professionals)}\n")
            f.write(f"  Depression rate (train): {self.y_train_professionals.mean():.2%}\n")
            f.write(f"  Depression rate (val): {self.y_val_professionals.mean():.2%}\n\n")
            
            # Top features for students
            f.write("TOP 20 FEATURES FOR STUDENTS:\n")
            f.write("-"*80 + "\n")
            for idx, row in self.importance_students.head(20).iterrows():
                f.write(f"{idx+1:2d}. {row['feature']:<45s} {row['importance']:>10.6f}\n")
            f.write("\n")
            
            # Top features for professionals
            f.write("TOP 20 FEATURES FOR WORKING PROFESSIONALS:\n")
            f.write("-"*80 + "\n")
            for idx, row in self.importance_professionals.head(20).iterrows():
                f.write(f"{idx+1:2d}. {row['feature']:<45s} {row['importance']:>10.6f}\n")
            f.write("\n")
            
            # Key differences
            f.write("KEY DIFFERENCES:\n")
            f.write("-"*80 + "\n\n")
            
            # Merge and compute differences
            df_comp = pd.merge(
                self.importance_students[['feature', 'importance']].rename(columns={'importance': 'imp_students'}),
                self.importance_professionals[['feature', 'importance']].rename(columns={'importance': 'imp_professionals'}),
                on='feature'
            )
            df_comp['difference'] = df_comp['imp_students'] - df_comp['imp_professionals']
            
            # Features more important for students
            f.write("Features MORE important for Students:\n")
            top_student_features = df_comp.nlargest(10, 'difference')
            for _, row in top_student_features.iterrows():
                f.write(f"  - {row['feature']}: +{row['difference']:.6f}\n")
            
            f.write("\nFeatures MORE important for Professionals:\n")
            top_prof_features = df_comp.nsmallest(10, 'difference')
            for _, row in top_prof_features.iterrows():
                f.write(f"  - {row['feature']}: {row['difference']:.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Saved: ablation_study_report.txt")
        
    def run_analysis(self, max_samples=None):
        """
        Run complete analysis pipeline
        
        Args:
            max_samples: Maximum samples per group for SHAP computation (None = use all)
        """
        # Load and split data
        self.load_and_split_data()
        
        # Process students
        (self.X_train_students, self.X_val_students, 
         self.y_train_students, self.y_val_students, 
         feature_names_students) = self.preprocess_group(self.df_students, "students")
        
        (self.X_train_students, self.X_val_students, 
         self.feature_names_students) = self.engineer_features_group(
            self.X_train_students, self.X_val_students, 
            feature_names_students, "students"
        )
        
        self.models_students = self.train_models_group(
            self.X_train_students, self.y_train_students,
            self.X_val_students, self.y_val_students, "students"
        )
        
        self.shap_values_students = self.compute_shap_values_group(
            self.models_students, self.X_val_students,
            self.feature_names_students, "students", max_samples
        )
        
        self.importance_students = self.average_shap_importance(
            self.shap_values_students, self.feature_names_students
        )
        
        # Process professionals
        (self.X_train_professionals, self.X_val_professionals,
         self.y_train_professionals, self.y_val_professionals,
         feature_names_professionals) = self.preprocess_group(self.df_professionals, "professionals")
        
        (self.X_train_professionals, self.X_val_professionals,
         self.feature_names_professionals) = self.engineer_features_group(
            self.X_train_professionals, self.X_val_professionals,
            feature_names_professionals, "professionals"
        )
        
        self.models_professionals = self.train_models_group(
            self.X_train_professionals, self.y_train_professionals,
            self.X_val_professionals, self.y_val_professionals, "professionals"
        )
        
        self.shap_values_professionals = self.compute_shap_values_group(
            self.models_professionals, self.X_val_professionals,
            self.feature_names_professionals, "professionals", max_samples
        )
        
        self.importance_professionals = self.average_shap_importance(
            self.shap_values_professionals, self.feature_names_professionals
        )
        
        # Generate outputs
        self.importance_students.to_csv(
            self.output_dir / 'feature_importance_students.csv', index=False
        )
        self.importance_professionals.to_csv(
            self.output_dir / 'feature_importance_professionals.csv', index=False
        )
        
        self.plot_comparison()
        self.generate_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  - feature_importance_students.csv")
        print("  - feature_importance_professionals.csv")
        print("  - feature_importance_comparison.csv")
        print("  - comparison_side_by_side.png")
        print("  - comparison_difference.png")
        print("  - ablation_study_report.txt")


def main():
    parser = argparse.ArgumentParser(
        description='SHAP Ablation Study with Separate Training'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per group for SHAP computation (default: use all)'
    )
    parser.add_argument(
        '--no-smote',
        action='store_true',
        help='Disable SMOTE balancing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='shap_ablation_separate_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SHAP ABLATION STUDY: STUDENTS VS WORKING PROFESSIONALS")
    print("Separate Training Pipeline Implementation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Max samples per group: {args.max_samples if args.max_samples else 'all'}")
    print(f"  Use SMOTE: {not args.no_smote}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Run analysis
    study = SHAPAblationSeparateTraining(
        output_dir=args.output_dir,
        use_smote=not args.no_smote
    )
    
    study.run_analysis(max_samples=args.max_samples)


if __name__ == '__main__':
    main()
