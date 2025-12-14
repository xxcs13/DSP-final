"""
Main Pipeline Orchestrator for Depression Prediction

This is the main entry point that orchestrates the entire pipeline:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing (Phase 1: Cleaning, Encoding, Imputation - NO scaling/SMOTE)
3. Feature Engineering (includes Scaling and SMOTE - CORRECTED ORDER)
4. Model Training
5. Model Evaluation
6. Model Interpretability (SHAP)
7. Prediction Generation

CORRECTED PIPELINE ORDER:
- Feature Engineering is performed on ORIGINAL-SCALE data
- Scaling is applied AFTER feature engineering
- SMOTE is applied AFTER scaling
This ensures features are created with meaningful values and proper scale.
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from eda_analysis import DepressionEDA
from new_data_preprocessing import preprocess_pipeline
from new_feature_engineering import engineer_pipeline
from feature_selection import select_features
from new_model_training import train_all_models
from model_interpretability import interpret_model
from prediction_pipeline import generate_predictions, generate_ensemble_predictions


class DepressionPredictionPipeline:
    def __init__(self, config=None):
        """
        Initialize the complete pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
    def _default_config(self):
        """Get default configuration"""
        return {
            'train_path': 'data/train.csv',
            'test_path': 'data/test.csv',
            'use_smote': True,
            'smote_method': 'smote',
            'val_size': 0.2,
            'scaling_method': 'robust',
            'encoding_method': 'label',  # 'label' or 'target' for Profession encoding
            'use_feature_selection': False,
            'n_selected_features': 20,
            'optimize_hyperparams': False,
            'n_trials': 50,
            'best_model': 'catboost',
            'ensemble_models': ['xgboost', 'lightgbm', 'catboost'],
            'shap_samples': 2000
        }
        
    def run_eda(self):
        """Run exploratory data analysis"""
        print("\n" + "="*80)
        print("STEP 1: EXPLORATORY DATA ANALYSIS")
        print("="*80 + "\n")
        
        eda = DepressionEDA(
            train_path=self.config['train_path'],
            test_path=self.config['test_path']
        )
        eda.run_full_analysis()
        
        print("\nEDA complete! Check 'eda_outputs/' for results.")
        
    def run_preprocessing(self):
        """Run data preprocessing (Phase 1: before feature engineering)"""
        print("\n" + "="*80)
        print("STEP 2: DATA PREPROCESSING (Phase 1)")
        print("="*80 + "\n")
        print("This phase includes:")
        print("  - Data cleaning")
        print("  - Categorical encoding")
        print("  - Missing value imputation")
        print("  - Train/validation split")
        print("\n⚠️  Scaling and SMOTE will be applied in Feature Engineering step")
        print()
        
        result = preprocess_pipeline(
            train_path=self.config['train_path'],
            test_path=self.config['test_path'],
            output_dir='processed_data',
            use_smote=self.config['use_smote'],
            smote_method=self.config['smote_method'],
            val_size=self.config['val_size'],
            scaling_method=self.config['scaling_method'],
            encoding_method=self.config.get('encoding_method', 'target')
        )
        
        print("\nPreprocessing (Phase 1) complete! Check 'processed_data/' for results.")
        return result
        
    def run_feature_engineering(self):
        """Run feature engineering (includes Scaling and SMOTE)"""
        print("\n" + "="*80)
        print("STEP 3: FEATURE ENGINEERING (with Scaling & SMOTE)")
        print("="*80 + "\n")
        print("CORRECTED PIPELINE ORDER:")
        print("  1. Feature Engineering (on original-scale data)")
        print("  2. Scaling (fit on training only)")
        print("  3. SMOTE (on training only)")
        print()
        
        result = engineer_pipeline(
            input_dir='processed_data',
            output_dir='engineered_data'
        )
        
        print("\nFeature engineering complete! Check 'engineered_data/' for results.")
        return result
    
    def run_feature_selection(self):
        """Run feature selection using SHAP"""
        print("\n" + "="*80)
        print("STEP 3.5: FEATURE SELECTION")
        print("="*80 + "\n")
        
        selector = select_features(
            input_dir='engineered_data',
            output_dir='engineered_data',
            n_features=self.config['n_selected_features'],
            save_importance=True
        )
        
        print("\nFeature selection complete! Check 'engineered_data/' for selected features.")
        return selector
        
    def run_model_training(self):
        """Run model training"""
        print("\n" + "="*80)
        print("STEP 4: MODEL TRAINING")
        print("="*80 + "\n")
        
        import numpy as np
        import json
        from pathlib import Path
        
        # Check if feature selection was used
        use_selected_features = self.config['use_feature_selection'] and Path('engineered_data/X_train_selected.npy').exists()
        
        if use_selected_features:
            print("Loading SELECTED features...")
            X_train = np.load('engineered_data/X_train_selected.npy')
            X_val = np.load('engineered_data/X_val_selected.npy')
            print(f"Using {X_train.shape[1]} selected features\n")
        else:
            print("Loading ALL engineered features...")
            X_train = np.load('engineered_data/X_train_eng.npy')
            X_val = np.load('engineered_data/X_val_eng.npy')
            print(f"Using all {X_train.shape[1]} features\n")
        
        y_train = np.load('engineered_data/y_train.npy')
        y_val = np.load('engineered_data/y_val.npy')
        
        # Save metadata about feature selection usage
        model_dir = Path('trained_models')
        model_dir.mkdir(exist_ok=True)
        metadata = {
            'use_feature_selection': use_selected_features,
            'n_features': X_train.shape[1]
        }
        with open(model_dir / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved training metadata: feature_selection={use_selected_features}, n_features={X_train.shape[1]}\n")
        
        results, trainer, evaluator, optimal_thresholds, optimal_learning_rates, optimal_tree_counts = train_all_models(
            X_train, y_train, X_val, y_val,
            optimize=self.config['optimize_hyperparams'],
            n_trials=self.config['n_trials'],
            optimize_learning_rate=self.config.get('optimize_learning_rate', False),
            output_dir='trained_models'
        )
        
        print("\nModel training complete! Check 'trained_models/' and 'evaluation_results/' for results.")
        return results, trainer, evaluator, optimal_thresholds, optimal_learning_rates, optimal_tree_counts
        
    def run_interpretability(self):
        """Run model interpretability analysis"""
        print("\n" + "="*80)
        print("STEP 5: MODEL INTERPRETABILITY")
        print("="*80 + "\n")
        
        # Interpret all models instead of just one
        interpreters = interpret_model(
            model_name=None,  # None means all models
            data_dir='engineered_data',
            model_dir='trained_models',
            max_samples=self.config['shap_samples']
        )
        
        print(f"\nInterpretability analysis complete!")
        print(f"Check 'interpretability_results/' for results:")
        print(f"  - Individual model reports in separate subdirectories")
        print(f"  - ensemble_all_models/: Combined report for all 5 models")
        print(f"  - ensemble_tree_models/: Combined report for tree models (XGBoost, LightGBM, CatBoost)")
        return interpreters
        
    def run_prediction(self):
        """Generate predictions for all models"""
        print("\n" + "="*80)
        print("STEP 6: PREDICTION GENERATION")
        print("="*80 + "\n")
        
        # Define all individual models
        all_models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
        
        submissions = {}
        
        # Generate predictions for each individual model
        print("Generating individual model predictions...")
        print("-" * 80)
        for model_name in all_models:
            print(f"\n{model_name.upper()}:")
            submission = generate_predictions(
                model_name=model_name,
                data_dir='engineered_data',
                model_dir='trained_models',
                output_path=f"submission_{model_name}.csv"
            )
            submissions[model_name] = submission
            print(f"  Saved to: submission_{model_name}.csv")
        
        # Generate ensemble predictions
        print("\n" + "-" * 80)
        print("\nENSEMBLE (XGBoost + LightGBM + CatBoost):")
        submission_ensemble = generate_ensemble_predictions(
            model_names=self.config['ensemble_models'],
            data_dir='engineered_data',
            model_dir='trained_models',
            output_path='submission_ensemble.csv'
        )
        submissions['ensemble'] = submission_ensemble
        
        print("\n" + "="*80)
        print("PREDICTION GENERATION COMPLETE")
        print("="*80)
        print("\nGenerated submission files:")
        for model_name in all_models:
            print(f"  - submission_{model_name}.csv")
        print("  - submission_ensemble.csv")
        
        return submissions
        
    def run_full_pipeline(self, skip_eda=False):
        """
        Run the complete pipeline
        
        Args:
            skip_eda: Whether to skip EDA (if already done)
        """
        print("\n" + "="*80)
        print("DEPRESSION PREDICTION - COMPLETE PIPELINE")
        print("="*80)
        
        if not skip_eda:
            self.run_eda()
        else:
            print("\nSkipping EDA (already completed)")
        
        self.run_preprocessing()
        self.run_feature_engineering()
        
        # Run feature selection if enabled
        if self.config['use_feature_selection']:
            self.run_feature_selection()
        
        self.run_model_training()
        self.run_interpretability()
        self.run_prediction()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print("\nAll steps completed successfully!")
        print("\nKey Outputs:")
        print("  1. EDA Results: eda_outputs/")
        print("  2. Preprocessed Data: processed_data/")
        print("  3. Engineered Features: engineered_data/")
        print("  4. Trained Models: trained_models/")
        print("  5. Evaluation Results: evaluation_results/")
        print(f"  6. Interpretability: interpretability_results/")
        print(f"     - Individual models: logistic_regression/, random_forest/, xgboost/, lightgbm/, catboost/")
        print(f"     - Ensemble (all 5 models): ensemble_all_models/")
        print(f"     - Ensemble (3 tree models): ensemble_tree_models/")
        print(f"  7. Submission Files:")
        print(f"     - submission_logistic_regression.csv")
        print(f"     - submission_random_forest.csv")
        print(f"     - submission_xgboost.csv")
        print(f"     - submission_lightgbm.csv")
        print(f"     - submission_catboost.csv")
        print(f"     - submission_ensemble.csv (Recommended)")
        print("\nRecommended submission: submission_ensemble.csv")


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Depression Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Run full pipeline
                python main.py --full
                
                # Run specific steps
                python main.py --eda
                python main.py --preprocess
                python main.py --train
                python main.py --predict
                
                # Run with hyperparameter optimization
                python main.py --full --optimize --trials 50
                
                # Skip EDA if already done
                python main.py --full --skip-eda
                """
    )
    
    # Pipeline steps
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    parser.add_argument('--eda', action='store_true', help='Run EDA only')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing only')
    parser.add_argument('--engineer', action='store_true', help='Run feature engineering only')
    parser.add_argument('--feature-selection', action='store_true', help='Run feature selection only')
    parser.add_argument('--train', action='store_true', help='Run model training only')
    parser.add_argument('--interpret', action='store_true', help='Run interpretability analysis only')
    parser.add_argument('--predict', action='store_true', help='Run prediction only')
    
    # Configuration options
    parser.add_argument('--train-path', default='data/train.csv', help='Path to training data')
    parser.add_argument('--test-path', default='data/test.csv', help='Path to test data')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE')
    parser.add_argument('--no-scaling', action='store_true', help='Disable feature scaling')
    parser.add_argument('--encoding-method', type=str, default='label', choices=['target', 'label'],
                        help='Encoding method for Profession feature: label or target (default: label)')
    parser.add_argument('--val-size', type=float, default=0.2, help='Validation set size')
    parser.add_argument('--optimize', action='store_true', help='Enable hyperparameter optimization')
    parser.add_argument('--optimize-lr', action='store_true', help='Enable learning rate optimization')
    parser.add_argument('--trials', type=int, default=30, help='Number of optimization trials')
    parser.add_argument('--model', default='catboost', help='Best model to use')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA step')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'train_path': args.train_path,
        'test_path': args.test_path,
        'use_smote': not args.no_smote,
        'smote_method': 'smote',
        'val_size': args.val_size,
        'scaling_method': 'none' if args.no_scaling else 'standard',
        'encoding_method': args.encoding_method,  # 'target' or 'label'
        'use_feature_selection': args.feature_selection,
        'n_selected_features': 20,
        'optimize_hyperparams': args.optimize,
        'optimize_learning_rate': args.optimize_lr,
        'n_trials': args.trials,
        'best_model': args.model,
        'ensemble_models': ['xgboost', 'lightgbm', 'catboost'],
        'shap_samples': 2000
    }
    
    # Initialize pipeline
    pipeline = DepressionPredictionPipeline(config)
    
    # Run requested steps
    if args.full:
        pipeline.run_full_pipeline(skip_eda=args.skip_eda)
    elif args.eda:
        pipeline.run_eda()
    elif args.preprocess:
        pipeline.run_preprocessing()
    elif args.engineer:
        pipeline.run_feature_engineering()
    elif args.feature_selection:
        pipeline.run_feature_selection()
    elif args.train:
        pipeline.run_model_training()
    elif args.interpret:
        pipeline.run_interpretability()
    elif args.predict:
        pipeline.run_prediction()
    else:
        parser.print_help()
        print("\nNo action specified. Use --full to run complete pipeline or specify individual steps.")
        sys.exit(1)


if __name__ == "__main__":
    main()

