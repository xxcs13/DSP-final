"""
Prediction Pipeline for Depression Detection

Generates predictions for test set and creates submission file
in the required format. Supports optimized probability thresholds.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def _handle_nan_for_prediction(X, model):
    """
    Handle NaN values for models that do not support NaN natively
    
    Tree-based models (XGBoost, LightGBM, CatBoost) handle NaN natively.
    Other models (LogisticRegression, RandomForest) need NaN to be imputed.
    
    Args:
        X: Feature matrix (may contain NaN)
        model: Model instance to check type
        
    Returns:
        X with NaN imputed if necessary
    """
    # Try to import tree model types
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = type(None)
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = type(None)
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        CatBoostClassifier = type(None)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    # Check if model supports NaN natively
    nan_native_models = (XGBClassifier, LGBMClassifier, CatBoostClassifier)
    
    if isinstance(model, nan_native_models):
        # Tree-based models handle NaN natively - return as-is
        return X
    
    # For other models, check if there are NaN values
    X_arr = np.asarray(X, dtype=np.float64)
    nan_mask = np.isnan(X_arr)
    
    if not nan_mask.any():
        return X
    
    # Impute NaN with column median
    X_copy = X_arr.copy()
    impute_values = np.nanmedian(X_copy, axis=0)
    # Handle columns where all values are NaN - use 0 as fallback
    impute_values = np.where(np.isnan(impute_values), 0.0, impute_values)
    
    nan_count = nan_mask.sum()
    print(f"  Note: Imputing {nan_count} NaN values for prediction")
    
    for col_idx in range(X_copy.shape[1]):
        col_nan_mask = nan_mask[:, col_idx]
        if col_nan_mask.any():
            X_copy[col_nan_mask, col_idx] = impute_values[col_idx]
    
    # Final fallback for any remaining NaN
    X_copy = np.nan_to_num(X_copy, nan=0.0)
    
    return X_copy


class PredictionPipeline:
    def __init__(self, model_path, test_data_path, test_ids_path, threshold=0.5):
        """
        Initialize prediction pipeline
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test features
            test_ids_path: Path to test IDs
            threshold: Classification threshold (default 0.5)
        """
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.test_ids_path = Path(test_ids_path)
        self.threshold = threshold
        self.model = None
        self.X_test = None
        self.test_ids = None
        
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("Model loaded successfully!")
        
    def load_test_data(self):
        """Load test data"""
        print(f"Loading test data from: {self.test_data_path}")
        self.X_test = np.load(self.test_data_path)
        self.test_ids = np.load(self.test_ids_path)
        print(f"Test data loaded: {self.X_test.shape}")
        print(f"Test IDs loaded: {len(self.test_ids)}")
        
    def predict(self):
        """
        Make predictions on test set using specified threshold
        
        Handles NaN values appropriately based on model type:
        - Tree-based models (XGBoost, LightGBM, CatBoost): Keep NaN (native support)
        - Other models (LogisticRegression, RandomForest): Impute NaN with median
        
        Returns:
            Predicted labels and probabilities
        """
        print(f"\nMaking predictions with threshold={self.threshold:.3f}...")
        
        # Handle NaN values for models that don't support them natively
        X_test_processed = _handle_nan_for_prediction(self.X_test, self.model)
        
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        print(f"Predictions generated for {len(y_pred)} samples")
        print(f"Predicted distribution:")
        print(f"  No Depression (0): {np.sum(y_pred == 0)} ({np.sum(y_pred == 0) / len(y_pred) * 100:.2f}%)")
        print(f"  Depression (1): {np.sum(y_pred == 1)} ({np.sum(y_pred == 1) / len(y_pred) * 100:.2f}%)")
        
        return y_pred, y_pred_proba
        
    def create_submission(self, y_pred, output_path='submission.csv'):
        """
        Create submission file in required format
        
        Args:
            y_pred: Predicted labels
            output_path: Path to save submission file
        """
        submission_df = pd.DataFrame({
            'id': self.test_ids,
            'Depression': y_pred
        })
        
        submission_df.to_csv(output_path, index=False)
        print(f"\nSubmission file saved to: {output_path}")
        
        # Display first few rows
        print("\nSubmission preview:")
        print(submission_df.head(10))
        
        return submission_df
        
    def run_pipeline(self, output_path='submission.csv'):
        """
        Run complete prediction pipeline
        
        Args:
            output_path: Path to save submission file
            
        Returns:
            Submission DataFrame
        """
        print("="*80)
        print("PREDICTION PIPELINE")
        print("="*80 + "\n")
        
        self.load_model()
        self.load_test_data()
        y_pred, y_pred_proba = self.predict()
        submission_df = self.create_submission(y_pred, output_path)
        
        print("\n" + "="*80)
        print("PREDICTION PIPELINE COMPLETE")
        print("="*80)
        
        return submission_df


def generate_predictions(model_name='catboost', 
                        data_dir='engineered_data',
                        model_dir='trained_models',
                        output_path='submission.csv',
                        use_optimal_threshold=True):
    """
    Generate predictions using specified model
    
    Args:
        model_name: Name of the model to use
        data_dir: Directory containing test data
        model_dir: Directory containing trained models
        output_path: Path to save submission file
        use_optimal_threshold: Use optimal threshold from training (default True)
        
    Returns:
        Submission DataFrame
    """
    # Load optimal threshold if available
    threshold = 0.5
    if use_optimal_threshold:
        thresholds_path = Path(model_dir) / 'optimal_thresholds.json'
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                optimal_thresholds = json.load(f)
            threshold = optimal_thresholds.get(model_name, 0.5)
            print(f"Using optimal threshold for {model_name}: {threshold:.3f}")
        else:
            print(f"Optimal thresholds file not found. Using default threshold: {threshold}")
    
    # Check training metadata to determine which features to use
    metadata_path = Path(model_dir) / 'training_metadata.json'
    use_selected_features = False
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            use_selected_features = metadata.get('use_feature_selection', False)
            print(f"Training metadata found: using {'SELECTED' if use_selected_features else 'ALL'} features")
    else:
        print("Warning: No training metadata found. Using ALL features by default.")
    
    # Load appropriate test data based on training metadata
    if use_selected_features:
        test_data_path = Path(data_dir) / 'X_test_selected.npy'
        if not test_data_path.exists():
            raise FileNotFoundError(f"Model was trained with selected features, but {test_data_path} not found")
    else:
        test_data_path = Path(data_dir) / 'X_test_eng.npy'
    
    pipeline = PredictionPipeline(
        model_path=Path(model_dir) / f"{model_name}.pkl",
        test_data_path=test_data_path,
        test_ids_path=Path(data_dir) / 'test_ids.npy',
        threshold=threshold
    )
    
    submission = pipeline.run_pipeline(output_path)
    
    return submission


def generate_ensemble_predictions(model_names=['xgboost', 'lightgbm', 'catboost'],
                                  data_dir='engineered_data',
                                  model_dir='trained_models',
                                  output_path='submission_ensemble.csv',
                                  use_optimal_threshold=True,
                                  ensemble_threshold=0.55):
    """
    Generate ensemble predictions by averaging multiple models
    
    Args:
        model_names: List of model names to ensemble
        data_dir: Directory containing test data
        model_dir: Directory containing trained models
        output_path: Path to save submission file
        use_optimal_threshold: Use average of optimal thresholds (default True)
        ensemble_threshold: Manual ensemble threshold (used if use_optimal_threshold=False)
        
    Returns:
        Submission DataFrame
    """
    print("="*80)
    print("ENSEMBLE PREDICTION PIPELINE")
    print("="*80 + "\n")
    
    # Check training metadata to determine which features to use
    metadata_path = Path(model_dir) / 'training_metadata.json'
    use_selected_features = False
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            use_selected_features = metadata.get('use_feature_selection', False)
            print(f"Training metadata found: using {'SELECTED' if use_selected_features else 'ALL'} features")
    else:
        print("Warning: No training metadata found. Using ALL features by default.")
    
    # Load test data based on training metadata
    if use_selected_features:
        test_path = Path(data_dir) / 'X_test_selected.npy'
        if not test_path.exists():
            raise FileNotFoundError(f"Model was trained with selected features, but {test_path} not found")
        X_test = np.load(test_path)
        print(f"Using SELECTED features: {X_test.shape[1]} features")
    else:
        X_test = np.load(Path(data_dir) / 'X_test_eng.npy')
        print(f"Using ALL engineered features: {X_test.shape[1]} features")
    
    test_ids = np.load(Path(data_dir) / 'test_ids.npy')
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Models to ensemble: {model_names}\n")
    
    # Load optimal thresholds if available
    final_threshold = ensemble_threshold
    if use_optimal_threshold:
        thresholds_path = Path(model_dir) / 'optimal_thresholds.json'
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                optimal_thresholds = json.load(f)
            
            # Average optimal thresholds for ensemble models
            ensemble_thresholds = [optimal_thresholds.get(name, 0.5) for name in model_names]
            final_threshold = np.mean(ensemble_thresholds)
            print(f"Individual optimal thresholds:")
            for name, thresh in zip(model_names, ensemble_thresholds):
                print(f"  {name}: {thresh:.3f}")
            print(f"Average ensemble threshold: {final_threshold:.3f}\n")
        else:
            print(f"Optimal thresholds file not found. Using default threshold: {final_threshold}\n")
    
    # Collect predictions from all models
    all_predictions = []
    
    for model_name in model_names:
        print(f"Loading {model_name}...")
        model_path = Path(model_dir) / f"{model_name}.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        all_predictions.append(y_pred_proba)
        print(f"  Predictions obtained from {model_name}")
    
    # Average predictions
    print("\nAveraging predictions...")
    avg_predictions = np.mean(all_predictions, axis=0)
    
    # Convert to binary labels using optimal threshold
    y_pred_ensemble = (avg_predictions >= final_threshold).astype(int)
    
    print(f"\nEnsemble predictions generated for {len(y_pred_ensemble)} samples")
    print(f"Final threshold used: {final_threshold:.3f}")
    print(f"Predicted distribution:")
    print(f"  No Depression (0): {np.sum(y_pred_ensemble == 0)} ({np.sum(y_pred_ensemble == 0) / len(y_pred_ensemble) * 100:.2f}%)")
    print(f"  Depression (1): {np.sum(y_pred_ensemble == 1)} ({np.sum(y_pred_ensemble == 1) / len(y_pred_ensemble) * 100:.2f}%)")
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Depression': y_pred_ensemble
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"\nEnsemble submission file saved to: {output_path}")
    
    # Display first few rows
    print("\nSubmission preview:")
    print(submission_df.head(10))
    
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTION PIPELINE COMPLETE")
    print("="*80)
    
    return submission_df


if __name__ == "__main__":
    print("Generating predictions using best model (CatBoost)...\n")
    
    # Generate single model predictions
    # submission = generate_predictions(
    #     model_name='catboost',
    #     output_path='submission_catboost.csv'
    # )
    
    # print("\n\n")
    
    # Generate ensemble predictions
    submission_ensemble = generate_ensemble_predictions(
        model_names=['xgboost', 'lightgbm', 'catboost'],
        output_path='submission_ensemble.csv'
    )
    
    print("\n" + "="*80)
    print("ALL PREDICTIONS GENERATED")
    print("="*80)
    print("\nOutput files:")
    # print("  - submission_catboost.csv (CatBoost model)")
    print("  - submission_ensemble.csv (Ensemble of XGBoost, LightGBM, CatBoost)")
    print("\nRecommendation: Use the ensemble submission for better robustness!")

