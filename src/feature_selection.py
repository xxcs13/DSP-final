"""
Feature Selection Module Using SHAP

This module provides SHAP-based feature selection functionality to identify
and select the most important features for model training.
"""

import numpy as np
import pandas as pd
import shap
from pathlib import Path
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


class SHAPFeatureSelector:
    """
    Feature selector using SHAP values to identify important features
    """
    
    def __init__(self, n_features=20, model_type='lightgbm', random_state=42):
        """
        Initialize feature selector
        
        Args:
            n_features: Number of top features to select
            model_type: Type of model to use for SHAP calculation ('lightgbm' recommended)
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.selected_features = None
        self.feature_importance = None
        
    def fit(self, X, y, feature_names=None):
        """
        Fit a model and calculate SHAP-based feature importance
        
        Args:
            X: Training features (numpy array or DataFrame)
            y: Training labels
            feature_names: List of feature names (optional)
            
        Returns:
            self
        """
        print("\n" + "=" * 80)
        print("SHAP-BASED FEATURE SELECTION")
        print("=" * 80)
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        print(f"  Input features: {len(feature_names)}")
        print(f"  Target features to select: {self.n_features}")
        
        # Adjust n_features if it exceeds available features
        if self.n_features > len(feature_names):
            print(f"  WARNING: Requested {self.n_features} features but only {len(feature_names)} available")
            print(f"  Using all {len(feature_names)} features")
            self.n_features = len(feature_names)
        
        # Train a simple LightGBM model for SHAP calculation
        print("\n  Step 1: Training LightGBM model for SHAP calculation...")
        self.model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
            verbose=-1
        )
        self.model.fit(X, y)
        print("  Model trained successfully")
        
        # Calculate SHAP values
        print("\n  Step 2: Calculating SHAP values...")
        # Use TreeExplainer for tree-based models (faster)
        explainer = shap.TreeExplainer(self.model)
        
        # Sample data if too large (for speed)
        sample_size = min(2000, X.shape[0])
        if X.shape[0] > sample_size:
            print(f"  Sampling {sample_size} instances for SHAP calculation...")
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        print("  SHAP values calculated")
        
        # Calculate mean absolute SHAP value for each feature
        print("\n  Step 3: Computing feature importance...")
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # Select top N features
        self.selected_features = importance_df.head(self.n_features)['feature'].tolist()
        
        print(f"\n  Top {self.n_features} features selected:")
        for i, (feature, importance) in enumerate(importance_df.head(self.n_features).values, 1):
            print(f"    {i:2d}. {feature:40s} (importance: {importance:.6f})")
        
        print("\n" + "=" * 80)
        
        return self
    
    def transform(self, X, feature_names=None):
        """
        Transform data to keep only selected features
        
        Args:
            X: Input features (numpy array or DataFrame)
            feature_names: List of feature names (optional, required if X is numpy array)
            
        Returns:
            X_selected: Data with only selected features
        """
        if self.selected_features is None:
            raise ValueError("Must call fit() before transform()")
        
        # Handle DataFrame
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features].values
        
        # Handle numpy array
        if feature_names is None:
            raise ValueError("feature_names must be provided when X is a numpy array")
        
        # Get indices of selected features
        feature_indices = [feature_names.index(f) for f in self.selected_features]
        
        return X[:, feature_indices]
    
    def fit_transform(self, X, y, feature_names=None):
        """
        Fit and transform in one step
        
        Args:
            X: Training features
            y: Training labels
            feature_names: List of feature names (optional)
            
        Returns:
            X_selected: Data with only selected features
        """
        self.fit(X, y, feature_names)
        return self.transform(X, feature_names)
    
    def get_feature_names(self):
        """
        Get list of selected feature names
        
        Returns:
            List of selected feature names
        """
        if self.selected_features is None:
            raise ValueError("Must call fit() before getting feature names")
        
        return self.selected_features
    
    def save_importance(self, output_path):
        """
        Save feature importance to file
        
        Args:
            output_path: Path to save the feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Must call fit() before saving importance")
        
        self.feature_importance.to_csv(output_path, index=False)
        print(f"Feature importance saved to: {output_path}")


def select_features(input_dir='engineered_data', output_dir='engineered_data', 
                   n_features=20, save_importance=True):
    """
    Main function to perform feature selection on engineered data
    
    Args:
        input_dir: Directory containing engineered features
        output_dir: Directory to save selected features
        n_features: Number of features to select
        save_importance: Whether to save feature importance
        
    Returns:
        selector: Fitted SHAPFeatureSelector object
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("FEATURE SELECTION PIPELINE")
    print("=" * 80)
    
    # Load engineered data
    print("\nLoading engineered data...")
    X_train = np.load(input_dir / 'X_train_eng.npy')
    y_train = np.load(input_dir / 'y_train.npy')
    X_val = np.load(input_dir / 'X_val_eng.npy')
    X_test = np.load(input_dir / 'X_test_eng.npy')
    test_ids = np.load(input_dir / 'test_ids.npy')
    
    # Load feature names
    with open(input_dir / 'feature_names_eng.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Number of features: {len(feature_names)}")
    
    # Perform feature selection
    selector = SHAPFeatureSelector(n_features=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train, feature_names)
    
    # Transform validation and test sets
    print("\nTransforming validation and test sets...")
    X_val_selected = selector.transform(X_val, feature_names)
    X_test_selected = selector.transform(X_test, feature_names)
    
    print(f"  X_train_selected shape: {X_train_selected.shape}")
    print(f"  X_val_selected shape: {X_val_selected.shape}")
    print(f"  X_test_selected shape: {X_test_selected.shape}")
    
    # Save selected features
    print("\nSaving selected features...")
    np.save(output_dir / 'X_train_selected.npy', X_train_selected)
    np.save(output_dir / 'X_val_selected.npy', X_val_selected)
    np.save(output_dir / 'X_test_selected.npy', X_test_selected)
    
    # Save selected feature names
    with open(output_dir / 'feature_names_selected.txt', 'w') as f:
        for feature in selector.get_feature_names():
            f.write(f"{feature}\n")
    
    print(f"  Saved to: {output_dir}/")
    
    # Save feature importance
    if save_importance:
        selector.save_importance(output_dir / 'feature_importance_shap.csv')
    
    print("\n" + "=" * 80)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 80)
    print(f"\nSelected {n_features} features from {len(feature_names)} original features")
    print(f"Reduction: {(1 - n_features/len(feature_names))*100:.1f}%")
    
    return selector


if __name__ == "__main__":
    # Example usage
    selector = select_features(
        input_dir='engineered_data',
        output_dir='engineered_data',
        n_features=20,
        save_importance=True
    )
