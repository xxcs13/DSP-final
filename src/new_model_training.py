"""
Model Training Module for Depression Prediction

Implements multiple models:
1. XGBoost
2. LightGBM
3. CatBoost
4. Random Forest
5. Logistic Regression (baseline)

Features:
- Hyperparameter tuning with Optuna
- Cross-validation
- Model persistence
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def detect_gpu_availability():
    """
    Detect GPU availability for XGBoost, LightGBM, and CatBoost
    
    Returns:
        Dictionary with GPU availability status for each library
    """
    gpu_status = {
        'xgboost': False,
        'lightgbm': False,
        'catboost': False,
        'device_info': 'CPU'
    }
    
    try:
        # Check CUDA availability through PyTorch or TensorFlow if available
        import torch
        if torch.cuda.is_available():
            gpu_status['device_info'] = f'GPU - {torch.cuda.get_device_name(0)}'
            gpu_status['xgboost'] = True
            gpu_status['lightgbm'] = True
            gpu_status['catboost'] = True
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  - XGBoost: GPU available")
            print(f"  - LightGBM: GPU available")
            print(f"  - CatBoost: GPU available")
            return gpu_status
    except ImportError:
        pass
    
    try:
        # Alternative: Check CUDA through environment or direct CUDA check
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            gpu_status['device_info'] = 'GPU detected via nvidia-smi'
            gpu_status['xgboost'] = True
            gpu_status['lightgbm'] = True
            gpu_status['catboost'] = True
            print("GPU detected via nvidia-smi")
            print("  - XGBoost: GPU available")
            print("  - LightGBM: GPU available")
            print("  - CatBoost: GPU available")
            return gpu_status
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    print("No GPU detected - using CPU")
    print("  - XGBoost: CPU mode")
    print("  - LightGBM: CPU mode")
    print("  - CatBoost: CPU mode")
    
    return gpu_status


class ModelTrainer:
    def __init__(self, random_state=55, n_jobs=-1):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.best_params = {}
        self.training_history = {}  # Store training history for each model
        
        # Detect GPU availability
        print("\n" + "="*80)
        print("GPU DETECTION")
        print("="*80)
        self.gpu_status = detect_gpu_availability()
        print("="*80 + "\n")
        
    def get_xgboost_model(self, params=None):
        """
        Get XGBoost classifier
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            XGBoost classifier
        """
        default_params = {
            'n_estimators': 500,   
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Configure based on GPU availability
        if self.gpu_status['xgboost']:
            # GPU configuration for XGBoost 2.0+
            default_params['device'] = 'cuda'
            default_params['tree_method'] = 'hist'
            # Don't set n_jobs with GPU (handled by CUDA)
            print("  XGBoost: Configured for GPU acceleration (device=cuda)")
        else:
            # CPU configuration
            default_params['n_jobs'] = self.n_jobs
            default_params['tree_method'] = 'hist'  # hist is faster on CPU too
        
        if params:
            default_params.update(params)
        
        return XGBClassifier(**default_params)
        
    def get_lightgbm_model(self, params=None):
        """
        Get LightGBM classifier
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            LightGBM classifier
        """
        default_params = {
            'n_estimators': 500,   
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'num_leaves': 31,
            'min_split_gain': 0.0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': self.random_state,
            'verbose': -1,
            'force_col_wise': True
        }
        
        # Configure based on GPU availability
        if self.gpu_status['lightgbm']:
            default_params['device'] = 'gpu'
            # Don't set n_jobs with GPU
            print("  LightGBM: Configured for GPU acceleration")
        else:
            default_params['n_jobs'] = self.n_jobs
        
        if params:
            default_params.update(params)
        
        return LGBMClassifier(**default_params)
        
    def get_catboost_model(self, params=None):
        """
        Get CatBoost classifier
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            CatBoost classifier
        """
        default_params = {
            'iterations': 500,   
            'depth': 5,
            'learning_rate': 0.1,
            'l2_leaf_reg': 1.25,
            'random_state': self.random_state,
            'verbose': False,
            # Disable automatic file writing to avoid creating catboost_info directory
            'allow_writing_files': False,
            'train_dir': None
        }
        
        # Configure based on GPU availability
        if self.gpu_status['catboost']:
            default_params['task_type'] = 'GPU'
            # CatBoost GPU doesn't use thread_count
            print("  CatBoost: Configured for GPU acceleration")
        else:
            # CPU configuration
            default_params['thread_count'] = self.n_jobs if self.n_jobs > 0 else None
        
        if params:
            default_params.update(params)
        
        return CatBoostClassifier(**default_params)
        
    def get_random_forest_model(self, params=None):
        """
        Get Random Forest classifier
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            Random Forest classifier
        """
        default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        
        if params:
            default_params.update(params)
        
        return RandomForestClassifier(**default_params)
        
    def get_logistic_regression_model(self, params=None):
        """
        Get Logistic Regression classifier (baseline)
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            Logistic Regression classifier
        """
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        
        if params:
            default_params.update(params)
        
        return LogisticRegression(**default_params)
        
    def _handle_nan_for_model(self, X, model, model_name=None, fit=True):
        """
        Handle NaN values based on model type
        
        Tree-based models (XGBoost, LightGBM, CatBoost) handle NaN natively.
        Other models (LogisticRegression, RandomForest) need NaN to be imputed.
        
        Args:
            X: Feature matrix (may contain NaN)
            model: Model instance to check type
            model_name: Name of the model (for storing imputation values)
            fit: If True, compute and store imputation values; if False, use stored values
            
        Returns:
            X with NaN handled appropriately for the model type
        """
        # Check if model supports NaN natively
        nan_native_models = (XGBClassifier, LGBMClassifier, CatBoostClassifier)
        
        if isinstance(model, nan_native_models):
            # Tree-based models handle NaN natively - return as-is
            return X
        
        # For other models, impute NaN with column median
        X_copy = np.array(X, dtype=np.float64)  # Ensure we have a writable copy
        
        # Initialize storage for imputation values if needed
        if not hasattr(self, '_nan_impute_values_per_model'):
            self._nan_impute_values_per_model = {}
        
        if fit:
            # Compute and store column medians for imputation
            impute_values = np.nanmedian(X_copy, axis=0)
            # Handle columns where all values are NaN - use 0 as fallback
            impute_values = np.where(np.isnan(impute_values), 0.0, impute_values)
            if model_name:
                self._nan_impute_values_per_model[model_name] = impute_values
            self._current_impute_values = impute_values
        else:
            # Use stored values
            if model_name and model_name in self._nan_impute_values_per_model:
                impute_values = self._nan_impute_values_per_model[model_name]
            elif hasattr(self, '_current_impute_values'):
                impute_values = self._current_impute_values
            else:
                # Fallback: compute from current data
                impute_values = np.nanmedian(X_copy, axis=0)
                impute_values = np.where(np.isnan(impute_values), 0.0, impute_values)
        
        # Find NaN positions and impute
        nan_mask = np.isnan(X_copy)
        if nan_mask.any():
            nan_count = nan_mask.sum()
            print(f"  Note: Imputing {nan_count} NaN values for {type(model).__name__} (does not support NaN natively)")
            
            # Use numpy advanced indexing for efficient imputation
            # Create a broadcast-able imputation array
            for col_idx in range(X_copy.shape[1]):
                col_nan_mask = nan_mask[:, col_idx]
                if col_nan_mask.any():
                    X_copy[col_nan_mask, col_idx] = impute_values[col_idx]
            
            # Verify no NaN remaining
            remaining_nan = np.isnan(X_copy).sum()
            if remaining_nan > 0:
                print(f"  Warning: {remaining_nan} NaN values could not be imputed, filling with 0")
                X_copy = np.nan_to_num(X_copy, nan=0.0)
        
        return X_copy
    
    def predict_with_model(self, model, X, model_name=None):
        """
        Make predictions with a trained model, handling NaN values appropriately
        
        Args:
            model: Trained model instance
            X: Feature matrix (may contain NaN)
            model_name: Name of the model (for retrieving stored imputation values)
            
        Returns:
            Predictions (y_pred, y_pred_proba)
        """
        # Handle NaN values for non-tree models
        X_processed = self._handle_nan_for_model(X, model, model_name=model_name, fit=False)
        
        y_pred = model.predict(X_processed)
        y_pred_proba = model.predict_proba(X_processed)[:, 1]
        
        return y_pred, y_pred_proba
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None, model_name='model'):
        """
        Train a model with early stopping support and training monitoring
        
        Automatically handles NaN values based on model type:
        - Tree-based models (XGBoost, LightGBM, CatBoost): Keep NaN (native support)
        - Other models (LogisticRegression, RandomForest): Impute NaN with median
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")
        
        # Handle NaN values based on model type
        X_train = self._handle_nan_for_model(X_train, model, model_name=model_name, fit=True)
        if X_val is not None:
            X_val = self._handle_nan_for_model(X_val, model, model_name=model_name, fit=False)
        
        # Initialize training history
        self.training_history[model_name] = {
            'train_scores': [],
            'val_scores': [],
            'best_iteration': None,
            'best_score': None
        }
        
        # Check if model supports early stopping and validation set is provided
        if X_val is not None and y_val is not None:
            if isinstance(model, XGBClassifier):
                # XGBoost with early stopping
                print(f"  Training with early stopping (patience=50)...")
                
                # For XGBoost 1.3.0+, we need to set early_stopping_rounds as a fit parameter
                # XGBoost 2.0+ uses a different approach - check model params first
                model_params = model.get_params()
                
                # Train with early stopping
                model.set_params(early_stopping_rounds=50)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False
                )
                
                # Extract training history
                results = model.evals_result()
                self.training_history[model_name]['train_scores'] = results['validation_0']['logloss']
                self.training_history[model_name]['val_scores'] = results['validation_1']['logloss']
                self.training_history[model_name]['best_iteration'] = model.best_iteration
                self.training_history[model_name]['best_score'] = model.best_score
                
                print(f"  ✓ Best iteration: {model.best_iteration}")
                print(f"  ✓ Best validation score: {model.best_score:.4f}")
                
            elif isinstance(model, LGBMClassifier):
                # LightGBM with early stopping (using callbacks)
                print(f"  Training with early stopping (patience=50)...")
                from lightgbm import early_stopping
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    eval_metric='binary_logloss',
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                # Extract training history
                results = model.evals_result_
                self.training_history[model_name]['train_scores'] = results['training']['binary_logloss']
                self.training_history[model_name]['val_scores'] = results['valid_1']['binary_logloss']
                self.training_history[model_name]['best_iteration'] = model.best_iteration_
                self.training_history[model_name]['best_score'] = results['valid_1']['binary_logloss'][model.best_iteration_ - 1]
                
                print(f"  ✓ Best iteration: {model.best_iteration_}")
                print(f"  ✓ Best validation score: {self.training_history[model_name]['best_score']:.4f}")
                
            elif isinstance(model, CatBoostClassifier):
                # CatBoost with early stopping
                print(f"  Training with early stopping (patience=50)...")
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                # Extract training history from CatBoost using get_evals_result()
                evals_result = model.get_evals_result()
                
                # CatBoost stores results in 'learn' and 'validation' keys
                if 'learn' in evals_result and 'Logloss' in evals_result['learn']:
                    self.training_history[model_name]['train_scores'] = evals_result['learn']['Logloss']
                
                if 'validation' in evals_result and 'Logloss' in evals_result['validation']:
                    self.training_history[model_name]['val_scores'] = evals_result['validation']['Logloss']
                
                self.training_history[model_name]['best_iteration'] = model.best_iteration_
                self.training_history[model_name]['best_score'] = model.best_score_
                
                print(f"  ✓ Best iteration: {model.best_iteration_}")
                if isinstance(model.best_score_, dict):
                    best_score_val = model.best_score_.get('validation', model.best_score_.get('learn', 'N/A'))
                    print(f"  ✓ Best validation score: {best_score_val}")
                else:
                    print(f"  ✓ Best score: {model.best_score_}")
                
            else:
                # Models without early stopping (RF, LR)
                # Train the model and capture validation performance
                print(f"  Training without early stopping...")
                model.fit(X_train, y_train)
                
                # For RF and LR, we can still track performance on validation set
                if X_val is not None and y_val is not None:
                    from sklearn.metrics import log_loss
                    
                    # Get predictions on training and validation sets
                    try:
                        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                        y_val_pred_proba = model.predict_proba(X_val)[:, 1]
                        
                        train_loss = log_loss(y_train, y_train_pred_proba)
                        val_loss = log_loss(y_val, y_val_pred_proba)
                        
                        # Store single-point "history" for these models
                        self.training_history[model_name]['train_scores'] = [train_loss]
                        self.training_history[model_name]['val_scores'] = [val_loss]
                        self.training_history[model_name]['best_iteration'] = 1
                        self.training_history[model_name]['best_score'] = val_loss
                        
                        print(f"  ✓ Training loss: {train_loss:.4f}")
                        print(f"  ✓ Validation loss: {val_loss:.4f}")
                    except Exception as e:
                        print(f"  Warning: Could not compute validation metrics: {e}")

        else:
            # No validation set provided
            print(f"  Training without validation set...")
            model.fit(X_train, y_train)
        
        print(f"  ✓ {model_name} training complete!")
        self.models[model_name] = model
        return model
        
    def optimize_xgboost(self, X_train, y_train, n_trials=50):
        """
        Optimize XGBoost hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        print(f"\nOptimizing XGBoost hyperparameters ({n_trials} trials)...")
        
        # Check GPU availability
        use_gpu = self.gpu_status['xgboost']
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            
            # Add GPU or CPU specific parameters
            if use_gpu:
                params['device'] = 'cuda'
                params['tree_method'] = 'hist'  # Use 'hist' for GPU in XGBoost 2.0+
            else:
                params['n_jobs'] = self.n_jobs
            
            model = XGBClassifier(**params)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                    scoring='f1', n_jobs=1)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best F1-Score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        self.best_params['xgboost'] = study.best_params
        return study.best_params
        
    def optimize_lightgbm(self, X_train, y_train, n_trials=50):
        """
        Optimize LightGBM hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        print(f"\nOptimizing LightGBM hyperparameters ({n_trials} trials)...")
        
        # Check GPU availability
        use_gpu = self.gpu_status['lightgbm']
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'random_state': self.random_state,
                'verbose': -1,
                'force_col_wise': True
            }
            
            # Add GPU or CPU specific parameters
            if use_gpu:
                params['device'] = 'gpu'
            else:
                params['n_jobs'] = self.n_jobs
            
            try:
                model = LGBMClassifier(**params)
                
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                        scoring='f1', n_jobs=1)
                
                return scores.mean()
            except Exception as e:
                print(f"Trial failed with params: {params}")
                print(f"Error: {str(e)}")
                return 0.0
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best F1-Score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        self.best_params['lightgbm'] = study.best_params
        return study.best_params
        
    def optimize_catboost(self, X_train, y_train, n_trials=50):
        """
        Optimize CatBoost hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        print(f"\nOptimizing CatBoost hyperparameters ({n_trials} trials)...")
        
        # Check GPU availability
        use_gpu = self.gpu_status['catboost']
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': self.random_state,
                'verbose': False,
                # Disable automatic file writing
                'allow_writing_files': False,
                'train_dir': None
            }
            
            # Add GPU or CPU specific parameters
            if use_gpu:
                params['task_type'] = 'GPU'
            else:
                params['thread_count'] = self.n_jobs if self.n_jobs > 0 else -1
            
            model = CatBoostClassifier(**params)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                    scoring='f1', n_jobs=1)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize', study_name='catboost_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best F1-Score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        self.best_params['catboost'] = study.best_params
        return study.best_params
        
    def save_model(self, model_name, output_dir='trained_models'):
        """
        Save trained model to disk
        
        Args:
            model_name: Name of the model to save
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        filepath = output_path / f"{model_name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        print(f"Model saved to: {filepath}")
        
    def load_model(self, model_name, model_dir='trained_models'):
        """
        Load trained model from disk
        
        Args:
            model_name: Name of the model to load
            model_dir: Directory containing models
            
        Returns:
            Loaded model
        """
        filepath = Path(model_dir) / f"{model_name}.pkl"
        
        if not filepath.exists():
            print(f"Model file not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        print(f"Model loaded from: {filepath}")
        
        return model
        
    def cross_validate_model(self, model, X, y, cv=5, scoring='f1'):
        """
        Perform cross-validation on a model
        
        Args:
            model: Model instance
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=self.n_jobs)
        
        print(f"Cross-validation {scoring} scores: {scores}")
        print(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def plot_learning_curves(self, model_names=None, output_dir='training_curves'):
        """
        Plot learning curves for trained models
        
        Args:
            model_names: List of model names to plot (None = all models with history)
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if model_names is None:
            model_names = [name for name in self.training_history.keys() 
                          if len(self.training_history[name].get('train_scores', [])) > 0]
        
        if not model_names:
            print("No training history available to plot.")
            return
        
        print(f"\nPlotting learning curves for {len(model_names)} models...")
        
        for model_name in model_names:
            history = self.training_history.get(model_name, {})
            train_scores = history.get('train_scores', [])
            val_scores = history.get('val_scores', [])
            
            if not train_scores or not val_scores:
                print(f"  Skipping {model_name} (no training history)")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            iterations = range(1, len(train_scores) + 1)
            
            # Plot training and validation scores
            ax.plot(iterations, train_scores, label='Training Loss', 
                   linewidth=2, color='blue', alpha=0.8)
            ax.plot(iterations, val_scores, label='Validation Loss', 
                   linewidth=2, color='red', alpha=0.8)
            
            # Mark best iteration
            best_iter = history.get('best_iteration')
            if best_iter is not None:
                best_score = history.get('best_score')
                ax.axvline(x=best_iter, color='green', linestyle='--', 
                          linewidth=1.5, alpha=0.7, 
                          label=f'Best Iteration ({best_iter})')
                ax.scatter([best_iter], [val_scores[best_iter-1]], 
                          color='green', s=100, zorder=5, marker='*')
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Log Loss', fontsize=12)
            ax.set_title(f'Learning Curve - {model_name.upper()}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = output_path / f'{model_name}_learning_curve.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {plot_path}")
        
        # Create comparison plot if multiple models
        if len(model_names) > 1:
            self._plot_comparison_curves(model_names, output_path)
    
    def _plot_comparison_curves(self, model_names, output_path):
        """
        Plot comparison of validation curves for multiple models
        
        Args:
            model_names: List of model names
            output_path: Path object for output directory
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(range(len(model_names)))
        
        for idx, model_name in enumerate(model_names):
            history = self.training_history.get(model_name, {})
            val_scores = history.get('val_scores', [])
            
            if not val_scores:
                continue
            
            iterations = range(1, len(val_scores) + 1)
            ax.plot(iterations, val_scores, label=model_name.upper(), 
                   linewidth=2, color=colors[idx], alpha=0.8)
            
            # Mark best iteration
            best_iter = history.get('best_iteration')
            if best_iter is not None:
                ax.scatter([best_iter], [val_scores[best_iter-1]], 
                          color=colors[idx], s=80, zorder=5, marker='o')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Validation Log Loss', fontsize=12)
        ax.set_title('Model Comparison - Validation Learning Curves', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = output_path / 'model_comparison_learning_curves.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved comparison: {plot_path}")
    
    def save_training_history(self, output_dir='training_history'):
        """
        Save training history to CSV files
        
        Args:
            output_dir: Directory to save history files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving training history...")
        
        for model_name, history in self.training_history.items():
            train_scores = history.get('train_scores', [])
            val_scores = history.get('val_scores', [])
            
            if not train_scores or not val_scores:
                continue
            
            # Create DataFrame
            df = pd.DataFrame({
                'iteration': range(1, len(train_scores) + 1),
                'train_loss': train_scores,
                'val_loss': val_scores
            })
            
            # Add metadata
            df.attrs['best_iteration'] = history.get('best_iteration')
            df.attrs['best_score'] = history.get('best_score')
            
            # Save to CSV
            csv_path = output_path / f'{model_name}_training_history.csv'
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved: {csv_path}")
        
        # Save summary
        summary_data = []
        for model_name, history in self.training_history.items():
            if history.get('best_iteration'):
                summary_data.append({
                    'model': model_name,
                    'best_iteration': history['best_iteration'],
                    'best_score': history.get('best_score', 'N/A'),
                    'total_iterations': len(history.get('train_scores', []))
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = output_path / 'training_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"  ✓ Saved summary: {summary_path}")
            print("\nTraining Summary:")
            print(summary_df.to_string(index=False))


def train_all_models(X_train, y_train, X_val, y_val, 
                     optimize=False, n_trials=50,
                     optimize_learning_rate=False,
                     lr_search_range=None,
                     lr_max_iterations=1000,
                     lr_early_stopping_rounds=50,
                     output_dir='trained_models'):
    """
    Train all models and save them
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        optimize: Whether to perform hyperparameter optimization
        n_trials: Number of optimization trials
        optimize_learning_rate: Whether to optimize learning rates for tree models
        lr_search_range: List of learning rates to test during optimization
                        If None, uses default [0.07, 0.1, 0.13, 0.15, 0.17, 0.2, 0.3]
        lr_max_iterations: Maximum iterations for early stopping in LR tuning (default: 1000)
        lr_early_stopping_rounds: Early stopping patience for LR tuning (default: 50)
        output_dir: Directory to save models
        
    Returns:
        Tuple of (results, trainer, evaluator, optimal_thresholds, optimal_learning_rates, optimal_tree_counts)
    """
    from evaluation_metrics import ModelEvaluator
    
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    results = {}
    optimal_learning_rates = {}
    
    print("="*80)
    print("MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Optimize learning rates if requested
    optimal_tree_counts = {}  # Initialize optimal tree counts
    if optimize_learning_rate:
        print("\n" + "="*80)
        print("LEARNING RATE OPTIMIZATION")
        print("="*80)
        from learning_rate_tuning import optimize_all_models_learning_rates
        
        # Prepare model configurations for learning rate optimization
        # Note: n_estimators removed from params - will be determined by early stopping
        models_config = {
            'xgboost': {
                'params': {
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'min_child_weight': 1,
                    'reg_alpha': 0,
                    'reg_lambda': 0.1
                },
                'gpu_available': trainer.gpu_status['xgboost']
            },
            'lightgbm': {
                'params': {
                    'num_leaves': 31,
                    'max_depth': -1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 20,
                    'reg_alpha': 0,
                    'reg_lambda': 0.1
                },
                'gpu_available': trainer.gpu_status['lightgbm']
            },
            'catboost': {
                'params': {
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'border_count': 254,
                    'bagging_temperature': 1,
                    'random_strength': 1
                },
                'gpu_available': trainer.gpu_status['catboost']
            }
        }
        
        # Use smaller sample for faster learning rate optimization
        sample_size = min(20000, len(X_train))
        print(f"\nUsing sample of {sample_size} training samples for learning rate optimization...")
        indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]
        
        # Optimize learning rates with early stopping (returns tuple: (optimal_lrs, optimal_trees))
        # If lr_search_range is None, function will use sensible defaults: [0.07, 0.1, 0.13, 0.15, 0.17, 0.2, 0.3]
        optimal_learning_rates, optimal_tree_counts = optimize_all_models_learning_rates(
            models_config=models_config,
            X_train=X_train_sample,
            y_train=y_train_sample,
            X_val=X_val,
            y_val=y_val,
            learning_rates=lr_search_range,
            metric='accuracy',
            verbose=True,
            random_state=trainer.random_state,
            n_jobs=trainer.n_jobs,
            use_early_stopping=True,
            max_iterations=lr_max_iterations,
            early_stopping_rounds=lr_early_stopping_rounds
        )
        
        # Save optimal learning rates and tree counts
        import json
        lr_path = Path(output_dir) / 'optimal_learning_rates.json'
        with open(lr_path, 'w') as f:
            json.dump({'learning_rates': optimal_learning_rates, 'tree_counts': optimal_tree_counts}, f, indent=2)
        print(f"\nOptimal learning rates and tree counts saved to: {lr_path}")
        
        # Display LR-Trees relationship
        print("\nLearning Rate - Trees Relationship:")
        print("-" * 50)
        for model_name in optimal_learning_rates:
            lr = optimal_learning_rates[model_name]
            trees = optimal_tree_counts.get(model_name, 'N/A')
            print(f"{model_name:15s}: LR={lr:.4f}, Trees={trees}")
        print("-" * 50)
    else:
        print("\nSkipping learning rate optimization (use --optimize-lr flag to enable)")
        optimal_learning_rates = {}
        optimal_tree_counts = {}
    
    # 1. Logistic Regression (baseline)
    print("\n" + "="*80)
    print("1. LOGISTIC REGRESSION (BASELINE)")
    print("="*80)
    lr_model = trainer.get_logistic_regression_model()
    trainer.train_model(lr_model, X_train, y_train, X_val, y_val, 'logistic_regression')
    
    y_pred_lr, y_pred_proba_lr = trainer.predict_with_model(lr_model, X_val, 'logistic_regression')
    results['logistic_regression'] = {
        'model': lr_model,
        'y_pred': y_pred_lr,
        'y_pred_proba': y_pred_proba_lr
    }
    
    evaluator.evaluate_model('LogisticRegression', y_val, y_pred_lr, y_pred_proba_lr)
    trainer.save_model('logistic_regression', output_dir)
    
    # 2. Random Forest
    print("\n" + "="*80)
    print("2. RANDOM FOREST")
    print("="*80)
    rf_model = trainer.get_random_forest_model()
    trainer.train_model(rf_model, X_train, y_train, X_val, y_val, 'random_forest')
    
    y_pred_rf, y_pred_proba_rf = trainer.predict_with_model(rf_model, X_val, 'random_forest')
    results['random_forest'] = {
        'model': rf_model,
        'y_pred': y_pred_rf,
        'y_pred_proba': y_pred_proba_rf
    }
    
    evaluator.evaluate_model('RandomForest', y_val, y_pred_rf, y_pred_proba_rf)
    trainer.save_model('random_forest', output_dir)
    
    # 3. XGBoost
    print("\n" + "="*80)
    print("3. XGBOOST")
    print("="*80)
    
    if optimize:
        xgb_params = trainer.optimize_xgboost(X_train, y_train, n_trials=n_trials)
        # Override learning rate and n_estimators if optimized
        if 'xgboost' in optimal_learning_rates:
            xgb_params['learning_rate'] = optimal_learning_rates['xgboost']
            print(f"Using optimized learning rate: {optimal_learning_rates['xgboost']:.4f}")
        if 'xgboost' in optimal_tree_counts:
            xgb_params['n_estimators'] = optimal_tree_counts['xgboost']
            print(f"Using optimized n_estimators: {optimal_tree_counts['xgboost']}")
        xgb_model = trainer.get_xgboost_model(xgb_params)
    else:
        # Use optimal learning rate and n_estimators if available
        xgb_params = {}
        if 'xgboost' in optimal_learning_rates:
            xgb_params['learning_rate'] = optimal_learning_rates['xgboost']
            print(f"Using optimized learning rate: {optimal_learning_rates['xgboost']:.4f}")
        if 'xgboost' in optimal_tree_counts:
            xgb_params['n_estimators'] = optimal_tree_counts['xgboost']
            print(f"Using optimized n_estimators: {optimal_tree_counts['xgboost']}")
        xgb_model = trainer.get_xgboost_model(xgb_params if xgb_params else None)
    
    trainer.train_model(xgb_model, X_train, y_train, X_val, y_val, 'xgboost')
    
    y_pred_xgb = xgb_model.predict(X_val)
    y_pred_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
    results['xgboost'] = {
        'model': xgb_model,
        'y_pred': y_pred_xgb,
        'y_pred_proba': y_pred_proba_xgb
    }
    
    evaluator.evaluate_model('XGBoost', y_val, y_pred_xgb, y_pred_proba_xgb)
    trainer.save_model('xgboost', output_dir)
    
    # 4. LightGBM
    print("\n" + "="*80)
    print("4. LIGHTGBM")
    print("="*80)
    
    if optimize:
        lgb_params = trainer.optimize_lightgbm(X_train, y_train, n_trials=n_trials)
        # Override learning rate and n_estimators if optimized
        if 'lightgbm' in optimal_learning_rates:
            lgb_params['learning_rate'] = optimal_learning_rates['lightgbm']
            print(f"Using optimized learning rate: {optimal_learning_rates['lightgbm']:.4f}")
        if 'lightgbm' in optimal_tree_counts:
            lgb_params['n_estimators'] = optimal_tree_counts['lightgbm']
            print(f"Using optimized n_estimators: {optimal_tree_counts['lightgbm']}")
        lgb_model = trainer.get_lightgbm_model(lgb_params)
    else:
        # Use optimal learning rate and n_estimators if available
        lgb_params = {}
        if 'lightgbm' in optimal_learning_rates:
            lgb_params['learning_rate'] = optimal_learning_rates['lightgbm']
            print(f"Using optimized learning rate: {optimal_learning_rates['lightgbm']:.4f}")
        if 'lightgbm' in optimal_tree_counts:
            lgb_params['n_estimators'] = optimal_tree_counts['lightgbm']
            print(f"Using optimized n_estimators: {optimal_tree_counts['lightgbm']}")
        lgb_model = trainer.get_lightgbm_model(lgb_params if lgb_params else None)
    
    trainer.train_model(lgb_model, X_train, y_train, X_val, y_val, 'lightgbm')
    
    y_pred_lgb = lgb_model.predict(X_val)
    y_pred_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]
    results['lightgbm'] = {
        'model': lgb_model,
        'y_pred': y_pred_lgb,
        'y_pred_proba': y_pred_proba_lgb
    }
    
    evaluator.evaluate_model('LightGBM', y_val, y_pred_lgb, y_pred_proba_lgb)
    trainer.save_model('lightgbm', output_dir)
    
    # 5. CatBoost
    print("\n" + "="*80)
    print("5. CATBOOST")
    print("="*80)
    
    if optimize:
        cat_params = trainer.optimize_catboost(X_train, y_train, n_trials=n_trials)
        # Override learning rate and iterations if optimized
        if 'catboost' in optimal_learning_rates:
            cat_params['learning_rate'] = optimal_learning_rates['catboost']
            print(f"Using optimized learning rate: {optimal_learning_rates['catboost']:.4f}")
        if 'catboost' in optimal_tree_counts:
            cat_params['iterations'] = optimal_tree_counts['catboost']
            print(f"Using optimized iterations: {optimal_tree_counts['catboost']}")
        cat_model = trainer.get_catboost_model(cat_params)
    else:
        # Use optimal learning rate and iterations if available
        cat_params = {}
        if 'catboost' in optimal_learning_rates:
            cat_params['learning_rate'] = optimal_learning_rates['catboost']
            print(f"Using optimized learning rate: {optimal_learning_rates['catboost']:.4f}")
        if 'catboost' in optimal_tree_counts:
            cat_params['iterations'] = optimal_tree_counts['catboost']
            print(f"Using optimized iterations: {optimal_tree_counts['catboost']}")
        cat_model = trainer.get_catboost_model(cat_params if cat_params else None)
    
    trainer.train_model(cat_model, X_train, y_train, X_val, y_val, 'catboost')
    
    y_pred_cat = cat_model.predict(X_val)
    y_pred_proba_cat = cat_model.predict_proba(X_val)[:, 1]
    results['catboost'] = {
        'model': cat_model,
        'y_pred': y_pred_cat,
        'y_pred_proba': y_pred_proba_cat
    }
    
    evaluator.evaluate_model('CatBoost', y_val, y_pred_cat, y_pred_proba_cat)
    trainer.save_model('catboost', output_dir)
    
    # 6. Ensemble (Average of XGBoost, LightGBM, CatBoost)
    print("\n" + "="*80)
    print("6. ENSEMBLE MODEL")
    print("="*80)
    
    # Average predictions from tree-based models
    y_pred_proba_ensemble = (y_pred_proba_xgb + y_pred_proba_lgb + y_pred_proba_cat) / 3
    y_pred_ensemble = (y_pred_proba_ensemble >= 0.5).astype(int)
    
    results['ensemble'] = {
        'model': None,  # Ensemble is a combination, not a single model
        'y_pred': y_pred_ensemble,
        'y_pred_proba': y_pred_proba_ensemble
    }
    
    evaluator.evaluate_model('Ensemble', y_val, y_pred_ensemble, y_pred_proba_ensemble)
    
    # Generate comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    evaluator.compare_models()
    evaluator.save_results_to_csv()
    evaluator.generate_summary_report()
    
    # Save training history and plot learning curves
    print("\n" + "="*80)
    print("TRAINING ANALYSIS")
    print("="*80)
    trainer.save_training_history(output_dir='training_history')
    trainer.plot_learning_curves(output_dir='training_curves')
    
    # Optimize thresholds for all models
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION")
    print("="*80)
    from threshold_tuning import optimize_all_models_thresholds
    
    models_dict = {
        'logistic_regression': lr_model,
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'catboost': cat_model
    }
    
    optimal_thresholds = optimize_all_models_thresholds(
        models_dict, X_val, y_val, metric='accuracy', verbose=False
    )
    
    # Save optimal thresholds
    import json
    thresholds_path = Path(output_dir) / 'optimal_thresholds.json'
    with open(thresholds_path, 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"\nOptimal thresholds saved to: {thresholds_path}")
    
    # Store optimal thresholds in results
    for model_name in optimal_thresholds:
        if model_name in results:
            results[model_name]['optimal_threshold'] = optimal_thresholds[model_name]
    
    # Store optimal learning rates in results
    for model_name in optimal_learning_rates:
        if model_name in results:
            results[model_name]['optimal_learning_rate'] = optimal_learning_rates[model_name]
    
    # Store optimal tree counts in results
    for model_name in optimal_tree_counts:
        if model_name in results:
            results[model_name]['optimal_tree_count'] = optimal_tree_counts[model_name]
    
    return results, trainer, evaluator, optimal_thresholds, optimal_learning_rates, optimal_tree_counts


if __name__ == "__main__":
    # Load engineered data
    print("Loading engineered data...")
    X_train = np.load('engineered_data/X_train_eng.npy')
    y_train = np.load('engineered_data/y_train.npy')
    X_val = np.load('engineered_data/X_val_eng.npy')
    y_val = np.load('engineered_data/y_val.npy')
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train all models (without optimization for speed)
    results, trainer, evaluator, optimal_thresholds, optimal_learning_rates, optimal_tree_counts = train_all_models(
        X_train, y_train, X_val, y_val,
        optimize=False,  # Set to True for hyperparameter tuning
        n_trials=30
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nAll models trained and saved!")
    if optimal_learning_rates:
        print("\nOptimal Learning Rates:")
        for model_name, lr in optimal_learning_rates.items():
            trees = optimal_tree_counts.get(model_name, 'N/A')
            print(f"  {model_name}: LR={lr:.4f}, Trees={trees}")
    if optimal_thresholds:
        print("\nOptimal Thresholds:")
        for model_name, threshold in optimal_thresholds.items():
            print(f"  {model_name}: {threshold:.4f}")

