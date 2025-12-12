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


class ModelTrainer:
    def __init__(self, random_state=50, n_jobs=-1):
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
        
    def get_xgboost_model(self, params=None):
        """
        Get XGBoost classifier
        
        Args:
            params: Hyperparameters dictionary
            
        Returns:
            XGBoost classifier
        """
        default_params = {
            'n_estimators': 250,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'min_child_weight': 1,
            'reg_alpha': 0,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
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
            'n_estimators': 250,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': -1
        }
        
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
            'iterations': 250,
            'depth': 5,
            'learning_rate': 0.1,
            'l2_leaf_reg': 1.25,
            'random_state': self.random_state,
            'verbose': False,
            'thread_count': self.n_jobs if self.n_jobs > 0 else None
        }
        
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
            'n_estimators': 250,
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
        
    def train_model(self, model, X_train, y_train, model_name='model'):
        """
        Train a model
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        print(f"{model_name} training complete!")
        
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
                'n_jobs': self.n_jobs,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            
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
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1
            }
            
            model = LGBMClassifier(**params)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, 
                                    scoring='f1', n_jobs=1)
            
            return scores.mean()
        
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
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': self.random_state,
                'verbose': False,
                'thread_count': 1
            }
            
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


def train_all_models(X_train, y_train, X_val, y_val, 
                     optimize=False, n_trials=30,
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
        output_dir: Directory to save models
        
    Returns:
        Dictionary of trained models and predictions
    """
    from evaluation_metrics import ModelEvaluator
    
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    results = {}
    
    print("="*80)
    print("MODEL TRAINING PIPELINE")
    print("="*80)
    
    # 1. Logistic Regression (baseline)
    print("\n" + "="*80)
    print("1. LOGISTIC REGRESSION (BASELINE)")
    print("="*80)
    lr_model = trainer.get_logistic_regression_model()
    trainer.train_model(lr_model, X_train, y_train, 'logistic_regression')
    
    y_pred_lr = lr_model.predict(X_val)
    y_pred_proba_lr = lr_model.predict_proba(X_val)[:, 1]
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
    trainer.train_model(rf_model, X_train, y_train, 'random_forest')
    
    y_pred_rf = rf_model.predict(X_val)
    y_pred_proba_rf = rf_model.predict_proba(X_val)[:, 1]
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
        xgb_model = trainer.get_xgboost_model(xgb_params)
    else:
        xgb_model = trainer.get_xgboost_model()
    
    trainer.train_model(xgb_model, X_train, y_train, 'xgboost')
    
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
        lgb_model = trainer.get_lightgbm_model(lgb_params)
    else:
        lgb_model = trainer.get_lightgbm_model()
    
    trainer.train_model(lgb_model, X_train, y_train, 'lightgbm')
    
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
        cat_model = trainer.get_catboost_model(cat_params)
    else:
        cat_model = trainer.get_catboost_model()
    
    trainer.train_model(cat_model, X_train, y_train, 'catboost')
    
    y_pred_cat = cat_model.predict(X_val)
    y_pred_proba_cat = cat_model.predict_proba(X_val)[:, 1]
    results['catboost'] = {
        'model': cat_model,
        'y_pred': y_pred_cat,
        'y_pred_proba': y_pred_proba_cat
    }
    
    evaluator.evaluate_model('CatBoost', y_val, y_pred_cat, y_pred_proba_cat)
    trainer.save_model('catboost', output_dir)
    
    # Generate comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    evaluator.compare_models()
    evaluator.save_results_to_csv()
    evaluator.generate_summary_report()
    
    return results, trainer, evaluator


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
    results, trainer, evaluator = train_all_models(
        X_train, y_train, X_val, y_val,
        optimize=False,  # Set to True for hyperparameter tuning
        n_trials=30
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nAll models trained and saved!")

