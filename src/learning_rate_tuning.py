"""
Learning Rate Optimization Module

This module provides functionality to optimize learning rates for
tree-based models (XGBoost, LightGBM, CatBoost) using validation set performance.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


class LearningRateOptimizer:
    """
    Optimizer for finding optimal learning rates for gradient boosting models
    """
    
    def __init__(self, model_type, base_params, X_train, y_train, X_val, y_val, 
                 random_state=55, n_jobs=-1, gpu_available=False, use_early_stopping=True,
                 max_iterations=1000, early_stopping_rounds=50):
        """
        Initialize learning rate optimizer
        
        IMPORTANT: Learning rate and number of trees are strongly correlated.
        Lower learning rates require more trees to converge, while higher learning rates
        converge faster but may overfit with too many trees.
        
        This optimizer uses early stopping to automatically determine the optimal number
        of trees for each learning rate, properly accounting for the LR-trees trade-off.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
            base_params: Base parameters for the model (without learning_rate and n_estimators)
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            random_state: Random seed
            n_jobs: Number of parallel jobs
            gpu_available: Whether GPU is available for this model
            use_early_stopping: Whether to use early stopping (default: True, recommended)
            max_iterations: Maximum number of trees/iterations (default: 1000)
            early_stopping_rounds: Rounds without improvement before stopping (default: 50)
        """
        self.model_type = model_type.lower()
        self.base_params = base_params.copy()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.gpu_available = gpu_available
        self.use_early_stopping = use_early_stopping
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        
        # Remove learning_rate and n_estimators from base_params if present
        self.base_params.pop('learning_rate', None)
        self.base_params.pop('eta', None)
        self.base_params.pop('n_estimators', None)
        self.base_params.pop('iterations', None)
        self.base_params.pop('num_iterations', None)
        
        # Results storage
        self.results = {}
        
    def _create_model(self, learning_rate):
        """
        Create model with specified learning rate and max iterations
        
        The model is configured with a high max_iterations value.
        Early stopping will determine the optimal number of trees for this learning rate.
        
        Args:
            learning_rate: Learning rate value
            
        Returns:
            Model instance
        """
        params = self.base_params.copy()
        params['learning_rate'] = learning_rate
        params['random_state'] = self.random_state
        
        if self.model_type == 'xgboost':
            params['n_estimators'] = self.max_iterations
            if self.gpu_available:
                params['device'] = 'cuda'
                params['tree_method'] = 'hist'
            else:
                params['n_jobs'] = self.n_jobs
                params['tree_method'] = 'hist'
            params['eval_metric'] = 'logloss'
            params['use_label_encoder'] = False
            return XGBClassifier(**params)
            
        elif self.model_type == 'lightgbm':
            params['n_estimators'] = self.max_iterations
            if self.gpu_available:
                params['device'] = 'gpu'
                params['gpu_platform_id'] = 0
                params['gpu_device_id'] = 0
            else:
                params['n_jobs'] = self.n_jobs
            params['verbosity'] = -1
            params['force_col_wise'] = True
            # Ensure minimum safety constraints
            if 'min_child_samples' not in params:
                params['min_child_samples'] = 20
            if 'num_leaves' not in params:
                params['num_leaves'] = 31
            return LGBMClassifier(**params)
            
        elif self.model_type == 'catboost':
            params['iterations'] = self.max_iterations
            if self.gpu_available:
                params['task_type'] = 'GPU'
                params['devices'] = '0'
            else:
                params['thread_count'] = self.n_jobs
            params['verbose'] = False
            params['allow_writing_files'] = False
            return CatBoostClassifier(**params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def evaluate_learning_rate(self, learning_rate, verbose=False):
        """
        Evaluate model performance with specific learning rate
        
        Uses early stopping to automatically determine optimal number of trees.
        Lower learning rates typically require more trees, higher rates fewer trees.
        
        Args:
            learning_rate: Learning rate to test
            verbose: Print progress
            
        Returns:
            Dictionary with metrics including optimal tree count
        """
        if verbose:
            print(f"  Testing learning_rate={learning_rate:.4f}...", end=' ')
        
        # Create model with max_iterations
        model = self._create_model(learning_rate)
        
        # Train with or without early stopping
        if self.use_early_stopping:
            # Use early stopping to find optimal tree count
            if self.model_type == 'xgboost':
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False
                )
                best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
                
            elif self.model_type == 'lightgbm':
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    callbacks=[warnings.filterwarnings('ignore')]
                )
                best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
                
            elif self.model_type == 'catboost':
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=(self.X_val, self.y_val),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
                best_iteration = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.tree_count_
        else:
            # Train without early stopping (use all max_iterations)
            model.fit(self.X_train, self.y_train)
            best_iteration = self.max_iterations
        
        # Make predictions
        y_pred = model.predict(self.X_val)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        auc = roc_auc_score(self.y_val, y_pred_proba)
        
        results = {
            'learning_rate': learning_rate,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'best_iteration': best_iteration,
            'trees_used': best_iteration
        }
        
        if verbose:
            print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Trees: {best_iteration}")
        
        return results
    
    def find_optimal_learning_rate(self, learning_rates=None, metric='accuracy', verbose=True):
        """
        Find optimal learning rate by testing multiple values
        
        With early stopping enabled, each learning rate automatically determines
        its optimal number of trees. Lower LRs typically use more trees.
        
        Args:
            learning_rates: List of learning rates to test. If None, uses default range
            metric: Metric to optimize ('accuracy', 'f1_score', 'auc')
            verbose: Print progress
            
        Returns:
            Tuple of (optimal_learning_rate, all_results)
        """
        if learning_rates is None:
            # Default learning rate search space (log scale)
            learning_rates = [0.07, 0.1, 0.13, 0.15, 0.17, 0.2, 0.3]
        
        if verbose:
            early_stop_msg = " (with early stopping)" if self.use_early_stopping else ""
            print(f"\nTesting {len(learning_rates)} learning rates for {self.model_type}{early_stop_msg}:")
            print(f"Learning rates to test: {learning_rates}")
            print(f"Optimizing for: {metric}")
            if self.use_early_stopping:
                print(f"Max iterations: {self.max_iterations}, Early stopping rounds: {self.early_stopping_rounds}")
            print()
        
        # Evaluate each learning rate
        for lr in learning_rates:
            results = self.evaluate_learning_rate(lr, verbose=verbose)
            self.results[lr] = results
        
        # Find best learning rate based on metric
        best_lr = max(self.results.keys(), key=lambda k: self.results[k][metric])
        best_score = self.results[best_lr][metric]
        best_trees = self.results[best_lr]['trees_used']
        
        if verbose:
            print(f"\nOptimal learning rate: {best_lr:.4f}")
            print(f"Best {metric}: {best_score:.4f}")
            print(f"Optimal trees used: {best_trees}")
            
            # Show LR-trees relationship
            if self.use_early_stopping and len(self.results) > 1:
                print(f"\nLearning Rate vs Trees Relationship:")
                print(f"{'LR':<12}{'Trees':<12}{metric.upper()}")
                print("-" * 36)
                for lr in sorted(self.results.keys()):
                    trees = self.results[lr]['trees_used']
                    score = self.results[lr][metric]
                    marker = " <-- BEST" if lr == best_lr else ""
                    print(f"{lr:<12.4f}{trees:<12}{score:.4f}{marker}")
        
        return best_lr, self.results
    
    def compare_learning_rates(self, learning_rates, metrics=['accuracy', 'f1_score', 'auc']):
        """
        Compare performance across different learning rates
        
        Args:
            learning_rates: List of learning rates to compare
            metrics: List of metrics to display
        """
        print(f"\nLearning Rate Comparison for {self.model_type}:")
        print("="*80)
        
        # Header
        header = f"{'LR':<10}"
        for metric in metrics:
            header += f"{metric.upper():<12}"
        print(header)
        print("-"*80)
        
        # Results
        for lr in sorted(learning_rates):
            if lr in self.results:
                row = f"{lr:<10.4f}"
                for metric in metrics:
                    row += f"{self.results[lr][metric]:<12.4f}"
                print(row)
        
        print("="*80)


def optimize_model_learning_rate(model_type, base_params, X_train, y_train, X_val, y_val,
                                 learning_rates=None, metric='accuracy', verbose=True,
                                 random_state=55, n_jobs=-1, gpu_available=False,
                                 use_early_stopping=True, max_iterations=1000, 
                                 early_stopping_rounds=50):
    """
    Optimize learning rate for a single model
    
    IMPORTANT: Uses early stopping to automatically determine optimal tree count
    for each learning rate. This properly accounts for the LR-trees trade-off:
    - Lower LR (e.g., 0.01): Better generalization, needs more trees (e.g., 500+)
    - Higher LR (e.g., 0.3): Faster convergence, needs fewer trees (e.g., 50-100)
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
        base_params: Base parameters for the model (should NOT include n_estimators/iterations)
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        learning_rates: List of learning rates to test
        metric: Metric to optimize
        verbose: Print progress
        random_state: Random seed
        n_jobs: Number of parallel jobs
        gpu_available: Whether GPU is available
        use_early_stopping: Whether to use early stopping (default: True, recommended)
        max_iterations: Maximum number of trees/iterations (default: 1000)
        early_stopping_rounds: Rounds without improvement before stopping (default: 50)
        
    Returns:
        Tuple of (optimal_learning_rate, optimizer)
    """
    optimizer = LearningRateOptimizer(
        model_type=model_type,
        base_params=base_params,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state,
        n_jobs=n_jobs,
        gpu_available=gpu_available,
        use_early_stopping=use_early_stopping,
        max_iterations=max_iterations,
        early_stopping_rounds=early_stopping_rounds
    )
    
    optimal_lr, results = optimizer.find_optimal_learning_rate(
        learning_rates=learning_rates,
        metric=metric,
        verbose=verbose
    )
    
    return optimal_lr, optimizer


def optimize_all_models_learning_rates(models_config, X_train, y_train, X_val, y_val,
                                       learning_rates=None, metric='accuracy', 
                                       verbose=True, random_state=55, n_jobs=-1,
                                       use_early_stopping=True, max_iterations=1000,
                                       early_stopping_rounds=50):
    """
    Optimize learning rates for all tree-based models
    
    IMPORTANT: Uses early stopping to handle LR-trees trade-off correctly.
    Each model will automatically determine its optimal tree count for each LR tested.
    
    Args:
        models_config: Dictionary with model configurations
                      Format: {model_name: {'params': {...}, 'gpu_available': bool}}
                      Note: params should NOT include n_estimators/iterations
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        learning_rates: List of learning rates to test
        metric: Metric to optimize
        verbose: Print progress
        random_state: Random seed
        n_jobs: Number of parallel jobs
        use_early_stopping: Whether to use early stopping (default: True, recommended)
        max_iterations: Maximum number of trees/iterations (default: 1000)
        early_stopping_rounds: Rounds without improvement before stopping (default: 50)
        
    Returns:
        Dictionary of optimal learning rates and tree counts per model
    """
    optimal_learning_rates = {}
    optimal_tree_counts = {}
    
    for model_name, config in models_config.items():
        if verbose:
            print("\n" + "="*80)
            print(f"OPTIMIZING LEARNING RATE: {model_name.upper()}")
            print("="*80)
        
        optimal_lr, optimizer = optimize_model_learning_rate(
            model_type=model_name,
            base_params=config['params'],
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            learning_rates=learning_rates,
            metric=metric,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            gpu_available=config.get('gpu_available', False),
            use_early_stopping=use_early_stopping,
            max_iterations=max_iterations,
            early_stopping_rounds=early_stopping_rounds
        )
        
        optimal_learning_rates[model_name] = optimal_lr
        optimal_tree_counts[model_name] = optimizer.results[optimal_lr]['trees_used']
    
    if verbose:
        print("\n" + "="*80)
        print("OPTIMAL LEARNING RATES AND TREE COUNTS SUMMARY")
        print("="*80)
        print(f"{'Model':<20}{'Learning Rate':<20}{'Trees':<12}")
        print("-"*80)
        for model_name in optimal_learning_rates:
            lr = optimal_learning_rates[model_name]
            trees = optimal_tree_counts[model_name]
            print(f"{model_name:<20}{lr:<20.4f}{trees:<12}")
        print("="*80)
        
        if use_early_stopping:
            print("\nNote: Tree counts were automatically determined via early stopping.")
            print("Lower learning rates typically require more trees to converge.")
    
    return optimal_learning_rates, optimal_tree_counts


if __name__ == "__main__":
    # Example usage
    print("Learning Rate Optimization Module")
    print("This module is designed to be imported and used in the training pipeline.")
    print("\nExample usage:")
    print("""
    from learning_rate_tuning import optimize_model_learning_rate
    
    optimal_lr, optimizer = optimize_model_learning_rate(
        model_type='xgboost',
        base_params={'n_estimators': 500, 'max_depth': 5},
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        metric='accuracy'
    )
    """)
