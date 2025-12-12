"""
Probability Threshold Tuning Module

Optimizes classification threshold for binary classification models
to maximize specified metrics (F1-score, Accuracy, etc.)
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class ThresholdOptimizer:
    def __init__(self, metric='accuracy', thresholds=None):
        """
        Initialize threshold optimizer
        
        Args:
            metric: Optimization metric ('f1', 'accuracy', 'precision', 'recall')
            thresholds: Array of thresholds to test (default: 0.1 to 0.9 with 0.01 step)
        """
        self.metric = metric
        self.thresholds = thresholds if thresholds is not None else np.arange(0.1, 0.91, 0.01)
        self.best_threshold = 0.5
        self.best_score = 0.0
        self.threshold_scores = {}
        
    def find_optimal_threshold(self, y_true, y_pred_proba, verbose=True):
        """
        Find optimal threshold by testing multiple threshold values
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            verbose: Print optimization progress
            
        Returns:
            Optimal threshold value
        """
        if verbose:
            print(f"\nOptimizing threshold for {self.metric} metric...")
            print(f"Testing {len(self.thresholds)} threshold values from {self.thresholds[0]:.2f} to {self.thresholds[-1]:.2f}")
        
        best_threshold = 0.5
        best_score = 0.0
        threshold_scores = {}
        
        for threshold in self.thresholds:
            # Convert probabilities to binary predictions using current threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metric based on selection
            if self.metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif self.metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif self.metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif self.metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            threshold_scores[threshold] = score
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        self.best_score = best_score
        self.threshold_scores = threshold_scores
        
        if verbose:
            print(f"\nOptimal threshold found: {best_threshold:.3f}")
            print(f"Best {self.metric} score: {best_score:.4f}")
            default_score = threshold_scores.get(0.5, best_score)
            print(f"Score at default 0.5 threshold: {default_score:.4f}")
            print(f"Improvement: {(best_score - default_score):.4f}")
        
        return best_threshold
    
    def evaluate_threshold(self, y_true, y_pred_proba, threshold, verbose=True):
        """
        Evaluate all metrics at a specific threshold
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            threshold: Threshold value to evaluate
            verbose: Print evaluation results
            
        Returns:
            Dictionary of metric values
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        if verbose:
            print(f"\nMetrics at threshold {threshold:.3f}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
        
        return metrics
    
    def compare_thresholds(self, y_true, y_pred_proba, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
        """
        Compare multiple threshold values
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            thresholds: List of thresholds to compare
            
        Returns:
            Dictionary mapping thresholds to their metrics
        """
        print("\nThreshold Comparison:")
        print("="*80)
        
        results = {}
        for threshold in thresholds:
            metrics = self.evaluate_threshold(y_true, y_pred_proba, threshold, verbose=False)
            results[threshold] = metrics
            
            print(f"Threshold {threshold:.2f}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1']:.4f}, Prec={metrics['precision']:.4f}, "
                  f"Rec={metrics['recall']:.4f}")
        
        return results


def optimize_model_threshold(model, X_val, y_val, metric='accuracy', verbose=True):
    """
    Optimize threshold for a trained model
    
    Args:
        model: Trained classification model with predict_proba method
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ('f1', 'accuracy', etc.)
        verbose: Print optimization details
        
    Returns:
        Optimal threshold and optimizer object
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(metric=metric)
    
    # Find optimal threshold
    best_threshold = optimizer.find_optimal_threshold(y_val, y_pred_proba, verbose=verbose)
    
    # Compare with default threshold
    if verbose:
        print("\nComparison with default threshold:")
        print("-"*80)
        optimizer.compare_thresholds(y_val, y_pred_proba, thresholds=[0.5, best_threshold])
    
    return best_threshold, optimizer


def optimize_all_models_thresholds(models_dict, X_val, y_val, metric='accuracy', verbose=True):
    """
    Optimize thresholds for multiple models
    
    Args:
        models_dict: Dictionary mapping model names to trained models
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize
        verbose: Print optimization details
        
    Returns:
        Dictionary mapping model names to optimal thresholds
    """
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION FOR ALL MODELS")
    print("="*80)
    
    optimal_thresholds = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'='*80}")
        print(f"Optimizing threshold for {model_name.upper()}")
        print("="*80)
        
        best_threshold, optimizer = optimize_model_threshold(
            model, X_val, y_val, metric=metric, verbose=verbose
        )
        
        optimal_thresholds[model_name] = best_threshold
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("="*80)
    for model_name, threshold in optimal_thresholds.items():
        print(f"{model_name:20s}: {threshold:.3f}")
    
    return optimal_thresholds


if __name__ == "__main__":
    # Example usage
    print("Threshold Tuning Module")
    print("="*80)
    print("This module optimizes classification thresholds for binary classification")
    print("\nUsage:")
    print("  from threshold_tuning import optimize_model_threshold")
    print("  best_threshold, optimizer = optimize_model_threshold(model, X_val, y_val)")
