"""
Evaluation Metrics Module for Depression Prediction

Calculates comprehensive performance metrics:
- Accuracy
- AUC (Area Under ROC Curve)
- F1-Score
- Precision and Recall
- Confusion Matrix
- Classification Report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self, output_dir='evaluation_results'):
        """
        Initialize model evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for AUC)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC requires predicted probabilities
        if y_pred_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['auc'] = None
        else:
            metrics['auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0)
        
        return metrics
        
    def print_metrics(self, metrics, dataset_name='Dataset'):
        """
        Print metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics
            dataset_name: Name of the dataset (e.g., 'Validation', 'Test')
        """
        print(f"\n{'='*80}")
        print(f"{dataset_name.upper()} METRICS")
        print(f"{'='*80}\n")
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        if metrics['auc'] is not None:
            print(f"AUC:       {metrics['auc']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        
        print(f"\n{'='*80}")
        print("CONFUSION MATRIX")
        print(f"{'='*80}\n")
        print(metrics['confusion_matrix'])
        
        print(f"\n{'='*80}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*80}\n")
        print(metrics['classification_report'])
        
    def plot_confusion_matrix(self, confusion_mat, save_name='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            confusion_mat: Confusion matrix
            save_name: Filename to save plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Depression', 'Depression'],
                   yticklabels=['No Depression', 'Depression'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_roc_curve(self, y_true, y_pred_proba, save_name='roc_curve.png'):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_name: Filename to save plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_name='pr_curve.png'):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_name: Filename to save plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    def evaluate_model(self, model_name, y_true, y_pred, y_pred_proba=None,
                      dataset_name='Validation', save_plots=True):
        """
        Complete evaluation of a model
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            dataset_name: Name of the dataset
            save_plots: Whether to save plots
            
        Returns:
            Dictionary of metrics
        """
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Print metrics
        self.print_metrics(metrics, dataset_name)
        
        # Save plots
        if save_plots:
            prefix = f"{model_name}_{dataset_name.lower()}_"
            
            # Confusion matrix
            self.plot_confusion_matrix(metrics['confusion_matrix'], 
                                      save_name=f"{prefix}confusion_matrix.png")
            
            # ROC curve
            if y_pred_proba is not None:
                self.plot_roc_curve(y_true, y_pred_proba, 
                                   save_name=f"{prefix}roc_curve.png")
                self.plot_precision_recall_curve(y_true, y_pred_proba,
                                                save_name=f"{prefix}pr_curve.png")
        
        # Store results
        key = f"{model_name}_{dataset_name}"
        self.results[key] = metrics
        
        return metrics
        
    def compare_models(self, save_name='model_comparison.png'):
        """
        Create comparison plot for multiple models
        
        Args:
            save_name: Filename to save comparison plot
        """
        if len(self.results) == 0:
            print("No results to compare")
            return
        
        # Extract metrics for comparison
        comparison_data = []
        for key, metrics in self.results.items():
            model_name, dataset = key.rsplit('_', 1)
            comparison_data.append({
                'Model': model_name,
                'Dataset': dataset,
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['auc'] if metrics['auc'] is not None else 0,
                'F1-Score': metrics['f1_score']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics_to_plot = ['Accuracy', 'AUC', 'F1-Score']
        for idx, metric in enumerate(metrics_to_plot):
            pivot = df.pivot(index='Model', columns='Dataset', values=metric)
            pivot.plot(kind='bar', ax=axes[idx], rot=45)
            axes[idx].set_title(metric)
            axes[idx].set_ylabel('Score')
            axes[idx].set_ylim([0, 1])
            axes[idx].legend(title='Dataset')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nModel comparison plot saved to: {self.output_dir / save_name}")
        
    def save_results_to_csv(self, save_name='evaluation_results.csv'):
        """
        Save all results to CSV
        
        Args:
            save_name: Filename to save results
        """
        if len(self.results) == 0:
            print("No results to save")
            return
        
        results_list = []
        for key, metrics in self.results.items():
            model_name, dataset = key.rsplit('_', 1)
            results_list.append({
                'Model': model_name,
                'Dataset': dataset,
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['auc'] if metrics['auc'] is not None else None,
                'F1-Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
        
        df = pd.DataFrame(results_list)
        df.to_csv(self.output_dir / save_name, index=False)
        
        print(f"\nResults saved to: {self.output_dir / save_name}")
        
        return df
        
    def generate_summary_report(self, save_name='evaluation_summary.txt'):
        """
        Generate a text summary report of all evaluations
        
        Args:
            save_name: Filename to save report
        """
        if len(self.results) == 0:
            print("No results to summarize")
            return
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MODEL EVALUATION SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        for key, metrics in self.results.items():
            model_name, dataset = key.rsplit('_', 1)
            report_lines.append(f"{model_name} - {dataset}")
            report_lines.append("-"*80)
            report_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
            if metrics['auc'] is not None:
                report_lines.append(f"  AUC:       {metrics['auc']:.4f}")
            report_lines.append(f"  F1-Score:  {metrics['f1_score']:.4f}")
            report_lines.append(f"  Precision: {metrics['precision']:.4f}")
            report_lines.append(f"  Recall:    {metrics['recall']:.4f}")
            report_lines.append("")
        
        # Find best model
        val_results = {k: v for k, v in self.results.items() if 'Validation' in k}
        if val_results:
            best_model = max(val_results.items(), 
                           key=lambda x: (x[1]['f1_score'], x[1]['accuracy']))
            report_lines.append("="*80)
            report_lines.append("BEST MODEL (by F1-Score)")
            report_lines.append("="*80)
            report_lines.append(f"Model: {best_model[0]}")
            report_lines.append(f"F1-Score: {best_model[1]['f1_score']:.4f}")
            report_lines.append(f"Accuracy: {best_model[1]['accuracy']:.4f}")
            if best_model[1]['auc'] is not None:
                report_lines.append(f"AUC: {best_model[1]['auc']:.4f}")
        
        report_text = "\n".join(report_lines)
        
        # Print report
        print("\n" + report_text)
        
        # Save report
        with open(self.output_dir / save_name, 'w') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: {self.output_dir / save_name}")


if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics Module")
    print("This module provides comprehensive model evaluation capabilities.")
    print("\nKey features:")
    print("  - Calculate Accuracy, AUC, F1-Score, Precision, Recall")
    print("  - Generate confusion matrices and ROC curves")
    print("  - Compare multiple models")
    print("  - Export results to CSV and text reports")

