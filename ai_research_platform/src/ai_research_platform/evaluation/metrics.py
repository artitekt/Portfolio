"""Evaluation metrics for AI Research Platform."""

import numpy as np
from typing import Dict, Any, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """Calculate various evaluation metrics for ML models."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        metrics['per_class_metrics'] = {}
        
        for i, label in enumerate(unique_labels):
            if i < len(precision_per_class):
                metrics['per_class_metrics'][int(label) if np.issubdtype(label.dtype, np.integer) else str(label)] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'support': int(support[i])
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics
        metrics['n_samples'] = len(y_true)
        metrics['n_classes'] = len(unique_labels)
        
        logger.info(f"Classification metrics calculated - Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of regression metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['r2'] = float(r2_score(y_true, y_pred))
        
        # Additional metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = float(np.mean(residuals))
        metrics['std_residual'] = float(np.std(residuals))
        
        # Relative metrics
        if np.any(y_true != 0):
            metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100)
        else:
            metrics['mape'] = float('inf')
        
        # Target statistics
        metrics['target_mean'] = float(np.mean(y_true))
        metrics['target_std'] = float(np.std(y_true))
        metrics['prediction_mean'] = float(np.mean(y_pred))
        metrics['prediction_std'] = float(np.std(y_pred))
        
        # Sample info
        metrics['n_samples'] = len(y_true)
        
        logger.info(f"Regression metrics calculated - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        return metrics
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics based on task type.
        
        Args:
            y_true: True values/labels
            y_pred: Predicted values/labels
            task_type: 'classification' or 'regression'
            y_prob: Predicted probabilities (for classification)
            
        Returns:
            Dictionary of metrics
        """
        if task_type == "classification":
            return MetricsCalculator.calculate_classification_metrics(y_true, y_pred, y_prob)
        elif task_type == "regression":
            return MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @staticmethod
    def compare_models(
        model_results: Dict[str, Dict[str, Any]],
        metric: str,
        task_type: str
    ) -> Dict[str, Any]:
        """
        Compare multiple models on a specific metric.
        
        Args:
            model_results: Dictionary of model results
            metric: Metric to compare
            task_type: 'classification' or 'regression'
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for model_name, results in model_results.items():
            if metric in results.get('metrics', {}):
                comparison[model_name] = results['metrics'][metric]
            else:
                logger.warning(f"Metric '{metric}' not found for model '{model_name}'")
        
        # Sort by metric value
        if comparison:
            # For regression, higher is better for R2, lower is better for errors
            # For classification, higher is better for accuracy, precision, recall, f1
            if task_type == "regression":
                if metric in ["r2"]:
                    sorted_models = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
                else:  # mse, rmse, mae
                    sorted_models = sorted(comparison.items(), key=lambda x: x[1])
            else:  # classification
                sorted_models = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "rankings": dict(sorted_models),
                "best_model": sorted_models[0][0] if sorted_models else None,
                "best_score": sorted_models[0][1] if sorted_models else None
            }
        
        return {"rankings": {}, "best_model": None, "best_score": None}
    
    @staticmethod
    def get_metric_summary(metrics: Dict[str, Any], task_type: str) -> str:
        """
        Get a formatted summary of metrics.
        
        Args:
            metrics: Metrics dictionary
            task_type: 'classification' or 'regression'
            
        Returns:
            Formatted summary string
        """
        if task_type == "classification":
            return (
                f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                f"Precision: {metrics.get('precision', 0):.4f}\n"
                f"Recall: {metrics.get('recall', 0):.4f}\n"
                f"F1 Score: {metrics.get('f1', 0):.4f}"
            )
        else:  # regression
            return (
                f"R²: {metrics.get('r2', 0):.4f}\n"
                f"RMSE: {metrics.get('rmse', 0):.4f}\n"
                f"MAE: {metrics.get('mae', 0):.4f}\n"
                f"MSE: {metrics.get('mse', 0):.4f}"
            )
