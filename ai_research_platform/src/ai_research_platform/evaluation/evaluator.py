"""Model evaluator for AI Research Platform."""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ai_research_platform.models.model_registry import ModelRegistry, BaseModel
from ai_research_platform.utils.logger import get_logger
from ai_research_platform.evaluation.metrics import MetricsCalculator

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate ML models and compare performance."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_model(
        self,
        model: BaseModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        dataset_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test targets
            task_type: 'classification' or 'regression'
            dataset_name: Name of dataset
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model on {dataset_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities for classification if available
        y_prob = None
        if task_type == "classification" and hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                pass  # Some models don't support predict_proba
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            task_type=task_type,
            y_prob=y_prob
        )
        
        # Create result dictionary
        result = {
            "model_name": model.__class__.__name__,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "metrics": metrics,
            "predictions": y_pred.tolist() if len(y_pred) < 1000 else None,  # Limit size
            "true_values": y_test.tolist() if len(y_test) < 1000 else None,
            "model_info": model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }
        
        logger.info(f"Model evaluation completed - {task_type} accuracy: {metrics.get('accuracy', metrics.get('r2', 0)):.4f}")
        
        return result
    
    def cross_validate_model(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        cv_folds: int = 5,
        **model_params
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_class: Model class to evaluate
            X: Features
            y: Targets
            task_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            **model_params: Parameters for model
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Choose cross-validation strategy
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_class(**model_params)
            model.train(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            
            if task_type == "classification":
                score = accuracy_score(y_val, y_pred)
            else:
                score = mean_squared_error(y_val, y_pred)
            
            fold_scores.append(score)
            
            # Calculate detailed metrics
            fold_metric = self.metrics_calculator.calculate_metrics(
                y_true=y_val,
                y_pred=y_pred,
                task_type=task_type
            )
            fold_metrics.append(fold_metric)
        
        # Calculate mean and std
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        # Aggregate metrics
        aggregated_metrics = {}
        if fold_metrics:
            for key in fold_metrics[0].keys():
                values = [m[key] for m in fold_metrics if key in m and isinstance(m[key], (int, float))]
                if values:
                    aggregated_metrics[key] = {
                        "mean": np.mean(values),
                        "std": np.std(values)
                    }
        
        result = {
            "model_class": model_class.__name__,
            "task_type": task_type,
            "cv_folds": cv_folds,
            "mean_score": mean_score,
            "std_score": std_score,
            "fold_scores": fold_scores,
            "aggregated_metrics": aggregated_metrics,
            "model_params": model_params
        }
        
        logger.info(f"Cross-validation completed - Mean score: {mean_score:.4f} (+/- {std_score:.4f})")
        
        return result
    
    def compare_models(
        self,
        models: List[BaseModel],
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str,
        dataset_name: str = "comparison"
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: List of trained models
            X_test: Test features
            y_test: Test targets
            task_type: 'classification' or 'regression'
            dataset_name: Name of dataset
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        results = {}
        model_names = []
        scores = []
        
        for model in models:
            model_name = model.__class__.__name__
            model_names.append(model_name)
            
            # Evaluate model
            result = self.evaluate_model(model, X_test, y_test, task_type, dataset_name)
            results[model_name] = result
            
            # Extract main score for ranking
            if task_type == "classification":
                score = result["metrics"].get("accuracy", 0)
            else:
                score = result["metrics"].get("r2", 0)
            scores.append(score)
        
        # Sort models by performance
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        
        comparison = {
            "dataset_name": dataset_name,
            "task_type": task_type,
            "models": results,
            "ranking": {
                model_names[i]: scores[i] for i in sorted_indices
            },
            "best_model": model_names[sorted_indices[0]],
            "best_score": scores[sorted_indices[0]]
        }
        
        logger.info(f"Model comparison completed - Best: {comparison['best_model']} ({comparison['best_score']:.4f})")
        
        return comparison