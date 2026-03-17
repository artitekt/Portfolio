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
        """Evaluate a trained model."""
        logger.info(f"Evaluating model on {dataset_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            task_type=task_type
        )
        
        # Create result dictionary
        result = {
            "model_name": model.__class__.__name__,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "metrics": metrics,
            "model_info": model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }
        
        logger.info(f"Model evaluation completed")
        return result
