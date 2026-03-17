"""Baseline models for AI Research Platform."""

import numpy as np
from typing import Dict, Any
from ai_research_platform.models.model_registry import ModelRegistry
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineMeanModel(BaseModel):
    """Baseline model that predicts the mean of training targets."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_value = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train baseline mean model."""
        self.mean_value = np.mean(y)
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        self.training_metrics = {
            "train_mse": float(mse),
            "train_rmse": float(np.sqrt(mse)),
            "mean_prediction": float(self.mean_value),
            "n_samples": len(X)
        }
        
        logger.info(f"Baseline Mean Model trained - Mean: {self.mean_value:.4f}")
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using mean value."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return np.full(len(X), self.mean_value)


class BaselineMedianModel(BaseModel):
    """Baseline model that predicts the median of training targets."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.median_value = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train baseline median model."""
        self.median_value = np.median(y)
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        
        self.training_metrics = {
            "train_mse": float(mse),
            "train_rmse": float(np.sqrt(mse)),
            "median_prediction": float(self.median_value),
            "n_samples": len(X)
        }
        
        logger.info(f"Baseline Median Model trained - Median: {self.median_value:.4f}")
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using median value."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return np.full(len(X), self.median_value)


class BaselineMajorityClassModel(BaseModel):
    """Baseline model that predicts the majority class."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.majority_class = None
        self.classes_ = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train baseline majority class model."""
        unique_classes, counts = np.unique(y, return_counts=True)
        self.majority_class = unique_classes[np.argmax(counts)]
        self.classes_ = unique_classes
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        
        self.training_metrics = {
            "train_accuracy": float(accuracy),
            "majority_class": int(self.majority_class) if np.issubdtype(self.majority_class.dtype, np.integer) else str(self.majority_class),
            "class_distribution": dict(zip(unique_classes.tolist(), counts.tolist())),
            "n_samples": len(X),
            "n_classes": len(unique_classes)
        }
        
        logger.info(f"Baseline Majority Class Model trained - Class: {self.majority_class}, Accuracy: {accuracy:.4f}")
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority class."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return np.full(len(X), self.majority_class)


class BaselineRandomModel(BaseModel):
    """Baseline model that makes random predictions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.random_state = kwargs.get('random_state', 42)
        self.classes_ = None
        self.task_type = None  # 'regression' or 'classification'
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train baseline random model."""
        np.random.seed(self.random_state)
        
        # Determine task type
        if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10:
            self.task_type = "regression"
            self.min_val = np.min(y)
            self.max_val = np.max(y)
        else:
            self.task_type = "classification"
            self.classes_ = np.unique(y)
        
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.predict(X)
        
        if self.task_type == "regression":
            mse = np.mean((y - y_pred) ** 2)
            self.training_metrics = {
                "train_mse": float(mse),
                "train_rmse": float(np.sqrt(mse)),
                "task_type": "regression",
                "value_range": [float(self.min_val), float(self.max_val)],
                "n_samples": len(X)
            }
        else:
            accuracy = np.mean(y == y_pred)
            self.training_metrics = {
                "train_accuracy": float(accuracy),
                "task_type": "classification",
                "n_classes": len(self.classes_),
                "n_samples": len(X)
            }
        
        logger.info(f"Baseline Random Model trained - Type: {self.task_type}")
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make random predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.task_type == "regression":
            return np.random.uniform(self.min_val, self.max_val, len(X))
        else:
            return np.random.choice(self.classes_, len(X))
