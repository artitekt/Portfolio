"""Model registry for AI Research Platform."""

import inspect
from typing import Dict, Type, Any, List, Optional
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


def filter_model_params(model_class, params):
    """Filter parameters to only include those accepted by the model constructor."""
    sig = inspect.signature(model_class.__init__)
    return {k: v for k, v in params.items() if k in sig.parameters}


class BaseModel:
    """Base class for all models in the registry."""
    
    def __init__(self, **kwargs):
        """Initialize model with parameters."""
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        self.hyperparameters = kwargs
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Training metrics
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature array
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "hyperparameters": self.hyperparameters,
            "training_metrics": self.training_metrics
        }


class LinearRegressionModel(BaseModel):
    """Linear Regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Filter parameters to only include those accepted by LinearRegression
        filtered_params = filter_model_params(LinearRegression, kwargs)
        self.model = LinearRegression(**filtered_params)
    
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        return {
            "training_score": self.model.score(X, y),
            "coefficients": self.model.coef_.tolist() if hasattr(self.model, 'coef_') else None,
            "intercept": float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None
        }
    
    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def get_model_info(self):
        return {
            "model_type": "LinearRegression",
            "hyperparameters": self.hyperparameters,
            "is_trained": self.is_trained
        }


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model wrapper."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default parameters for classification
        default_params = {
            'random_state': 42,
            'max_iter': 1000
        }
        # Merge with provided params and filter
        all_params = {**default_params, **kwargs}
        filtered_params = filter_model_params(LogisticRegression, all_params)
        self.model = LogisticRegression(**filtered_params)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train logistic regression model."""
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        self.training_metrics = {
            "train_accuracy": float(accuracy),
            "train_precision": float(precision),
            "train_recall": float(recall),
            "train_f1": float(f1),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y))
        }
        
        logger.info(f"Logistic Regression trained - Accuracy: {self.training_metrics['train_accuracy']:.4f}")
        return self.training_metrics


class RandomForestRegressionModel(BaseModel):
    """Random Forest Regression model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'random_state': 42
        }
        # Merge with provided params and filter
        all_params = {**default_params, **kwargs}
        filtered_params = filter_model_params(RandomForestRegressor, all_params)
        super().__init__(**all_params)
        self.model = RandomForestRegressor(**filtered_params)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train random forest regression model."""
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        self.training_metrics = {
            "train_mse": float(mse),
            "train_rmse": float(np.sqrt(mse)),
            "train_r2": float(self.model.score(X, y)),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_estimators": self.model.n_estimators
        }
        
        logger.info(f"Random Forest Regression trained - R²: {self.training_metrics['train_r2']:.4f}")
        return self.training_metrics
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_


class RandomForestClassificationModel(BaseModel):
    """Random Forest Classification model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.model = RandomForestClassifier(**default_params)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train random forest classification model."""
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        self.training_metrics = {
            "train_accuracy": float(accuracy),
            "train_precision": float(precision),
            "train_recall": float(recall),
            "train_f1": float(f1),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_estimators": self.model.n_estimators,
            "n_classes": len(np.unique(y))
        }
        
        logger.info(f"Random Forest Classification trained - Accuracy: {self.training_metrics['train_accuracy']:.4f}")
        return self.training_metrics
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_


class GradientBoostingRegressionModel(BaseModel):
    """Gradient Boosting Regression model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.model = GradientBoostingRegressor(**default_params)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train gradient boosting regression model."""
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        self.training_metrics = {
            "train_mse": float(mse),
            "train_rmse": float(np.sqrt(mse)),
            "train_r2": float(self.model.score(X, y)),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_estimators": self.model.n_estimators
        }
        
        logger.info(f"Gradient Boosting Regression trained - R²: {self.training_metrics['train_r2']:.4f}")
        return self.training_metrics


class GradientBoostingClassificationModel(BaseModel):
    """Gradient Boosting Classification model wrapper."""
    
    def __init__(self, **kwargs):
        # Set default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(kwargs)
        super().__init__(**default_params)
        self.model = GradientBoostingClassifier(**default_params)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train gradient boosting classification model."""
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        
        self.training_metrics = {
            "train_accuracy": float(accuracy),
            "train_precision": float(precision),
            "train_recall": float(recall),
            "train_f1": float(f1),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_estimators": self.model.n_estimators,
            "n_classes": len(np.unique(y))
        }
        
        logger.info(f"Gradient Boosting Classification trained - Accuracy: {self.training_metrics['train_accuracy']:.4f}")
        return self.training_metrics


class ModelRegistry:
    """Registry of available ML models."""
    
    # Registry of available models
    _models = {
        # Regression models
        "linear_regression": LinearRegressionModel,
        "random_forest_regression": RandomForestRegressionModel,
        "gradient_boosting_regression": GradientBoostingRegressionModel,
        
        # Classification models
        "logistic_regression": LogisticRegressionModel,
        "random_forest_classification": RandomForestClassificationModel,
        "gradient_boosting_classification": GradientBoostingClassificationModel,
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get a dictionary of available models."""
        return cls._models.copy()
    
    @classmethod
    def get_regression_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get regression models only."""
        return {
            name: model_class for name, model_class in cls._models.items()
            if "regression" in name.lower()
        }
    
    @classmethod
    def get_classification_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get classification models only."""
        return {
            name: model_class for name, model_class in cls._models.items()
            if "classification" in name.lower() or "logistic" in name.lower()
        }
    
    @classmethod
    def get_model_by_name(cls, model_name: str) -> Type[BaseModel]:
        """Get a model class by name."""
        if model_name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
        return cls._models[model_name]
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> BaseModel:
        """Create a model instance."""
        model_class = cls.get_model_by_name(model_name)
        return model_class(**kwargs)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """Register a new model class."""
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must inherit from BaseModel")
        cls._models[name] = model_class
        logger.info(f"Registered new model: {name}")
    
    @classmethod
    def get_model_recommendation(cls, task_type: str, n_samples: int, n_features: int) -> List[str]:
        """
        Get model recommendations based on task characteristics.
        
        Args:
            task_type: 'regression' or 'classification'
            n_samples: Number of samples
            n_features: Number of features
            
        Returns:
            List of recommended model names
        """
        recommendations = []
        
        if task_type == "regression":
            if n_samples < 1000:
                recommendations.extend(["linear_regression", "random_forest_regression"])
            else:
                recommendations.extend(["random_forest_regression", "gradient_boosting_regression"])
        elif task_type == "classification":
            if n_samples < 1000:
                recommendations.extend(["logistic_regression", "random_forest_classification"])
            else:
                recommendations.extend(["random_forest_classification", "gradient_boosting_classification"])
        
        return recommendations
