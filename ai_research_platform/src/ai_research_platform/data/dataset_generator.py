"""Dataset generation utilities for AI Research Platform."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.datasets import make_classification, make_regression, make_friedman1
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetGenerator:
    """Generate synthetic datasets for ML experiments."""
    
    @staticmethod
    def generate_classification(
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: int = 10,
        n_classes: int = 2,
        random_state: int = 42,
        flip_y: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic classification dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features
            n_classes: Number of classes
            random_state: Random seed
            flip_y: Fraction of samples whose class is randomly flipped
            
        Returns:
            Tuple of (X, y, metadata)
        """
        logger.info(f"Generating classification dataset: {n_samples} samples, {n_features} features")
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            random_state=random_state,
            flip_y=flip_y
        )
        
        metadata = {
            "dataset_type": "classification",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "n_classes": n_classes,
            "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
            "feature_names": [f"feature_{i}" for i in range(n_features)]
        }
        
        logger.info(f"Generated dataset with class distribution: {metadata['class_distribution']}")
        return X, y, metadata
    
    @staticmethod
    def generate_regression(
        n_samples: int = 1000,
        n_features: int = 10,
        n_informative: int = 5,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic regression dataset.
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            n_informative: Number of informative features
            noise: Noise level
            random_state: Random seed
            
        Returns:
            Tuple of (X, y, metadata)
        """
        logger.info(f"Generating regression dataset: {n_samples} samples, {n_features} features")
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=random_state
        )
        
        metadata = {
            "dataset_type": "regression",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative": n_informative,
            "target_mean": np.mean(y),
            "target_std": np.std(y),
            "target_range": [float(np.min(y)), float(np.max(y))],
            "feature_names": [f"feature_{i}" for i in range(n_features)]
        }
        
        logger.info(f"Generated dataset with target range: {metadata['target_range']}")
        return X, y, metadata
    
    @staticmethod
    def generate_timeseries(
        n_samples: int = 1000,
        n_features: int = 5,
        trend: float = 0.01,
        seasonality: bool = True,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic time series dataset.
        
        Args:
            n_samples: Number of time steps
            n_features: Number of features
            trend: Trend component strength
            seasonality: Whether to include seasonal component
            noise: Noise level
            random_state: Random seed
            
        Returns:
            Tuple of (X, y, metadata)
        """
        logger.info(f"Generating time series dataset: {n_samples} timesteps, {n_features} features")
        
        np.random.seed(random_state)
        
        # Generate time index
        time = np.arange(n_samples)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with trend and seasonality
        y = trend * time + noise * np.random.randn(n_samples)
        
        if seasonality:
            # Add seasonal component
            seasonal_period = n_samples // 4
            y += 2 * np.sin(2 * np.pi * time / seasonal_period)
        
        # Add some relationship with features
        for i in range(min(n_features, 3)):
            y += 0.5 * X[:, i] * (i + 1)
        
        metadata = {
            "dataset_type": "timeseries",
            "n_samples": n_samples,
            "n_features": n_features,
            "trend": trend,
            "seasonality": seasonality,
            "target_mean": np.mean(y),
            "target_std": np.std(y),
            "feature_names": [f"feature_{i}" for i in range(n_features)]
        }
        
        logger.info(f"Generated time series with trend {trend} and seasonality {seasonality}")
        return X, y, metadata
    
    @staticmethod
    def generate_friedman(
        n_samples: int = 1000,
        n_features: int = 10,
        noise: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate Friedman dataset (non-linear regression).
        
        Args:
            n_samples: Number of samples
            n_features: Total number of features
            noise: Noise level
            random_state: Random seed
            
        Returns:
            Tuple of (X, y, metadata)
        """
        logger.info(f"Generating Friedman dataset: {n_samples} samples, {n_features} features")
        
        X, y = make_friedman1(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        
        metadata = {
            "dataset_type": "friedman_nonlinear",
            "n_samples": n_samples,
            "n_features": n_features,
            "target_mean": np.mean(y),
            "target_std": np.std(y),
            "feature_names": [f"feature_{i}" for i in range(n_features)]
        }
        
        logger.info(f"Generated Friedman dataset with target range: [{np.min(y):.3f}, {np.max(y):.3f}]")
        return X, y, metadata
    
    @classmethod
    def get_available_datasets(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available dataset types.
        
        Returns:
            Dictionary of dataset types and their descriptions
        """
        return {
            "classification": {
                "description": "Binary or multi-class classification dataset",
                "parameters": ["n_samples", "n_features", "n_informative", "n_classes", "noise"],
                "target_type": "categorical"
            },
            "regression": {
                "description": "Linear regression dataset",
                "parameters": ["n_samples", "n_features", "n_informative", "noise"],
                "target_type": "continuous"
            },
            "timeseries": {
                "description": "Time series dataset with trend and seasonality",
                "parameters": ["n_samples", "n_features", "trend", "seasonality", "noise"],
                "target_type": "continuous"
            },
            "friedman": {
                "description": "Non-linear regression dataset (Friedman function)",
                "parameters": ["n_samples", "n_features", "noise"],
                "target_type": "continuous"
            }
        }
