"""Dataset loading utilities for AI Research Platform."""

import numpy as np
import pandas as pd
import inspect
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ai_research_platform.data.dataset_generator import DatasetGenerator
from ai_research_platform.utils.logger import get_logger
from ai_research_platform.utils.config import config

logger = get_logger(__name__)


def filter_params(func, params):
    """Filter parameters to only include those accepted by the function."""
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters}


class DatasetLoader:
    """Load and prepare datasets for ML experiments."""
    
    def __init__(self):
        """Initialize dataset loader."""
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.target_name = None
    
    def load_synthetic(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load a synthetic dataset.
        
        Args:
            **kwargs: Dataset-specific parameters
            
        Returns:
            Tuple of (X, y, metadata)
        """
        logger.info(f"Loading synthetic dataset: {kwargs.get('dataset_type', 'unknown')}")
        
        generator = DatasetGenerator()
        
        if kwargs.get("dataset_type") == "classification":
            # Filter parameters for classification
            X, y, metadata = generator.generate_classification(**filter_params(generator.generate_classification, kwargs))
        elif kwargs.get("dataset_type") == "regression":
            # Filter parameters for regression
            X, y, metadata = generator.generate_regression(**filter_params(generator.generate_regression, kwargs))
        elif kwargs.get("dataset_type") == "timeseries":
            # Filter parameters for timeseries
            X, y, metadata = generator.generate_timeseries(**filter_params(generator.generate_timeseries, kwargs))
        elif kwargs.get("dataset_type") == "friedman":
            # Filter parameters for friedman
            X, y, metadata = generator.generate_friedman(**filter_params(generator.generate_friedman, kwargs))
        else:
            raise ValueError(f"Unknown dataset type: {kwargs.get('dataset_type')}")
        
        self.feature_names = metadata.get("feature_names", None)
        return X, y, metadata
    
    def load_csv(
        self,
        file_path: Union[str, Path],
        target_column: str,
        feature_columns: Optional[list] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            feature_columns: List of feature columns (default: all except target)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Tuple of (X, y, metadata)
        """
        logger.info(f"Loading CSV dataset from: {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load data
        data = pd.read_csv(file_path, **kwargs)
        
        # Validate target column
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Validate feature columns
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
        
        # Extract features and target
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Store metadata
        metadata = {
            "dataset_type": "csv",
            "source_file": str(file_path),
            "n_samples": len(X),
            "n_features": len(feature_columns),
            "feature_names": feature_columns,
            "target_name": target_column,
            "target_type": str(y.dtype),
            "missing_values": data.isnull().sum().sum()
        }
        
        # Add target statistics
        if np.issubdtype(y.dtype, np.number):
            metadata.update({
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
                "target_range": [float(np.min(y)), float(np.max(y))]
            })
        else:
            metadata.update({
                "target_classes": dict(zip(*np.unique(y, return_counts=True)))
            })
        
        self.feature_names = feature_columns
        self.target_name = target_column
        
        logger.info(f"Loaded dataset: {metadata['n_samples']} samples, {metadata['n_features']} features")
        return X, y, metadata
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        scale_features: bool = True,
        encode_labels: bool = True,
        random_state: Optional[int] = None,
        task_type: str = "classification"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for ML training.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of test data (default: from config)
            scale_features: Whether to scale features
            encode_labels: Whether to encode categorical labels
            random_state: Random seed
            task_type: "classification" or "regression"
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for ML training")
        
        # Use config defaults
        test_size = test_size or config.get("experiments.default_test_size", 0.2)
        random_state = random_state or config.get("experiments.random_state", 42)
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found missing values in features, filling with mean")
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        if np.isnan(y).any():
            logger.warning("Found missing values in target, removing those samples")
            valid_mask = ~np.isnan(y)
            X, y = X[valid_mask], y[valid_mask]
        
        # Encode categorical labels if needed
        if encode_labels and not np.issubdtype(y.dtype, np.number):
            logger.info("Encoding categorical labels")
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        
        # Scale features if needed
        if scale_features:
            logger.info("Scaling features")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Split data - only stratify for classification with discrete labels
        stratify_param = y if task_type == "classification" and len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> Optional[list]:
        """Get feature names."""
        return self.feature_names
    
    def get_target_name(self) -> Optional[str]:
        """Get target name."""
        return self.target_name
    
    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform encoded labels.
        
        Args:
            y: Encoded labels
            
        Returns:
            Original labels
        """
        if self.label_encoder is None:
            return y
        return self.label_encoder.inverse_transform(y)
    
    def get_label_classes(self) -> Optional[np.ndarray]:
        """Get original label classes."""
        if self.label_encoder is None:
            return None
        return self.label_encoder.classes_
