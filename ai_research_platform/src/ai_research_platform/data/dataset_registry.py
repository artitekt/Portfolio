"""Dataset registry for managing available datasets."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetRegistry:
    """Registry for managing datasets and their metadata."""
    
    def __init__(self, registry_path: str = "results/datasets/registry.json"):
        """
        Initialize dataset registry.
        
        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load dataset registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading registry: {e}")
                return {"datasets": {}, "created_at": datetime.now().isoformat()}
        else:
            return {"datasets": {}, "created_at": datetime.now().isoformat()}
    
    def _save_registry(self):
        """Save dataset registry to file."""
        try:
            self._registry["updated_at"] = datetime.now().isoformat()
            with open(self.registry_path, 'w') as f:
                json.dump(self._registry, f, indent=2)
            logger.info(f"Registry saved to {self.registry_path}")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise
    
    def register_dataset(
        self,
        name: str,
        path: str,
        metadata: Dict[str, Any],
        overwrite: bool = False
    ) -> bool:
        """
        Register a dataset in the registry.
        
        Args:
            name: Dataset name
            path: Path to dataset file
            metadata: Dataset metadata
            overwrite: Whether to overwrite existing dataset
            
        Returns:
            True if registration successful
        """
        if name in self._registry["datasets"] and not overwrite:
            logger.warning(f"Dataset '{name}' already exists. Use overwrite=True to replace.")
            return False
        
        # Validate dataset exists
        dataset_path = Path(path)
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {path}")
            return False
        
        # Prepare dataset entry
        dataset_entry = {
            "name": name,
            "path": str(dataset_path.absolute()),
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # Add required metadata fields
        required_fields = ["rows", "features", "task"]
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Missing required metadata field: {field}")
        
        # Add optional metadata fields with defaults
        default_metadata = {
            "description": "",
            "target_column": None,
            "feature_columns": None,
            "categorical_features": [],
            "numerical_features": [],
            "missing_values": False,
            "size_mb": 0.0,
            "format": "csv"
        }
        
        for key, default_value in default_metadata.items():
            if key not in metadata:
                metadata[key] = default_value
        
        # Calculate file size if not provided
        if metadata.get("size_mb", 0) == 0 and dataset_path.exists():
            metadata["size_mb"] = dataset_path.stat().st_size / (1024 * 1024)
        
        # Save to registry
        self._registry["datasets"][name] = dataset_entry
        self._save_registry()
        
        logger.info(f"Dataset '{name}' registered successfully")
        return True
    
    def list_datasets(self, task_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        List all registered datasets.
        
        Args:
            task_type: Filter by task type (classification, regression, etc.)
            
        Returns:
            Dictionary of datasets
        """
        datasets = self._registry["datasets"].copy()
        
        if task_type:
            filtered_datasets = {}
            for name, info in datasets.items():
                if info["metadata"].get("task") == task_type:
                    filtered_datasets[name] = info
            datasets = filtered_datasets
        
        return datasets
    
    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset information or None if not found
        """
        return self._registry["datasets"].get(name)
    
    def load_dataset(self, name: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Load a registered dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        dataset_info = self.get_dataset_info(name)
        if not dataset_info:
            raise ValueError(f"Dataset '{name}' not found in registry")
        
        path = dataset_info["path"]
        metadata = dataset_info["metadata"]
        
        try:
            # Load dataset
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith('.json'):
                df = pd.read_json(path)
            elif path.endswith('.parquet'):
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            # Extract features and target
            target_column = metadata.get("target_column")
            feature_columns = metadata.get("feature_columns")
            
            if target_column and target_column in df.columns:
                y = df[target_column]
                
                if feature_columns:
                    X = df[feature_columns]
                else:
                    X = df.drop(columns=[target_column])
            else:
                X = df
                y = None
            
            logger.info(f"Loaded dataset '{name}': {X.shape[0]} rows, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading dataset '{name}': {e}")
            raise
    
    def unregister_dataset(self, name: str) -> bool:
        """
        Remove a dataset from the registry.
        
        Args:
            name: Dataset name
            
        Returns:
            True if removal successful
        """
        if name not in self._registry["datasets"]:
            logger.warning(f"Dataset '{name}' not found in registry")
            return False
        
        del self._registry["datasets"][name]
        self._save_registry()
        
        logger.info(f"Dataset '{name}' removed from registry")
        return True
    
    def update_dataset_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a registered dataset.
        
        Args:
            name: Dataset name
            metadata: New metadata to merge
            
        Returns:
            True if update successful
        """
        if name not in self._registry["datasets"]:
            logger.warning(f"Dataset '{name}' not found in registry")
            return False
        
        # Merge metadata
        current_metadata = self._registry["datasets"][name]["metadata"]
        current_metadata.update(metadata)
        
        self._save_registry()
        logger.info(f"Metadata updated for dataset '{name}'")
        return True
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about registered datasets.
        
        Returns:
            Dictionary with dataset statistics
        """
        datasets = self._registry["datasets"]
        
        if not datasets:
            return {
                "total_datasets": 0,
                "by_task": {},
                "total_size_mb": 0.0,
                "total_rows": 0
            }
        
        stats = {
            "total_datasets": len(datasets),
            "by_task": {},
            "total_size_mb": 0.0,
            "total_rows": 0
        }
        
        for name, info in datasets.items():
            metadata = info["metadata"]
            
            # Count by task type
            task = metadata.get("task", "unknown")
            stats["by_task"][task] = stats["by_task"].get(task, 0) + 1
            
            # Sum sizes
            stats["total_size_mb"] += metadata.get("size_mb", 0)
            stats["total_rows"] += metadata.get("rows", 0)
        
        return stats
    
    def export_registry(self, output_path: str, format: str = "json"):
        """
        Export registry to file.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv)
        """
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(self._registry, f, indent=2)
            elif format.lower() == "csv":
                # Convert to CSV format
                rows = []
                for name, info in self._registry["datasets"].items():
                    row = {
                        "name": name,
                        "path": info["path"],
                        "registered_at": info["registered_at"]
                    }
                    row.update(info["metadata"])
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Registry exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
            raise
    
    def validate_dataset(self, name: str) -> Dict[str, Any]:
        """
        Validate a registered dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Validation results
        """
        dataset_info = self.get_dataset_info(name)
        if not dataset_info:
            return {"valid": False, "error": "Dataset not found"}
        
        path = dataset_info["path"]
        metadata = dataset_info["metadata"]
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check file exists
            if not Path(path).exists():
                validation_results["valid"] = False
                validation_results["errors"].append("Dataset file not found")
                return validation_results
            
            # Load and validate dataset
            X, y = self.load_dataset(name)
            
            # Validate dimensions
            expected_rows = metadata.get("rows")
            expected_features = metadata.get("features")
            
            if expected_rows and X.shape[0] != expected_rows:
                validation_results["warnings"].append(
                    f"Row count mismatch: expected {expected_rows}, got {X.shape[0]}"
                )
            
            if expected_features and X.shape[1] != expected_features:
                validation_results["warnings"].append(
                    f"Feature count mismatch: expected {expected_features}, got {X.shape[1]}"
                )
            
            # Check for missing values
            if X.isnull().any().any():
                has_missing = metadata.get("missing_values", False)
                if not has_missing:
                    validation_results["warnings"].append("Dataset contains missing values")
            
            # Validate target column
            target_column = metadata.get("target_column")
            if target_column and y is not None:
                if y.isnull().any():
                    validation_results["errors"].append("Target column contains missing values")
                    validation_results["valid"] = False
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {e}")
        
        return validation_results
