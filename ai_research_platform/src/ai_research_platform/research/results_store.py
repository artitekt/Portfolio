"""Results storage for AI Research Platform."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from ai_research_platform.utils.logger import get_logger
from ai_research_platform.utils.config import config


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        return super().default(obj)
    
    def encode(self, obj):
        """Override encode to handle dictionary keys."""
        # Convert the object first
        if isinstance(obj, dict):
            # Convert all keys and values
            converted = {}
            for k, v in obj.items():
                # Convert numpy keys to regular types
                if isinstance(k, (np.integer, np.floating)):
                    k = k.item() if hasattr(k, 'item') else int(k)
                # Convert values
                converted[k] = self.default(v)
            return converted
        return super().encode(obj)

logger = get_logger(__name__)


class ResultsStore:
    """Store and manage experiment results."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize results store.
        
        Args:
            storage_path: Path to store results
        """
        self.storage_path = Path(storage_path or config.get("results.storage_path", "results"))
        self.storage_path.mkdir(exist_ok=True)
        (self.storage_path / "experiments").mkdir(exist_ok=True)
        (self.storage_path / "models").mkdir(exist_ok=True)
        (self.storage_path / "datasets").mkdir(exist_ok=True)
        (self.storage_path / "reports").mkdir(exist_ok=True)
        (self.storage_path / "predictions").mkdir(exist_ok=True)
    
    def save_results(
        self,
        results: Dict[str, Any],
        experiment_name: str,
        format: str = "json"
    ) -> str:
        """
        Save experiment results to file.
        
        Args:
            results: Results dictionary
            experiment_name: Name of the experiment
            format: File format ("json" or "csv")
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"{experiment_name}_{timestamp}.json"
            filepath = self.storage_path / filename
            
            # Convert numpy types before saving
            converted_results = self._convert_numpy_types(results)
            
            with open(filepath, 'w') as f:
                json.dump(converted_results, f, indent=2)
        
        elif format == "csv":
            # Flatten results for CSV storage
            flattened = self._flatten_dict(results)
            df = pd.DataFrame([flattened])
            
            filename = f"{experiment_name}_{timestamp}.csv"
            filepath = self.storage_path / filename
            
            df.to_csv(filepath, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved results to: {filepath}")
        return str(filepath)
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Results dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        if filepath.suffix == ".json":
            with open(filepath, 'r') as f:
                results = json.load(f)
        
        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
            results = df.iloc[0].to_dict()
            # Convert back to nested structure if needed
            results = self._unflatten_dict(results)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded results from: {filepath}")
        return results
    
    def save_comparison_report(
        self,
        comparison_results: Dict[str, Any],
        experiment_name: str
    ) -> str:
        """
        Save model comparison report.
        
        Args:
            comparison_results: Comparison results dictionary
            experiment_name: Name of the experiment
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_comparison_{timestamp}.json"
        filepath = self.storage_path / "reports" / filename
        
        # Convert numpy types before saving
        converted_results = self._convert_numpy_types(comparison_results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        logger.info(f"Saved comparison report to: {filepath}")
        return str(filepath)
    
    def create_leaderboard(self, metric: str, task_type: str) -> pd.DataFrame:
        """
        Create a leaderboard from all experiments.
        
        Args:
            metric: Metric to rank by
            task_type: 'classification' or 'regression'
            
        Returns:
            Leaderboard DataFrame
        """
        leaderboard_data = []
        
        # Scan all result files
        for result_file in self.storage_path.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                # Extract relevant information
                if "metrics" in results:
                    metrics = results["metrics"]
                    if metric in metrics:
                        leaderboard_data.append({
                            "experiment_name": results.get("experiment_name", result_file.stem),
                            "model_name": results.get("model_name", "Unknown"),
                            "dataset_name": results.get("dataset_name", "Unknown"),
                            "task_type": results.get("task_type", task_type),
                            "metric_value": metrics[metric],
                            "timestamp": results.get("timestamp", datetime.now().isoformat())
                        })
                
                elif "model_results" in results:
                    # Comparison results
                    for model_name, model_result in results["model_results"].items():
                        if "metrics" in model_result and metric in model_result["metrics"]:
                            leaderboard_data.append({
                                "experiment_name": results.get("dataset_name", result_file.stem),
                                "model_name": model_name,
                                "dataset_name": results.get("dataset_name", "Unknown"),
                                "task_type": results.get("task_type", task_type),
                                "metric_value": model_result["metrics"][metric],
                                "timestamp": results.get("timestamp", datetime.now().isoformat())
                            })
            
            except Exception as e:
                logger.warning(f"Error processing result file {result_file}: {e}")
        
        if not leaderboard_data:
            logger.warning(f"No results found for metric: {metric}")
            return pd.DataFrame()
        
        # Create DataFrame and sort
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by metric (higher is better for most metrics except error metrics)
        error_metrics = ["mse", "rmse", "mae"]
        if metric.lower() in error_metrics:
            df = df.sort_values("metric_value", ascending=True)
        else:
            df = df.sort_values("metric_value", ascending=False)
        
        # Add rank
        df["rank"] = range(1, len(df) + 1)
        
        logger.info(f"Created leaderboard with {len(df)} entries for metric: {metric}")
        return df
    
    def save_leaderboard(self, leaderboard: pd.DataFrame, metric: str) -> str:
        """
        Save leaderboard to file.
        
        Args:
            leaderboard: Leaderboard DataFrame
            metric: Metric name
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"leaderboard_{metric}_{timestamp}.csv"
        filepath = self.storage_path / "reports" / filename
        
        leaderboard.to_csv(filepath, index=False)
        
        logger.info(f"Saved leaderboard to: {filepath}")
        return str(filepath)
    
    def get_best_models(
        self,
        metric: str,
        task_type: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top k models for a specific metric.
        
        Args:
            metric: Metric to rank by
            task_type: 'classification' or 'regression'
            top_k: Number of top models to return
            
        Returns:
            List of top model information
        """
        leaderboard = self.create_leaderboard(metric, task_type)
        
        if leaderboard.empty:
            return []
        
        top_models = leaderboard.head(top_k).to_dict('records')
        
        logger.info(f"Retrieved top {len(top_models)} models for metric: {metric}")
        return top_models
    
    def search_results(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        task_type: Optional[str] = None,
        min_metric_value: Optional[float] = None,
        max_metric_value: Optional[float] = None,
        metric: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search results by various criteria.
        
        Args:
            model_name: Filter by model name
            dataset_name: Filter by dataset name
            task_type: Filter by task type
            min_metric_value: Minimum metric value
            max_metric_value: Maximum metric value
            metric: Metric to filter by
            
        Returns:
            List of matching results
        """
        matching_results = []
        
        for result_file in self.storage_path.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
                
                # Apply filters
                if model_name and results.get("model_name") != model_name:
                    continue
                
                if dataset_name and results.get("dataset_name") != dataset_name:
                    continue
                
                if task_type and results.get("task_type") != task_type:
                    continue
                
                if metric and min_metric_value is not None:
                    metric_value = results.get("metrics", {}).get(metric)
                    if metric_value is None or metric_value < min_metric_value:
                        continue
                
                if metric and max_metric_value is not None:
                    metric_value = results.get("metrics", {}).get(metric)
                    if metric_value is None or metric_value > max_metric_value:
                        continue
                
                matching_results.append(results)
            
            except Exception as e:
                logger.warning(f"Error processing result file {result_file}: {e}")
        
        logger.info(f"Found {len(matching_results)} matching results")
        return matching_results
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            # Convert keys to strings and values recursively
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _unflatten_dict(self, d: Dict[str, Any], sep: str = "_") -> Dict[str, Any]:
        """Unflatten dictionary."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        return result
