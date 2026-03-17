"""Experiment tracking for AI Research Platform."""

import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
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


class ExperimentTracker:
    """Track and manage ML experiments."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize experiment tracker.
        
        Args:
            storage_path: Path to store experiment results
        """
        self.storage_path = Path(storage_path or config.get("results.storage_path", "results"))
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.current_experiment = None
    
    def start_experiment(
        self,
        experiment_name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Experiment description
            tags: List of tags
            parameters: Experiment parameters
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        self.current_experiment = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "description": description,
            "tags": tags or [],
            "parameters": parameters or {},
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "results": {},
            "artifacts": []
        }
        
        logger.info(f"Started experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """
        Log parameters to current experiment.
        
        Args:
            parameters: Dictionary of parameters
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["parameters"].update(parameters)
        logger.info(f"Logged {len(parameters)} parameters to experiment")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to current experiment.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        if step is not None:
            if "step_metrics" not in self.current_experiment["results"]:
                self.current_experiment["results"]["step_metrics"] = {}
            self.current_experiment["results"]["step_metrics"][step] = metrics
        else:
            self.current_experiment["results"]["final_metrics"] = metrics
        
        logger.info(f"Logged {len(metrics)} metrics to experiment")
    
    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset information to current experiment.
        
        Args:
            dataset_info: Dataset information
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["results"]["dataset_info"] = dataset_info
        logger.info("Logged dataset information to experiment")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """
        Log model information to current experiment.
        
        Args:
            model_info: Model information
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["results"]["model_info"] = model_info
        logger.info("Logged model information to experiment")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """
        Log an artifact to current experiment.
        
        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["artifacts"].append({
            "path": artifact_path,
            "type": artifact_type,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Logged artifact: {artifact_path}")
    
    def end_experiment(self, status: str = "completed"):
        """
        End the current experiment.
        
        Args:
            status: Final experiment status
        """
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = status
        
        # Calculate duration
        start_time = datetime.fromisoformat(self.current_experiment["start_time"])
        end_time = datetime.fromisoformat(self.current_experiment["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.current_experiment["duration_seconds"] = duration
        
        # Save experiment
        self._save_experiment()
        
        logger.info(f"Ended experiment: {self.current_experiment['experiment_name']} (Duration: {duration:.2f}s)")
        
        experiment_id = self.current_experiment["experiment_id"]
        self.current_experiment = None
        
        return experiment_id
    
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
    
    def _save_experiment(self):
        """Save current experiment to file."""
        if not self.current_experiment:
            return
        
        experiment_file = self.storage_path / f"{self.current_experiment['experiment_id']}.json"
        
        # Convert numpy types before saving
        converted_experiment = self._convert_numpy_types(self.current_experiment)
        
        with open(experiment_file, 'w') as f:
            json.dump(converted_experiment, f, indent=2)
        
        logger.info(f"Saved experiment to: {experiment_file}")
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Load an experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment data
        """
        experiment_file = self.storage_path / f"{experiment_id}.json"
        
        if not experiment_file.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        
        with open(experiment_file, 'r') as f:
            experiment = json.load(f)
        
        return experiment
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            status: Filter by status
            tags: Filter by tags (experiment must have all specified tags)
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for experiment_file in self.storage_path.glob("*.json"):
            try:
                with open(experiment_file, 'r') as f:
                    experiment = json.load(f)
                
                # Apply filters
                if status and experiment.get("status") != status:
                    continue
                
                if tags:
                    experiment_tags = set(experiment.get("tags", []))
                    if not set(tags).issubset(experiment_tags):
                        continue
                
                # Create summary
                summary = {
                    "experiment_id": experiment["experiment_id"],
                    "experiment_name": experiment["experiment_name"],
                    "description": experiment.get("description", ""),
                    "status": experiment.get("status"),
                    "start_time": experiment.get("start_time"),
                    "end_time": experiment.get("end_time"),
                    "duration_seconds": experiment.get("duration_seconds"),
                    "tags": experiment.get("tags", [])
                }
                
                experiments.append(summary)
                
            except Exception as e:
                logger.warning(f"Error loading experiment {experiment_file}: {e}")
        
        # Sort by start time (newest first)
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if deleted successfully
        """
        experiment_file = self.storage_path / f"{experiment_id}.json"
        
        if experiment_file.exists():
            experiment_file.unlink()
            logger.info(f"Deleted experiment: {experiment_id}")
            return True
        
        return False
    
    def get_experiment_summary(self, experiment_id: str) -> str:
        """
        Get a formatted summary of an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Formatted summary string
        """
        experiment = self.load_experiment(experiment_id)
        
        summary = f"EXPERIMENT SUMMARY\n"
        summary += f"==================\n\n"
        summary += f"Name: {experiment['experiment_name']}\n"
        summary += f"ID: {experiment['experiment_id']}\n"
        summary += f"Description: {experiment.get('description', 'N/A')}\n"
        summary += f"Status: {experiment.get('status', 'Unknown')}\n"
        summary += f"Start Time: {experiment.get('start_time', 'N/A')}\n"
        summary += f"End Time: {experiment.get('end_time', 'N/A')}\n"
        
        if experiment.get("duration_seconds"):
            summary += f"Duration: {experiment['duration_seconds']:.2f} seconds\n"
        
        if experiment.get("tags"):
            summary += f"Tags: {', '.join(experiment['tags'])}\n"
        
        summary += f"\nPARAMETERS:\n"
        summary += "-----------\n"
        for key, value in experiment.get("parameters", {}).items():
            summary += f"{key}: {value}\n"
        
        if "final_metrics" in experiment.get("results", {}):
            summary += f"\nFINAL METRICS:\n"
            summary += "--------------\n"
            for key, value in experiment["results"]["final_metrics"].items():
                summary += f"{key}: {value}\n"
        
        if experiment.get("artifacts"):
            summary += f"\nARTIFACTS:\n"
            summary += "----------\n"
            for artifact in experiment["artifacts"]:
                summary += f"{artifact['type']}: {artifact['path']}\n"
        
        return summary
