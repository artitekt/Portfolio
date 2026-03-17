"""Experiment configuration for AI Research Platform."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from ai_research_platform.utils.config import config


@dataclass
class DatasetConfig:
    """Configuration for dataset generation/loading."""
    source_type: str = "synthetic"  # "synthetic" or "csv"
    dataset_type: str = "classification"  # "classification", "regression", "timeseries", "friedman"
    file_path: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    
    # Synthetic dataset parameters
    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 10
    n_classes: int = 2
    flip_y: float = 0.01
    trend: float = 0.01
    seasonality: bool = True
    
    # Data preparation parameters
    test_size: float = field(default_factory=lambda: config.get("experiments.default_test_size", 0.2))
    scale_features: bool = True
    encode_labels: bool = True
    random_state: int = field(default_factory=lambda: config.get("experiments.random_state", 42))


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_name: str = "random_forest_classification"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters with defaults."""
        # Get default hyperparameters from config
        default_params = config.get(f"models.default_hyperparameters.{self.model_name}", {})
        
        # Merge with provided hyperparameters
        params = default_params.copy()
        params.update(self.hyperparameters)
        
        return params


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    task_type: str = "classification"  # "classification" or "regression"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1"] if field(default_factory=lambda: "classification") else ["r2", "rmse", "mae"])
    cross_validation: bool = False
    cv_folds: int = field(default_factory=lambda: config.get("experiments.cross_validation_folds", 5))


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment tracking
    save_model: bool = True
    save_predictions: bool = True
    save_dataset_info: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "dataset": {
                "source_type": self.dataset.source_type,
                "dataset_type": self.dataset.dataset_type,
                "file_path": self.dataset.file_path,
                "target_column": self.dataset.target_column,
                "feature_columns": self.dataset.feature_columns,
                "n_samples": self.dataset.n_samples,
                "n_features": self.dataset.n_features,
                "n_informative": self.dataset.n_informative,
                "n_classes": self.dataset.n_classes,
                "flip_y": self.dataset.flip_y,
                "trend": self.dataset.trend,
                "seasonality": self.dataset.seasonality,
                "test_size": self.dataset.test_size,
                "scale_features": self.dataset.scale_features,
                "encode_labels": self.dataset.encode_labels,
                "random_state": self.dataset.random_state
            },
            "model": {
                "model_name": self.model.model_name,
                "hyperparameters": self.model.hyperparameters
            },
            "evaluation": {
                "task_type": self.evaluation.task_type,
                "metrics": self.evaluation.metrics,
                "cross_validation": self.evaluation.cross_validation,
                "cv_folds": self.evaluation.cv_folds
            },
            "save_model": self.save_model,
            "save_predictions": self.save_predictions,
            "save_dataset_info": self.save_dataset_info
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        
        return cls(
            experiment_name=config_dict["experiment_name"],
            description=config_dict.get("description", ""),
            tags=config_dict.get("tags", []),
            dataset=dataset_config,
            model=model_config,
            evaluation=evaluation_config,
            save_model=config_dict.get("save_model", True),
            save_predictions=config_dict.get("save_predictions", True),
            save_dataset_info=config_dict.get("save_dataset_info", True)
        )


# Predefined experiment configurations
CLASSIFICATION_EXAMPLE = ExperimentConfig(
    experiment_name="classification_example",
    description="Example classification experiment with synthetic data",
    tags=["classification", "synthetic", "example"],
    dataset=DatasetConfig(
        source_type="synthetic",
        dataset_type="classification",
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_classes=2,
        flip_y=0.01
    ),
    model=ModelConfig(
        model_name="random_forest_classification",
        hyperparameters={"n_estimators": 100, "max_depth": 10}
    ),
    evaluation=EvaluationConfig(
        task_type="classification",
        metrics=["accuracy", "precision", "recall", "f1"]
    )
)

REGRESSION_EXAMPLE = ExperimentConfig(
    experiment_name="regression_example",
    description="Example regression experiment with synthetic data",
    tags=["regression", "synthetic", "example"],
    dataset=DatasetConfig(
        source_type="synthetic",
        dataset_type="regression",
        n_samples=1000,
        n_features=10,
        n_informative=5,
        flip_y=0.1
    ),
    model=ModelConfig(
        model_name="random_forest_regression",
        hyperparameters={"n_estimators": 100, "max_depth": 10}
    ),
    evaluation=EvaluationConfig(
        task_type="regression",
        metrics=["r2", "rmse", "mae", "mse"]
    )
)

TIMESERIES_EXAMPLE = ExperimentConfig(
    experiment_name="timeseries_example",
    description="Example time series experiment with synthetic data",
    tags=["timeseries", "synthetic", "example"],
    dataset=DatasetConfig(
        source_type="synthetic",
        dataset_type="timeseries",
        n_samples=1000,
        n_features=5,
        trend=0.01,
        seasonality=True,
        flip_y=0.1
    ),
    model=ModelConfig(
        model_name="linear_regression",
        hyperparameters={}
    ),
    evaluation=EvaluationConfig(
        task_type="regression",
        metrics=["r2", "rmse", "mae"]
    )
)
