"""Experiment runner for AI Research Platform."""

import numpy as np
import inspect
from typing import Dict, Any, Optional, List
from ai_research_platform.data.dataset_loader import DatasetLoader
from ai_research_platform.models.model_registry import ModelRegistry
from ai_research_platform.evaluation.evaluator import ModelEvaluator
from ai_research_platform.research.experiment_tracker import ExperimentTracker
from ai_research_platform.research.results_store import ResultsStore
from ai_research_platform.utils.logger import get_logger
from .experiment_config import ExperimentConfig, ModelConfig

logger = get_logger(__name__)


class ExperimentRunner:
    """Run ML experiments with full tracking and evaluation."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.dataset_loader = DatasetLoader()
        self.model_registry = ModelRegistry()
        self.evaluator = ModelEvaluator()
        self.experiment_tracker = ExperimentTracker()
        self.results_store = ResultsStore()
    
    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run a complete experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment results
        """
        logger.info(f"Starting experiment: {config.experiment_name}")
        
        # Start experiment tracking
        experiment_id = self.experiment_tracker.start_experiment(
            experiment_name=config.experiment_name,
            description=config.description,
            tags=config.tags,
            parameters=config.to_dict()
        )
        
        try:
            # 1. Load dataset
            logger.info("Loading dataset...")
            X, y, dataset_metadata = self._load_dataset(config)
            
            # 2. Prepare data
            logger.info("Preparing data...")
            X_train, X_test, y_train, y_test = self.dataset_loader.prepare_data(
                X, y,
                test_size=config.dataset.test_size,
                scale_features=config.dataset.scale_features,
                encode_labels=config.dataset.encode_labels,
                random_state=config.dataset.random_state,
                task_type=config.evaluation.task_type
            )
            
            # Log dataset info
            if config.save_dataset_info:
                self.experiment_tracker.log_dataset_info({
                    "dataset_metadata": dataset_metadata,
                    "train_shape": X_train.shape,
                    "test_shape": X_test.shape,
                    "feature_names": self.dataset_loader.get_feature_names(),
                    "target_name": self.dataset_loader.get_target_name()
                })
            
            # 3. Create and train model
            logger.info(f"Training model: {config.model.model_name}")
            model = self._create_model(config)
            
            training_metrics = model.train(X_train, y_train)
            
            # Log model info
            self.experiment_tracker.log_model_info(model.get_model_info())
            
            # 4. Evaluate model
            logger.info("Evaluating model...")
            evaluation_results = self.evaluator.evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                task_type=config.evaluation.task_type,
                dataset_name=config.dataset.dataset_type
            )
            
            # 5. Cross-validation (optional)
            cv_results = None
            if config.evaluation.cross_validation:
                logger.info("Performing cross-validation...")
                cv_results = self.evaluator.cross_validate_model(
                    model_class=self.model_registry.get_model_by_name(config.model.model_name),
                    X=X,
                    y=y,
                    task_type=config.evaluation.task_type,
                    cv_folds=config.evaluation.cv_folds,
                    **config.model.get_hyperparameters()
                )
            
            # 6. Compile results
            results = {
                "experiment_id": experiment_id,
                "experiment_name": config.experiment_name,
                "config": config.to_dict(),
                "dataset_metadata": dataset_metadata,
                "training_metrics": training_metrics,
                "evaluation_results": evaluation_results,
                "cross_validation_results": cv_results,
                "feature_names": self.dataset_loader.get_feature_names(),
                "target_name": self.dataset_loader.get_target_name()
            }
            
            # 7. Log metrics and save results
            self.experiment_tracker.log_metrics(evaluation_results["metrics"])
            if cv_results:
                self.experiment_tracker.log_metrics({"cv_mean_score": cv_results["mean_score"]})
            
            # Save results
            results_path = self.results_store.save_results(
                results=results,
                experiment_name=config.experiment_name
            )
            self.experiment_tracker.log_artifact(results_path, "results")
            
            # Save model if requested
            if config.save_model:
                model_path = self._save_model(model, config.experiment_name)
                self.experiment_tracker.log_artifact(model_path, "model")
            
            # Save predictions if requested
            if config.save_predictions and evaluation_results.get("predictions"):
                predictions_path = self._save_predictions(
                    evaluation_results["predictions"],
                    evaluation_results.get("true_values"),
                    config.experiment_name
                )
                self.experiment_tracker.log_artifact(predictions_path, "predictions")
            
            # End experiment
            self.experiment_tracker.end_experiment("completed")
            
            logger.info(f"Experiment completed successfully: {config.experiment_name}")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.experiment_tracker.end_experiment("failed")
            raise
    
    def run_comparison(
        self,
        config: ExperimentConfig,
        model_names: List[str],
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a comparison experiment with multiple models.
        
        Args:
            config: Base experiment configuration
            model_names: List of model names to compare
            experiment_name: Name for the comparison experiment
            
        Returns:
            Comparison results
        """
        if experiment_name is None:
            experiment_name = f"{config.experiment_name}_comparison"
        
        logger.info(f"Starting comparison experiment: {experiment_name}")
        
        # Start experiment tracking
        experiment_id = self.experiment_tracker.start_experiment(
            experiment_name=experiment_name,
            description=f"Comparison of {len(model_names)} models",
            tags=["comparison"] + config.tags,
            parameters={
                "base_config": config.to_dict(),
                "models": model_names
            }
        )
        
        try:
            # 1. Load dataset
            logger.info("Loading dataset...")
            X, y, dataset_metadata = self._load_dataset(config)
            
            # 2. Prepare data
            logger.info("Preparing data...")
            X_train, X_test, y_train, y_test = self.dataset_loader.prepare_data(
                X, y,
                test_size=config.dataset.test_size,
                scale_features=config.dataset.scale_features,
                encode_labels=config.dataset.encode_labels,
                random_state=config.dataset.random_state,
                task_type=config.evaluation.task_type
            )
            
            # 3. Train all models
            models = {}
            for model_name in model_names:
                logger.info(f"Training model: {model_name}")
                
                # Create model config
                model_config = ModelConfig(model_name=model_name)
                model = self._create_model_from_config(model_config)
                
                # Train model
                model.train(X_train, y_train)
                models[model_name] = model
            
            # 4. Compare models
            logger.info("Comparing models...")
            comparison_results = self.evaluator.compare_models(
                models=models,
                X_test=X_test,
                y_test=y_test,
                task_type=config.evaluation.task_type,
                dataset_name=config.dataset.dataset_type
            )
            
            # 5. Compile results
            results = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "base_config": config.to_dict(),
                "models_compared": model_names,
                "dataset_metadata": dataset_metadata,
                "comparison_results": comparison_results,
                "feature_names": self.dataset_loader.get_feature_names(),
                "target_name": self.dataset_loader.get_target_name()
            }
            
            # 6. Log metrics and save results
            if "comparison" in comparison_results and "best_models" in comparison_results["comparison"]:
                self.experiment_tracker.log_metrics(comparison_results["comparison"]["best_models"])
            
            # Save comparison report
            comparison_path = self.results_store.save_comparison_report(
                comparison_results, experiment_name
            )
            self.experiment_tracker.log_artifact(comparison_path, "comparison_report")
            
            # End experiment
            self.experiment_tracker.end_experiment("completed")
            
            logger.info(f"Comparison experiment completed: {experiment_name}")
            return results
            
        except Exception as e:
            logger.error(f"Comparison experiment failed: {e}")
            self.experiment_tracker.end_experiment("failed")
            raise
    
    def _load_dataset(self, config: ExperimentConfig) -> tuple:
        """Load dataset based on configuration."""
        if config.dataset.source_type == "synthetic":
            # Pass all dataset parameters directly to loader
            # The loader will filter them appropriately
            return self.dataset_loader.load_synthetic(
                dataset_type=config.dataset.dataset_type,
                n_samples=config.dataset.n_samples,
                n_features=config.dataset.n_features,
                n_informative=config.dataset.n_informative,
                n_classes=config.dataset.n_classes,
                flip_y=config.dataset.flip_y,
                trend=config.dataset.trend,
                seasonality=config.dataset.seasonality,
                random_state=config.dataset.random_state
            )
        elif config.dataset.source_type == "csv":
            return self.dataset_loader.load_csv(
                file_path=config.dataset.file_path,
                target_column=config.dataset.target_column,
                feature_columns=config.dataset.feature_columns
            )
        else:
            raise ValueError(f"Unknown source type: {config.dataset.source_type}")
    
    def _create_model(self, config: ExperimentConfig):
        """Create model from configuration."""
        return self.model_registry.create_model(
            config.model.model_name,
            **config.model.get_hyperparameters()
        )
    
    def _create_model_from_config(self, model_config):
        """Create model from ModelConfig."""
        return self.model_registry.create_model(
            model_config.model_name,
            **model_config.get_hyperparameters()
        )
    
    def _save_model(self, model, experiment_name: str) -> str:
        """Save model to file."""
        import joblib
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"results/models/{experiment_name}_{timestamp}.pkl"
        
        joblib.dump(model, model_path)
        logger.info(f"Saved model to: {model_path}")
        return model_path
    
    def _save_predictions(self, predictions: List, true_values: List, experiment_name: str) -> str:
        """Save predictions to file."""
        import pandas as pd
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_path = f"results/predictions/{experiment_name}_{timestamp}.csv"
        
        df = pd.DataFrame({
            "true_values": true_values,
            "predictions": predictions
        })
        df.to_csv(predictions_path, index=False)
        
        logger.info(f"Saved predictions to: {predictions_path}")
        return predictions_path
