#!/usr/bin/env python3

import sys
from src.experiments.experiment_runner import ExperimentRunner
from src.experiments.experiment_config import ExperimentConfig

def test_simple():
    runner = ExperimentRunner()
    
    config = ExperimentConfig(
        experiment_name="test_simple",
        description="Simple test"
    )
    
    config.dataset.source_type = "synthetic"
    config.dataset.dataset_type = "classification"
    config.dataset.n_samples = 100
    config.dataset.n_features = 10
    config.dataset.n_informative = 5  # Less than n_features
    config.dataset.n_classes = 2
    
    config.model.model_name = "logistic_regression"
    config.model.hyperparameters = {}
    
    config.evaluation.task_type = "classification"
    config.evaluation.cross_validation = False  # Disable CV for now
    
    try:
        results = runner.run_experiment(config)
        print("SUCCESS: Simple experiment worked")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple()
