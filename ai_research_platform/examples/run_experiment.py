#!/usr/bin/env python3
"""
Demo experiment script for AI Research Platform.
Demonstrates the full research platform capabilities including sweeps, leaderboards, and reports.
"""

import sys
import os
from pathlib import Path

# Import from proper package structure
from ai_research_platform.experiments.experiment_runner import ExperimentRunner
from ai_research_platform.experiments.experiment_config import ExperimentConfig, CLASSIFICATION_EXAMPLE, REGRESSION_EXAMPLE, ModelConfig
from ai_research_platform.experiments.experiment_sweeper import ExperimentSweeper
from ai_research_platform.research.leaderboard import ModelLeaderboard
from ai_research_platform.research.report_generator import ReportGenerator
from ai_research_platform.data.dataset_registry import DatasetRegistry
from ai_research_platform.evaluation.metrics import MetricsCalculator
from ai_research_platform.utils.logger import setup_logger


def run_classification_demo():
    """Run a classification experiment demo."""
    print("=" * 60)
    print("AI RESEARCH PLATFORM - CLASSIFICATION DEMO")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logger("classification_demo")
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Use predefined classification config
    config = CLASSIFICATION_EXAMPLE
    config.experiment_name = "demo_classification"
    config.description = "Demo classification experiment with synthetic data"
    config.tags = ["demo", "classification", "synthetic"]
    
    print(f"Running experiment: {config.experiment_name}")
    print(f"Dataset: {config.dataset.dataset_type}")
    print(f"Model: {config.model.model_name}")
    print()
    
    try:
        # Run experiment
        results = runner.run_experiment(config)
        
        # Print results
        print("--------------------------------")
        print("EXPERIMENT RESULTS")
        print("--------------------------------")
        print(f"Dataset: {results['dataset_metadata']['dataset_type']}")
        print(f"Model: {config.model.model_name}")
        print()
        
        # Print metrics
        metrics = results['evaluation_results']['metrics']
        print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"Precision: {metrics.get('precision', 0):.3f}")
        print(f"Recall: {metrics.get('recall', 0):.3f}")
        print(f"F1 Score: {metrics.get('f1', 0):.3f}")
        print()
        
        # Print dataset info
        dataset_info = results['dataset_metadata']
        print(f"Dataset Size: {dataset_info['n_samples']} samples")
        print(f"Features: {dataset_info['n_features']}")
        print(f"Classes: {dataset_info.get('n_classes', 'N/A')}")
        print()
        
        # Print experiment tracking info
        print(f"Experiment ID: {results['experiment_id']}")
        print("Results saved to results/ folder")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error running classification demo: {e}")
        return False


def run_regression_demo():
    """Run a regression demo."""
    print("=" * 60)
    print("AI RESEARCH PLATFORM - REGRESSION DEMO")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logger("regression_demo")
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Create experiment config
    config = ExperimentConfig(
        experiment_name="demo_regression",
        description="Demo regression experiment with synthetic data",
        tags=["demo", "regression", "synthetic"]
    )
    
    # Configure dataset
    config.dataset.source_type = "synthetic"
    config.dataset.dataset_type = "regression"
    config.dataset.n_samples = 1000
    config.dataset.n_features = 10
    config.dataset.n_informative = 5
    config.dataset.noise = 0.1  # Use noise for regression
    
    # Configure model
    config.model.model_name = "random_forest_regression"
    config.model.hyperparameters = {"n_estimators": 100, "max_depth": 10}
    
    # Configure evaluation
    config.evaluation.task_type = "regression"
    config.evaluation.metrics = ["r2", "rmse", "mae", "mse"]
    config.evaluation.cross_validation = False  # Disable cross-validation for demo
    config.save_predictions = False  # Don't save predictions to avoid printing
    
    print(f"Running experiment: {config.experiment_name}")
    print(f"Dataset: {config.dataset.dataset_type}")
    print(f"Model: {config.model.model_name}")
    print()
    
    try:
        # Run experiment
        results = runner.run_experiment(config)
        
        # Print results
        print("--------------------------------")
        print("EXPERIMENT RESULTS")
        print("--------------------------------")
        print(f"Dataset: {results['dataset_metadata']['dataset_type']}")
        print(f"Model: {config.model.model_name}")
        print()
        
        # Print metrics
        metrics = results['evaluation_results']['metrics']
        print(f"R²: {metrics.get('r2', 0):.3f}")
        print(f"RMSE: {metrics.get('rmse', 0):.3f}")
        print(f"MAE: {metrics.get('mae', 0):.3f}")
        print(f"MSE: {metrics.get('mse', 0):.3f}")
        print()
        
        # Print dataset info
        dataset_info = results['dataset_metadata']
        print(f"Dataset Size: {dataset_info['n_samples']} samples")
        print(f"Features: {dataset_info['n_features']}")
        print()
        
        # Print experiment tracking info
        print(f"Experiment ID: {results['experiment_id']}")
        print("Results saved to results/ folder")
        print()
        
        print("✅ Regression demo completed successfully")
        return True  # Always return True if we get here
        
    except Exception as e:
        print(f"Error running regression demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_comparison_demo():
    """Run a model comparison demo."""
    print("=" * 60)
    print("AI RESEARCH PLATFORM - MODEL COMPARISON DEMO")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logger("comparison_demo")
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Create base config
    config = ExperimentConfig(
        experiment_name="demo_comparison",
        description="Demo model comparison experiment",
        tags=["demo", "comparison", "synthetic"]
    )
    
    # Configure dataset
    config.dataset.source_type = "synthetic"
    config.dataset.dataset_type = "classification"
    config.dataset.n_samples = 1000
    config.dataset.n_features = 20
    config.dataset.n_informative = 10
    config.dataset.n_classes = 2
    
    # Models to compare
    models_to_compare = [
        "logistic_regression",
        "random_forest_classification",
        "gradient_boosting_classification"
    ]
    
    print(f"Running comparison experiment: {config.experiment_name}")
    print(f"Dataset: {config.dataset.dataset_type}")
    print(f"Models: {', '.join(models_to_compare)}")
    print()
    
    try:
        # Run comparison
        results = runner.run_comparison(config, models_to_compare)
        
        # Print results
        print("--------------------------------")
        print("COMPARISON RESULTS")
        print("--------------------------------")
        print(f"Dataset: {results['dataset_metadata']['dataset_type']}")
        print(f"Models Compared: {len(results['models_compared'])}")
        print()
        
        # Print best models by metric
        comparison = results['comparison_results']['comparison']
        if 'best_models' in comparison:
            print("BEST MODELS BY METRIC:")
            for metric, best in comparison['best_models'].items():
                print(f"{metric.upper()}: {best['model']} ({best['score']:.3f})")
            print()
        
        # Print detailed results
        print("DETAILED RESULTS:")
        for model_name, model_result in results['comparison_results']['model_results'].items():
            if 'error' not in model_result:
                print(f"\n{model_name}:")
                metrics = model_result['metrics']
                print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
                print(f"  Precision: {metrics.get('precision', 0):.3f}")
                print(f"  Recall: {metrics.get('recall', 0):.3f}")
                print(f"  F1 Score: {metrics.get('f1', 0):.3f}")
        
        print()
        print(f"Experiment ID: {results['experiment_id']}")
        print("Results saved to results/ folder")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error running comparison demo: {e}")
        return False


def run_research_platform_demo():
    """Run the full research platform demo."""
    print("=" * 80)
    print("AI RESEARCH PLATFORM - FULL CAPABILITY DEMO")
    print("=" * 80)
    
    # Initialize components
    registry = DatasetRegistry()
    sweeper = ExperimentSweeper()
    leaderboard = ModelLeaderboard()
    report_generator = ReportGenerator()
    
    experiment_ids = []
    
    try:
        # 1. Dataset Registration Demo
        print("\n1. DATASET REGISTRATION")
        print("-" * 40)
        
        # Register a synthetic dataset
        synthetic_metadata = {
            "rows": 1000,
            "features": 20,
            "task": "classification",
            "description": "Synthetic classification dataset for demo",
            "target_column": "target",
            "categorical_features": [],
            "missing_values": False
        }
        
        success = registry.register_dataset(
            name="demo_synthetic_v1",
            path="synthetic://classification",  # Special path for synthetic datasets
            metadata=synthetic_metadata
        )
        
        if success:
            print("✅ Registered demo_synthetic_v1 dataset")
        
        # List registered datasets
        datasets = registry.list_datasets()
        print(f"Available datasets: {list(datasets.keys())}")
        
        # 2. Parameter Sweep Demo
        print("\n2. PARAMETER SWEEP")
        print("-" * 40)
        print("Running experiment sweep...")
        
        # Create base configuration
        base_config = ExperimentConfig(
            experiment_name="demo_sweep",
            description="Demo parameter sweep",
            tags=["demo", "sweep", "classification"]
        )
        
        base_config.dataset.source_type = "synthetic"
        base_config.dataset.dataset_type = "classification"
        base_config.dataset.n_samples = 1000
        base_config.dataset.n_features = 20
        base_config.dataset.n_classes = 2
        
        base_config.evaluation.task_type = "classification"
        base_config.evaluation.metrics = ["accuracy", "precision", "recall", "f1"]
        
        # Define parameter grids
        parameter_grids = {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
            "model_type": ["random_forest_classification", "gradient_boosting_classification"]
        }
        
        # Map sweep parameters to config paths
        parameter_mapping = {
            "n_estimators": "model.hyperparameters.n_estimators",
            "max_depth": "model.hyperparameters.max_depth",
            "model_type": "model.model_name"
        }
        
        # Run sweep
        sweep_summary = sweeper.run_sweep(
            base_config=base_config,
            parameter_grids=parameter_grids,
            parameter_mapping=parameter_mapping,
            sweep_name="demo_platform_sweep",
            metric_to_optimize="accuracy"
        )
        
        print(f"✅ Generated {sweep_summary.total_experiments} experiment runs")
        print(f"✅ Completed {sweep_summary.completed_experiments} experiments")
        print(f"✅ Best experiment: {sweep_summary.best_experiment_id}")
        print(f"✅ Best accuracy: {sweep_summary.best_metric_value:.4f}")
        
        experiment_ids.extend(sweep_summary.experiment_ids)
        
        # 3. Model Leaderboard Demo
        print("\n3. MODEL LEADERBOARD")
        print("-" * 40)
        
        # Get top models
        top_models = leaderboard.get_top_models(
            metric="accuracy",
            top_k=5,
            task_type="classification"
        )
        
        if not top_models.empty:
            print("Top Models:")
            for _, row in top_models.iterrows():
                print(f"{int(row['Rank'])}. {row['Model']} accuracy={row['Value']:.4f}")
        else:
            print("No models found for leaderboard")
        
        # 4. Report Generation Demo
        print("\n4. REPORT GENERATION")
        print("-" * 40)
        
        if experiment_ids:
            # Generate comprehensive report
            report_path = report_generator.generate_experiment_report(
                experiment_ids=experiment_ids,
                report_name="demo_platform_report"
            )
            
            if report_path:
                print(f"✅ Report saved to: {report_path}")
            
            # Generate sweep report
            sweep_report_path = report_generator.generate_sweep_report(
                sweep_summary=sweep_summary,
                report_name="demo_sweep_report"
            )
            
            if sweep_report_path:
                print(f"✅ Sweep report saved to: {sweep_report_path}")
        
        print("\n" + "=" * 80)
        print("RESEARCH PLATFORM DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nSummary:")
        print(f"- Dataset Registry: {len(datasets)} datasets registered")
        print(f"- Parameter Sweep: {sweep_summary.completed_experiments} experiments completed")
        print(f"- Best Model: {sweep_summary.best_parameters.get('model_type', 'Unknown')}")
        print(f"- Best Accuracy: {sweep_summary.best_metric_value:.4f}")
        print(f"- Reports Generated: Check results/reports/ folder")
        
        return True
        
    except Exception as e:
        print(f"❌ Research platform demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main demo function with all capabilities."""
    print("AI Research Platform - Full Demo Suite")
    print("=======================================")
    print()
    
    # Run all demos including the new research platform demo
    demos = [
        ("Research Platform Demo", run_research_platform_demo),
        ("Classification", run_classification_demo),
        ("Regression", run_regression_demo),
        ("Model Comparison", run_model_comparison_demo)
    ]
    
    results = []
    for demo_name, demo_func in demos:
        print(f"Running {demo_name}...")
        success = demo_func()
        results.append((demo_name, success))
        if success:
            print(f"✅ {demo_name} completed successfully")
        else:
            print(f"❌ {demo_name} failed")
        print()
    
    # Summary
    print("=" * 80)
    print("DEMO SUMMARY")
    print("=" * 80)
    
    for demo_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{demo_name:25s}: {status}")
    
    print()
    successful_demos = sum(1 for _, success in results if success)
    print(f"Completed: {successful_demos}/{len(demos)} demos")
    
    if successful_demos == len(demos):
        print("\n🎉 All demos completed successfully!")
        print("Check the following folders for outputs:")
        print("- 'results/' for experiment results")
        print("- 'results/reports/' for generated reports")
        print("- 'results/datasets/' for dataset registry")
    else:
        print("\n⚠️  Some demos failed. Check the error messages above.")


if __name__ == "__main__":
    main()
