# AI Research Platform

A comprehensive platform for AI experimentation, model evaluation, and research workflows. This project demonstrates a reusable framework for running machine learning experiments, comparing models, tracking results, and generating research reports.

## Project Overview

The AI Research Platform provides a modular and extensible system for:

- **Experiment Management**: Track and organize ML experiments with full metadata
- **Dataset Generation & Registry**: Create and manage synthetic datasets with metadata tracking
- **Model Registry**: Dynamic model registration and management
- **Parameter Sweeps**: Automated hyperparameter optimization and grid search
- **Model Leaderboards**: Rank and compare model performance across experiments
- **Evaluation System**: Comprehensive metrics and model comparison
- **Research Reports**: Generate markdown reports summarizing experiment results
- **Results Tracking**: Persistent storage and analysis of experiment results

## Architecture Overview

```
Dataset Registry Layer
     ↓
Experiment Sweeper
     ↓
Experiment Runner
     ↓
Model Registry
     ↓
Training
     ↓
Evaluation
     ↓
Leaderboard & Report Generation
     ↓
Experiment Tracker
```

## Key Features

- **Modular Research Workflows**: Flexible experiment configuration and execution
- **Experiment Tracking**: Complete logging of parameters, metrics, and artifacts
- **Parameter Sweeps**: Automated grid search with configurable parameter combinations
- **Model Registry**: Dynamic model registration with baseline models included
- **Dataset Registry**: Track and manage datasets with metadata validation
- **Model Leaderboards**: Rank models by performance metrics with filtering options
- **Report Generation**: Automatic markdown report creation for experiment results
- **Dataset Generation**: Synthetic dataset creation for various ML tasks
- **Reproducible Experiments**: Fixed random seeds and configuration management
- **Results Storage**: JSON-based storage with search and comparison capabilities

## Project Structure

```
portfolio/ai_research_platform/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── examples/
│   └── run_experiment.py        # Demo script with full platform capabilities
├── docs/
│   └── architecture.md          # Detailed architecture documentation
├── src/
│   ├── experiments/
│   │   ├── experiment_runner.py # Main experiment orchestrator
│   │   ├── experiment_config.py  # Configuration management
│   │   └── experiment_sweeper.py # Parameter sweep functionality
│   ├── data/
│   │   ├── dataset_loader.py    # Data loading and preparation
│   │   ├── dataset_generator.py # Synthetic dataset generation
│   │   └── dataset_registry.py  # Dataset registration and management
│   ├── models/
│   │   ├── model_registry.py    # Model registration system
│   │   └── baseline_models.py   # Baseline model implementations
│   ├── evaluation/
│   │   ├── evaluator.py         # Model evaluation and comparison
│   │   └── metrics.py           # Evaluation metrics calculation
│   ├── research/
│   │   ├── experiment_tracker.py # Experiment tracking and logging
│   │   ├── results_store.py     # Results storage and retrieval
│   │   ├── leaderboard.py       # Model leaderboard functionality
│   │   └── report_generator.py  # Research report generation
│   └── utils/
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging utilities
└── results/                     # Experiment outputs (created automatically)
    ├── experiments/             # Individual experiment results
    ├── models/                  # Saved models
    ├── predictions/             # Model predictions
    ├── datasets/                # Dataset registry
    └── reports/                 # Generated research reports
```

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the demo experiment:
```bash
PYTHONPATH=src python examples/run_experiment.py
```

### Basic Usage

```python
from src.experiments.experiment_runner import ExperimentRunner
from src.experiments.experiment_config import ExperimentConfig

# Create experiment configuration
config = ExperimentConfig(
    experiment_name="my_experiment",
    description="Test classification with synthetic data",
    tags=["test", "classification"]
)

# Configure dataset
config.dataset.source_type = "synthetic"
config.dataset.dataset_type = "classification"
config.dataset.n_samples = 1000
config.dataset.n_features = 20

# Configure model
config.model.model_name = "random_forest_classification"
config.model.hyperparameters = {"n_estimators": 100}

# Run experiment
runner = ExperimentRunner()
results = runner.run_experiment(config)

# View results
print(f"Accuracy: {results['evaluation_results']['metrics']['accuracy']:.3f}")
```

## Available Models

### Classification Models
- `logistic_regression` - Logistic Regression
- `random_forest_classification` - Random Forest Classifier
- `gradient_boosting_classification` - Gradient Boosting Classifier

### Regression Models
- `linear_regression` - Linear Regression
- `random_forest_regression` - Random Forest Regressor
- `gradient_boosting_regression` - Gradient Boosting Regressor

### Baseline Models
- `baseline_mean` - Predicts mean of training targets (regression)
- `baseline_median` - Predicts median of training targets (regression)
- `baseline_majority_class` - Predicts majority class (classification)
- `baseline_random` - Makes random predictions

## Dataset Types

### Synthetic Datasets
- **Classification**: Binary or multi-class classification with configurable features
- **Regression**: Linear regression with noise and configurable informative features
- **Time Series**: Data with trend, seasonality, and noise components
- **Friedman**: Non-linear regression dataset (Friedman function)

### Real Data
- **CSV Loading**: Load datasets from CSV files with automatic preprocessing

## Experiment Configuration

Experiments are configured using the `ExperimentConfig` class:

```python
config = ExperimentConfig(
    experiment_name="experiment_name",
    description="Experiment description",
    tags=["tag1", "tag2"]
)

# Dataset configuration
config.dataset.source_type = "synthetic"  # or "csv"
config.dataset.dataset_type = "classification"
config.dataset.n_samples = 1000
config.dataset.test_size = 0.2
config.dataset.scale_features = True

# Model configuration
config.model.model_name = "random_forest_classification"
config.model.hyperparameters = {"n_estimators": 100, "max_depth": 10}

# Evaluation configuration
config.evaluation.task_type = "classification"
config.evaluation.metrics = ["accuracy", "precision", "recall", "f1"]
config.evaluation.cross_validation = True
config.evaluation.cv_folds = 5
```

## Model Comparison

Compare multiple models on the same dataset:

```python
# Define models to compare
models = [
    "logistic_regression",
    "random_forest_classification",
    "gradient_boosting_classification"
]

# Run comparison
comparison_results = runner.run_comparison(config, models)

# View best models
best_models = comparison_results['comparison_results']['comparison']['best_models']
for metric, best in best_models.items():
    print(f"{metric}: {best['model']} ({best['score']:.3f})")
```

## Experiment Tracking

All experiments are automatically tracked with:

- **Parameters**: Dataset configuration, model hyperparameters
- **Metrics**: Evaluation metrics and cross-validation scores
- **Artifacts**: Saved models, predictions, and result files
- **Metadata**: Experiment name, description, tags, timestamps

## Results Analysis

### View Experiment History

```python
from src.research.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker()
experiments = tracker.list_experiments(limit=10)

for exp in experiments:
    print(f"{exp['experiment_name']}: {exp['status']}")
```

### Create Leaderboards

```python
from src.research.results_store import ResultsStore

store = ResultsStore()
leaderboard = store.create_leaderboard(metric="accuracy", task_type="classification")
print(leaderboard.head())
```

## Research Platform Features

### Experiment Sweeps

Automated parameter grid search for hyperparameter optimization:

```python
from src.experiments.experiment_sweeper import ExperimentSweeper

# Create sweeper
sweeper = ExperimentSweeper()

# Define parameter grids
parameter_grids = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.001, 0.01, 0.1]
}

# Map parameters to config paths
parameter_mapping = {
    "n_estimators": "model.hyperparameters.n_estimators",
    "max_depth": "model.hyperparameters.max_depth",
    "learning_rate": "model.hyperparameters.learning_rate"
}

# Run sweep
sweep_summary = sweeper.run_sweep(
    base_config=config,
    parameter_grids=parameter_grids,
    parameter_mapping=parameter_mapping,
    sweep_name="my_hyperparameter_sweep",
    metric_to_optimize="accuracy"
)

print(f"Generated {sweep_summary.total_experiments} experiments")
print(f"Best accuracy: {sweep_summary.best_metric_value:.4f}")
```

### Model Leaderboard

Rank and compare model performance across experiments:

```python
from src.research.leaderboard import ModelLeaderboard

# Create leaderboard
leaderboard = ModelLeaderboard()

# Get top models by accuracy
top_models = leaderboard.get_top_models(
    metric="accuracy",
    top_k=10,
    task_type="classification"
)

# Display leaderboard
leaderboard.print_leaderboard(top_models)

# Get model performance summary
summary = leaderboard.get_model_performance_summary(
    model_name="random_forest_classification",
    metric="accuracy"
)
print(f"Average accuracy: {summary['mean_performance']:.4f}")
```

### Dataset Registry

Track and manage datasets with metadata:

```python
from src.data.dataset_registry import DatasetRegistry

# Create registry
registry = DatasetRegistry()

# Register a dataset
metadata = {
    "rows": 10000,
    "features": 25,
    "task": "classification",
    "description": "Customer churn dataset",
    "target_column": "churn",
    "missing_values": True
}

registry.register_dataset(
    name="customer_churn_v1",
    path="data/customer_churn.csv",
    metadata=metadata
)

# List available datasets
datasets = registry.list_datasets(task_type="classification")
for name, info in datasets.items():
    print(f"{name}: {info['metadata']['rows']} rows")

# Load a registered dataset
X, y = registry.load_dataset("customer_churn_v1")
```

### Research Reports

Generate comprehensive markdown reports:

```python
from src.research.report_generator import ReportGenerator

# Create report generator
generator = ReportGenerator()

# Generate experiment report
experiment_ids = ["exp_123", "exp_124", "exp_125"]
report_path = generator.generate_experiment_report(
    experiment_ids=experiment_ids,
    report_name="my_experiment_report"
)

# Generate sweep report
sweep_report_path = generator.generate_sweep_report(
    sweep_summary=sweep_summary,
    report_name="my_sweep_report"
)
```

## Example Output

### Research Platform Demo

```
Running experiment sweep...
Generated 6 experiment runs
Completed 6 experiments
Best experiment: exp_abc123
Best accuracy: 0.9234

Top Models:
1. GradientBoostingClassification accuracy=0.9234
2. RandomForestClassification accuracy=0.9156
3. LogisticRegression accuracy=0.8892

Report saved to: results/reports/demo_platform_report.md
Sweep report saved to: results/reports/demo_sweep_report.md
```

### Generated Report Structure

The markdown reports include:

- **Experiment Summary**: Overview of experiments and datasets
- **Datasets Used**: Table of datasets with metadata
- **Models Evaluated**: Model configurations and hyperparameters
- **Best Performing Models**: Top models by metric
- **Metric Comparison Table**: Detailed performance comparison
- **Detailed Results**: Individual experiment results
- **Conclusions**: Automated insights and recommendations

## Configuration

The platform uses a hierarchical configuration system:

- Default values in `src/utils/config.py`
- Environment-specific overrides via `AI_RESEARCH_CONFIG` environment variable
- Runtime configuration via `ExperimentConfig` objects

## Extending the Platform

### Adding New Models

```python
from src.models.model_registry import BaseModel, ModelRegistry

class MyModel(BaseModel):
    def train(self, X, y):
        # Implementation here
        return training_metrics
    
    def predict(self, X):
        # Implementation here
        return predictions

# Register the model
ModelRegistry.register_model("my_model", MyModel)
```

### Adding New Dataset Types

```python
from src.data.dataset_generator import DatasetGenerator

class CustomGenerator(DatasetGenerator):
    @staticmethod
    def generate_custom_dataset(**kwargs):
        # Implementation here
        return X, y, metadata
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms and metrics
- **pydantic**: Data validation (optional, for enhanced type checking)

## License

This project is part of the Artitekt ecosystem and demonstrates AI research platform capabilities.

## Contributing

When extending the platform:

1. Follow the existing modular architecture
2. Add comprehensive logging and error handling
3. Include proper type hints and documentation
4. Test with the demo experiment script
5. Update this README for new features

## Support

For questions or issues related to this AI Research Platform, please refer to the main project documentation.
