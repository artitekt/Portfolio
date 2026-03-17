"""Experiment sweeper for parameter grid search."""

from typing import Dict, Any, List, Tuple
from itertools import product
from dataclasses import dataclass
from .experiment_config import ExperimentConfig
from .experiment_runner import ExperimentRunner
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SweepSummary:
    """Summary of experiment sweep results."""
    sweep_name: str
    total_experiments: int
    completed_experiments: int
    failed_experiments: int
    experiment_ids: List[str]
    parameter_combinations: List[Dict[str, Any]]
    best_experiment_id: str
    best_metric_value: float
    best_parameters: Dict[str, Any]


class ExperimentSweeper:
    """Run parameter sweeps over experiment configurations."""
    
    def __init__(self):
        """Initialize experiment sweeper."""
        self.runner = ExperimentRunner()
    
    def generate_parameter_combinations(
        self, 
        parameter_grids: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from parameter grids.
        
        Args:
            parameter_grids: Dictionary of parameter names to lists of values
            
        Returns:
            List of parameter combination dictionaries
        """
        if not parameter_grids:
            return [{}]
        
        # Get parameter names and value lists
        param_names = list(parameter_grids.keys())
        param_values = list(parameter_grids.values())
        
        # Generate all combinations
        combinations = []
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def create_experiment_configs(
        self,
        base_config: ExperimentConfig,
        parameter_grids: Dict[str, List[Any]],
        parameter_mapping: Dict[str, str] = None
    ) -> List[ExperimentConfig]:
        """
        Create experiment configs for all parameter combinations.
        
        Args:
            base_config: Base experiment configuration
            parameter_grids: Dictionary of parameter names to lists of values
            parameter_mapping: Mapping from sweep parameters to config paths
            
        Returns:
            List of experiment configurations
        """
        combinations = self.generate_parameter_combinations(parameter_grids)
        configs = []
        
        for i, combination in enumerate(combinations):
            # Create a copy of base config
            config = ExperimentConfig(
                experiment_name=f"{base_config.experiment_name}_sweep_{i+1}",
                description=f"{base_config.description} - Sweep {i+1}",
                tags=base_config.tags + ["sweep"]
            )
            
            # Copy base configuration
            config.dataset = base_config.dataset
            config.model = base_config.model
            config.evaluation = base_config.evaluation
            config.save_model = base_config.save_model
            config.save_predictions = base_config.save_predictions
            config.save_dataset_info = base_config.save_dataset_info
            
            # Apply parameter combination
            self._apply_parameters(config, combination, parameter_mapping)
            
            configs.append(config)
        
        return configs
    
    def _apply_parameters(
        self,
        config: ExperimentConfig,
        parameters: Dict[str, Any],
        parameter_mapping: Dict[str, str] = None
    ):
        """Apply parameters to experiment configuration."""
        if parameter_mapping is None:
            parameter_mapping = {}
        
        for param_name, param_value in parameters.items():
            # Use mapping if provided, otherwise use direct parameter name
            config_path = parameter_mapping.get(param_name, param_name)
            
            # Parse config path (e.g., "model.hyperparameters.n_estimators")
            path_parts = config_path.split('.')
            
            if len(path_parts) == 1:
                # Direct parameter
                if hasattr(config, path_parts[0]):
                    setattr(config, path_parts[0], param_value)
                else:
                    logger.warning(f"Unknown parameter: {config_path}")
            
            elif len(path_parts) == 2:
                # Nested parameter (e.g., "model.model_name")
                section, param = path_parts
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, param):
                        setattr(section_obj, param, param_value)
                    else:
                        logger.warning(f"Unknown parameter: {config_path}")
                else:
                    logger.warning(f"Unknown section: {section}")
            
            elif len(path_parts) == 3:
                # Deep nested parameter (e.g., "model.hyperparameters.n_estimators")
                section, subsection, param = path_parts
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, subsection):
                        if subsection == "hyperparameters":
                            # Special handling for hyperparameters dict
                            if param_value is not None:
                                section_obj.hyperparameters[param] = param_value
                        else:
                            logger.warning(f"Unknown subsection: {subsection}")
                    else:
                        logger.warning(f"Unknown subsection: {subsection}")
                else:
                    logger.warning(f"Unknown section: {section}")
            
            else:
                logger.warning(f"Invalid parameter path: {config_path}")
    
    def run_sweep(
        self,
        base_config: ExperimentConfig,
        parameter_grids: Dict[str, List[Any]],
        parameter_mapping: Dict[str, str] = None,
        sweep_name: str = None,
        metric_to_optimize: str = "accuracy",
        maximize_metric: bool = True
    ) -> SweepSummary:
        """
        Run a parameter sweep.
        
        Args:
            base_config: Base experiment configuration
            parameter_grids: Dictionary of parameter names to lists of values
            parameter_mapping: Mapping from sweep parameters to config paths
            sweep_name: Name for the sweep
            metric_to_optimize: Metric to use for finding best experiment
            maximize_metric: Whether to maximize (True) or minimize (False) the metric
            
        Returns:
            Sweep summary with results
        """
        if sweep_name is None:
            sweep_name = f"{base_config.experiment_name}_sweep"
        
        logger.info(f"Starting parameter sweep: {sweep_name}")
        
        # Generate experiment configurations
        configs = self.create_experiment_configs(
            base_config, parameter_grids, parameter_mapping
        )
        
        # Run experiments
        experiment_ids = []
        parameter_combinations = []
        completed_count = 0
        failed_count = 0
        
        best_experiment_id = None
        best_metric_value = float('-inf') if maximize_metric else float('inf')
        best_parameters = None
        
        for i, config in enumerate(configs):
            logger.info(f"Running experiment {i+1}/{len(configs)}: {config.experiment_name}")
            
            try:
                # Run experiment
                results = self.runner.run_experiment(config)
                
                # Extract metric value
                metrics = results.get('evaluation_results', {}).get('metrics', {})
                metric_value = metrics.get(metric_to_optimize)
                
                if metric_value is not None:
                    # Check if this is the best experiment
                    if (maximize_metric and metric_value > best_metric_value) or \
                       (not maximize_metric and metric_value < best_metric_value):
                        best_metric_value = metric_value
                        best_experiment_id = results['experiment_id']
                        best_parameters = parameter_grids.copy()
                        # Update best parameters with actual values used
                        for param_name, param_value in parameter_grids.items():
                            if param_name in config.model.hyperparameters:
                                best_parameters[param_name] = config.model.hyperparameters[param_name]
                
                experiment_ids.append(results['experiment_id'])
                parameter_combinations.append(config.model.hyperparameters.copy())
                completed_count += 1
                
                logger.info(f"Completed experiment {i+1}: {metric_to_optimize}={metric_value:.4f}")
                
            except Exception as e:
                logger.error(f"Failed experiment {i+1}: {e}")
                failed_count += 1
        
        # Create sweep summary
        summary = SweepSummary(
            sweep_name=sweep_name,
            total_experiments=len(configs),
            completed_experiments=completed_count,
            failed_experiments=failed_count,
            experiment_ids=experiment_ids,
            parameter_combinations=parameter_combinations,
            best_experiment_id=best_experiment_id,
            best_metric_value=best_metric_value,
            best_parameters=best_parameters or {}
        )
        
        logger.info(f"Sweep completed: {completed_count}/{len(configs)} experiments successful")
        logger.info(f"Best experiment: {best_experiment_id} with {metric_to_optimize}={best_metric_value:.4f}")
        
        return summary
    
    def get_sweep_results(self, experiment_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed results for sweep experiments.
        
        Args:
            experiment_ids: List of experiment IDs
            
        Returns:
            List of experiment results
        """
        from ai_research_platform.research.results_store import ResultsStore
        
        store = ResultsStore()
        results = []
        
        for exp_id in experiment_ids:
            try:
                result = store.load_results(exp_id)
                results.append(result)
            except Exception as e:
                logger.warning(f"Could not load results for experiment {exp_id}: {e}")
        
        return results
