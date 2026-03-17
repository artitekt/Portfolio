"""Experiment report generator for creating markdown reports."""

from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generate comprehensive markdown reports for experiment results."""
    
    def __init__(self, output_dir: str = "results/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_experiment_report(
        self,
        experiment_ids: List[str],
        report_name: str = None,
        include_plots: bool = False
    ) -> str:
        """
        Generate a comprehensive experiment report.
        
        Args:
            experiment_ids: List of experiment IDs to include
            report_name: Name for the report file
            include_plots: Whether to include plot placeholders
            
        Returns:
            Path to generated report
        """
        if report_name is None:
            report_name = f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Generating experiment report: {report_name}")
        
        # Load experiment data
        experiment_data = self._load_experiments(experiment_ids)
        
        if not experiment_data:
            logger.warning("No valid experiments found")
            return ""
        
        # Generate report sections
        report_content = self._generate_report_sections(experiment_data, include_plots)
        
        # Save report
        report_path = self.output_dir / f"{report_name}.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise
    
    def _load_experiments(self, experiment_ids: List[str]) -> List[Dict[str, Any]]:
        """Load experiment data from results store."""
        from ai_research_platform.research.results_store import ResultsStore
        
        store = ResultsStore()
        experiments = []
        
        for exp_id in experiment_ids:
            try:
                result = store.load_results(exp_id)
                if result:
                    experiments.append(result)
            except Exception as e:
                logger.warning(f"Could not load experiment {exp_id}: {e}")
        
        return experiments
    
    def _generate_report_sections(self, experiments: List[Dict[str, Any]], include_plots: bool) -> str:
        """Generate all report sections."""
        sections = []
        
        # Title and summary
        sections.append(self._generate_title_section(experiments))
        sections.append(self._generate_summary_section(experiments))
        
        # Datasets section
        sections.append(self._generate_datasets_section(experiments))
        
        # Models section
        sections.append(self._generate_models_section(experiments))
        
        # Best performing models
        sections.append(self._generate_best_models_section(experiments))
        
        # Metric comparison table
        sections.append(self._generate_metric_comparison_section(experiments))
        
        # Detailed results
        sections.append(self._generate_detailed_results_section(experiments))
        
        # Conclusions
        sections.append(self._generate_conclusions_section(experiments))
        
        # Plots section (if requested)
        if include_plots:
            sections.append(self._generate_plots_section(experiments))
        
        return "\n\n".join(sections)
    
    def _generate_title_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate title section."""
        total_experiments = len(experiments)
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""# Experiment Report

**Generated on:** {date}  
**Total Experiments:** {total_experiments}

---"""
    
    def _generate_summary_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate experiment summary section."""
        # Calculate summary statistics
        datasets = set()
        models = set()
        task_types = set()
        
        for exp in experiments:
            if 'config' in exp:
                config = exp['config']
                datasets.add(config.get('dataset', {}).get('dataset_type', 'Unknown'))
                models.add(config.get('model', {}).get('model_name', 'Unknown'))
                task_types.add(config.get('evaluation', {}).get('task_type', 'Unknown'))
        
        return f"""## Experiment Summary

This report summarizes {len(experiments)} machine learning experiments conducted to evaluate different models and configurations.

**Key Statistics:**
- **Datasets Used:** {len(datasets)} ({', '.join(datasets)})
- **Models Evaluated:** {len(models)} ({', '.join(models)})
- **Task Types:** {', '.join(task_types)}"""
    
    def _generate_datasets_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate datasets section."""
        dataset_info = {}
        
        for exp in experiments:
            if 'config' in exp:
                config = exp['config']
                dataset_config = config.get('dataset', {})
                dataset_name = dataset_config.get('dataset_type', 'Unknown')
                
                if dataset_name not in dataset_info:
                    dataset_info[dataset_name] = {
                        'experiments': 0,
                        'samples': dataset_config.get('n_samples', 'N/A'),
                        'features': dataset_config.get('n_features', 'N/A'),
                        'task_type': config.get('evaluation', {}).get('task_type', 'Unknown')
                    }
                
                dataset_info[dataset_name]['experiments'] += 1
        
        # Create table
        lines = ["## Datasets Used", ""]
        lines.append("| Dataset | Experiments | Samples | Features | Task Type |")
        lines.append("|---------|-------------|---------|----------|-----------|")
        
        for dataset_name, info in dataset_info.items():
            lines.append(f"| {dataset_name} | {info['experiments']} | {info['samples']} | {info['features']} | {info['task_type']} |")
        
        return "\n".join(lines)
    
    def _generate_models_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate models section."""
        model_info = {}
        
        for exp in experiments:
            if 'config' in exp:
                config = exp['config']
                model_name = config.get('model', {}).get('model_name', 'Unknown')
                
                if model_name not in model_info:
                    model_info[model_name] = {
                        'experiments': 0,
                        'hyperparameters': {}
                    }
                
                model_info[model_name]['experiments'] += 1
                
                # Collect unique hyperparameters
                hyperparams = config.get('model', {}).get('hyperparameters', {})
                for param, value in hyperparams.items():
                    if param not in model_info[model_name]['hyperparameters']:
                        model_info[model_name]['hyperparameters'][param] = set()
                    model_info[model_name]['hyperparameters'][param].add(str(value))
        
        # Create table
        lines = ["## Models Evaluated", ""]
        lines.append("| Model | Experiments | Key Hyperparameters |")
        lines.append("|-------|-------------|---------------------|")
        
        for model_name, info in model_info.items():
            # Format hyperparameters
            hyperparam_strs = []
            for param, values in info['hyperparameters'].items():
                if len(values) == 1:
                    hyperparam_strs.append(f"{param}={list(values)[0]}")
                else:
                    hyperparam_strs.append(f"{param}=[{', '.join(sorted(values))}]")
            
            hyperparam_display = ", ".join(hyperparam_strs[:3])  # Limit to 3 params
            if len(hyperparam_strs) > 3:
                hyperparam_display += "..."
            
            lines.append(f"| {model_name} | {info['experiments']} | {hyperparam_display} |")
        
        return "\n".join(lines)
    
    def _generate_best_models_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate best performing models section."""
        # Collect all metrics and find best performers
        all_metrics = {}
        
        for exp in experiments:
            if 'evaluation_results' in exp:
                metrics = exp['evaluation_results'].get('metrics', {})
                model_name = exp.get('config', {}).get('model', {}).get('model_name', 'Unknown')
                dataset_name = exp.get('config', {}).get('dataset', {}).get('dataset_type', 'Unknown')
                
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        
                        all_metrics[metric_name].append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'value': value,
                            'experiment_id': exp.get('experiment_id', 'Unknown')
                        })
        
        # Find best for each metric
        lines = ["## Best Performing Models", ""]
        
        for metric_name, results in all_metrics.items():
            if not results:
                continue
            
            # Determine if higher is better (most metrics except error metrics)
            error_metrics = ['rmse', 'mae', 'mse', 'log_loss']
            maximize = metric_name not in error_metrics
            
            # Sort and get best
            sorted_results = sorted(results, key=lambda x: x['value'], reverse=maximize)
            best = sorted_results[0]
            
            lines.append(f"### Best {metric_name.upper()}")
            lines.append(f"- **Model:** {best['model']}")
            lines.append(f"- **Dataset:** {best['dataset']}")
            lines.append(f"- **Score:** {best['value']:.4f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_metric_comparison_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate metric comparison table."""
        # Collect all metrics for comparison
        comparison_data = []
        
        for exp in experiments:
            if 'evaluation_results' in exp and 'config' in exp:
                metrics = exp['evaluation_results'].get('metrics', {})
                model_name = exp['config'].get('model', {}).get('model_name', 'Unknown')
                dataset_name = exp['config'].get('dataset', {}).get('dataset_type', 'Unknown')
                
                row = {
                    'Model': model_name,
                    'Dataset': dataset_name
                }
                
                # Add all numeric metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[metric_name.upper()] = f"{value:.4f}"
                
                comparison_data.append(row)
        
        if not comparison_data:
            return "## Metric Comparison Table\n\nNo metric data available."
        
        # Create DataFrame and table
        df = pd.DataFrame(comparison_data)
        
        lines = ["## Metric Comparison Table", ""]
        
        # Convert to markdown table
        table_str = df.to_string(index=False)
        lines.append("```\n" + table_str + "\n```")
        
        return "\n".join(lines)
    
    def _generate_detailed_results_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate detailed results section."""
        lines = ["## Detailed Results", ""]
        
        for i, exp in enumerate(experiments, 1):
            exp_name = exp.get('experiment_name', f'Experiment {i}')
            exp_id = exp.get('experiment_id', 'Unknown')
            
            lines.append(f"### {exp_name}")
            lines.append(f"**Experiment ID:** {exp_id}")
            
            if 'config' in exp:
                config = exp['config']
                lines.append(f"**Model:** {config.get('model', {}).get('model_name', 'Unknown')}")
                lines.append(f"**Dataset:** {config.get('dataset', {}).get('dataset_type', 'Unknown')}")
            
            if 'evaluation_results' in exp:
                metrics = exp['evaluation_results'].get('metrics', {})
                lines.append("**Metrics:**")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"- {metric}: {value:.4f}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_conclusions_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate conclusions section."""
        lines = ["## Conclusions", ""]
        
        # Analyze results to generate insights
        if not experiments:
            lines.append("No experiments to analyze.")
            return "\n".join(lines)
        
        # Find best overall model
        model_performance = {}
        for exp in experiments:
            if 'evaluation_results' in exp and 'config' in exp:
                metrics = exp['evaluation_results'].get('metrics', {})
                model_name = exp['config'].get('model', {}).get('model_name', 'Unknown')
                
                if model_name not in model_performance:
                    model_performance[model_name] = []
                
                # Use accuracy if available, otherwise first metric
                if 'accuracy' in metrics:
                    model_performance[model_name].append(metrics['accuracy'])
                elif metrics:
                    first_metric = list(metrics.values())[0]
                    if isinstance(first_metric, (int, float)):
                        model_performance[model_name].append(first_metric)
        
        # Calculate average performance
        avg_performance = {}
        for model, scores in model_performance.items():
            if scores:
                avg_performance[model] = sum(scores) / len(scores)
        
        if avg_performance:
            best_model = max(avg_performance, key=avg_performance.get)
            lines.append(f"**Best Overall Model:** {best_model} (avg performance: {avg_performance[best_model]:.4f})")
            lines.append("")
        
        # General insights
        lines.append("**Key Findings:**")
        lines.append(f"- Total of {len(experiments)} experiments conducted")
        lines.append(f"- {len(model_performance)} different models evaluated")
        
        if len(experiments) > 1:
            lines.append("- Performance varied across different configurations and datasets")
        
        lines.append("")
        lines.append("**Recommendations:**")
        lines.append("- Consider hyperparameter tuning for top performing models")
        lines.append("- Evaluate models on additional datasets for robustness")
        lines.append("- Document experiment configurations for reproducibility")
        
        return "\n".join(lines)
    
    def _generate_plots_section(self, experiments: List[Dict[str, Any]]) -> str:
        """Generate plots section with placeholders."""
        lines = ["## Visualizations", ""]
        lines.append("### Model Performance Comparison")
        lines.append("```python")
        lines.append("# Placeholder for model performance comparison plot")
        lines.append("# This would typically show a bar chart of model performance")
        lines.append("```")
        lines.append("")
        
        lines.append("### Metric Distribution")
        lines.append("```python")
        lines.append("# Placeholder for metric distribution plot")
        lines.append("# This would typically show distribution of metric values")
        lines.append("```")
        
        return "\n".join(lines)
    
    def generate_sweep_report(
        self,
        sweep_summary,
        report_name: str = None
    ) -> str:
        """
        Generate a report for parameter sweep results.
        
        Args:
            sweep_summary: SweepSummary object from experiment sweeper
            report_name: Name for the report file
            
        Returns:
            Path to generated report
        """
        if report_name is None:
            report_name = f"sweep_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        lines = [f"# Parameter Sweep Report: {sweep_summary.sweep_name}", ""]
        lines.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("## Sweep Summary")
        lines.append(f"- **Total Experiments:** {sweep_summary.total_experiments}")
        lines.append(f"- **Completed:** {sweep_summary.completed_experiments}")
        lines.append(f"- **Failed:** {sweep_summary.failed_experiments}")
        lines.append(f"- **Best Experiment ID:** {sweep_summary.best_experiment_id}")
        lines.append(f"- **Best Metric Value:** {sweep_summary.best_metric_value:.4f}")
        lines.append("")
        
        # Best parameters
        if sweep_summary.best_parameters:
            lines.append("## Best Parameters")
            for param, value in sweep_summary.best_parameters.items():
                lines.append(f"- **{param}:** {value}")
            lines.append("")
        
        # All combinations
        lines.append("## All Parameter Combinations")
        lines.append("| Experiment ID | Parameters |")
        lines.append("|---------------|------------|")
        
        for i, (exp_id, params) in enumerate(zip(sweep_summary.experiment_ids, sweep_summary.parameter_combinations)):
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            lines.append(f"| {exp_id} | {param_str} |")
        
        # Save report
        report_path = self.output_dir / f"{report_name}.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            
            logger.info(f"Sweep report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error saving sweep report: {e}")
            raise
