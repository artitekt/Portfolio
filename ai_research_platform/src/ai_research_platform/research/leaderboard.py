"""Model leaderboard for ranking experiment results."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from ai_research_platform.utils.logger import get_logger

logger = get_logger(__name__)


class ModelLeaderboard:
    """Create and manage model leaderboards based on experiment results."""
    
    def __init__(self):
        """Initialize model leaderboard."""
        from ai_research_platform.research.results_store import ResultsStore
        self.results_store = ResultsStore()
    
    def get_top_models(
        self,
        metric: str = "accuracy",
        top_k: int = 5,
        task_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get top performing models based on a metric.
        
        Args:
            metric: Metric to rank by (accuracy, f1, r2, etc.)
            top_k: Number of top models to return
            task_type: Filter by task type (classification, regression)
            dataset_name: Filter by dataset name
            tags: Filter by experiment tags
            
        Returns:
            DataFrame with top models ranked by metric
        """
        logger.info(f"Creating leaderboard for metric: {metric}")
        
        # Load all experiment results
        all_results = self.results_store.list_experiments()
        
        if not all_results:
            logger.warning("No experiments found")
            return pd.DataFrame()
        
        # Filter and process results
        leaderboard_data = []
        
        for exp_result in all_results:
            try:
                # Extract experiment data
                exp_data = self._extract_experiment_data(exp_result)
                
                # Apply filters
                if task_type and exp_data.get('task_type') != task_type:
                    continue
                
                if dataset_name and exp_data.get('dataset_name') != dataset_name:
                    continue
                
                if tags and not any(tag in exp_data.get('tags', []) for tag in tags):
                    continue
                
                # Get metric value
                metrics = exp_data.get('metrics', {})
                metric_value = metrics.get(metric)
                
                if metric_value is None:
                    continue
                
                # Add to leaderboard
                leaderboard_data.append({
                    'Rank': 0,  # Will be set after sorting
                    'Model': exp_data.get('model_name', 'Unknown'),
                    'Dataset': exp_data.get('dataset_name', 'Unknown'),
                    'Metric': metric,
                    'Value': metric_value,
                    'Experiment_ID': exp_data.get('experiment_id'),
                    'Experiment_Name': exp_data.get('experiment_name'),
                    'Task_Type': exp_data.get('task_type'),
                    'Created_At': exp_data.get('created_at'),
                    'Tags': ', '.join(exp_data.get('tags', []))
                })
                
            except Exception as e:
                logger.warning(f"Error processing experiment {exp_result.get('experiment_id', 'unknown')}: {e}")
                continue
        
        if not leaderboard_data:
            logger.warning("No valid experiments found for leaderboard")
            return pd.DataFrame()
        
        # Create DataFrame and sort
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by metric value (descending for most metrics)
        df = df.sort_values('Value', ascending=False).reset_index(drop=True)
        
        # Set ranks
        df['Rank'] = range(1, len(df) + 1)
        
        # Return top k
        result = df.head(top_k)
        
        logger.info(f"Created leaderboard with {len(result)} entries")
        return result
    
    def _extract_experiment_data(self, exp_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data from experiment result."""
        # Handle different result formats
        if 'config' in exp_result:
            # Full experiment result
            config = exp_result['config']
            model_name = config.get('model', {}).get('model_name', 'Unknown')
            dataset_name = config.get('dataset', {}).get('dataset_type', 'Unknown')
            task_type = config.get('evaluation', {}).get('task_type', 'Unknown')
            tags = config.get('tags', [])
        else:
            # Simplified result format
            model_name = exp_result.get('model_name', 'Unknown')
            dataset_name = exp_result.get('dataset_name', 'Unknown')
            task_type = exp_result.get('task_type', 'Unknown')
            tags = exp_result.get('tags', [])
        
        # Get metrics
        if 'evaluation_results' in exp_result:
            metrics = exp_result['evaluation_results'].get('metrics', {})
        elif 'metrics' in exp_result:
            metrics = exp_result['metrics']
        else:
            metrics = {}
        
        return {
            'experiment_id': exp_result.get('experiment_id'),
            'experiment_name': exp_result.get('experiment_name'),
            'model_name': model_name,
            'dataset_name': dataset_name,
            'task_type': task_type,
            'tags': tags,
            'metrics': metrics,
            'created_at': exp_result.get('created_at', datetime.now().isoformat())
        }
    
    def create_comparison_table(
        self,
        experiment_ids: List[str],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Create a comparison table for specific experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to include in comparison
            
        Returns:
            DataFrame with experiment comparison
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "r2", "rmse", "mae"]
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                # Load experiment result
                result = self.results_store.load_results(exp_id)
                
                # Extract data
                exp_data = self._extract_experiment_data(result)
                
                # Get metrics
                available_metrics = exp_data.get('metrics', {})
                row_data = {
                    'Experiment_ID': exp_id,
                    'Experiment_Name': exp_data.get('experiment_name'),
                    'Model': exp_data.get('model_name'),
                    'Dataset': exp_data.get('dataset_name'),
                    'Task_Type': exp_data.get('task_type')
                }
                
                # Add requested metrics
                for metric in metrics:
                    row_data[metric.upper()] = available_metrics.get(metric, None)
                
                comparison_data.append(row_data)
                
            except Exception as e:
                logger.warning(f"Error loading experiment {exp_id}: {e}")
                continue
        
        if not comparison_data:
            logger.warning("No valid experiments found for comparison")
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        logger.info(f"Created comparison table with {len(df)} experiments")
        return df
    
    def get_model_performance_summary(
        self,
        model_name: str,
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Get performance summary for a specific model.
        
        Args:
            model_name: Name of the model to analyze
            metric: Metric to analyze
            
        Returns:
            Dictionary with performance summary
        """
        # Get all experiments for this model
        all_results = self.results_store.list_experiments()
        
        model_experiments = []
        metric_values = []
        datasets = []
        
        for exp_result in all_results:
            try:
                exp_data = self._extract_experiment_data(exp_result)
                
                if exp_data.get('model_name') != model_name:
                    continue
                
                metrics = exp_data.get('metrics', {})
                metric_value = metrics.get(metric)
                
                if metric_value is not None:
                    model_experiments.append(exp_data)
                    metric_values.append(metric_value)
                    datasets.append(exp_data.get('dataset_name'))
                    
            except Exception as e:
                logger.warning(f"Error processing experiment: {e}")
                continue
        
        if not metric_values:
            return {
                'model_name': model_name,
                'metric': metric,
                'total_experiments': 0,
                'mean_performance': 0.0,
                'best_performance': 0.0,
                'worst_performance': 0.0,
                'datasets_tested': []
            }
        
        # Calculate summary statistics
        summary = {
            'model_name': model_name,
            'metric': metric,
            'total_experiments': len(metric_values),
            'mean_performance': sum(metric_values) / len(metric_values),
            'best_performance': max(metric_values),
            'worst_performance': min(metric_values),
            'datasets_tested': list(set(datasets)),
            'best_experiment_id': None,
            'worst_experiment_id': None
        }
        
        # Find best and worst experiments
        best_idx = metric_values.index(summary['best_performance'])
        worst_idx = metric_values.index(summary['worst_performance'])
        
        summary['best_experiment_id'] = model_experiments[best_idx].get('experiment_id')
        summary['worst_experiment_id'] = model_experiments[worst_idx].get('experiment_id')
        
        return summary
    
    def export_leaderboard(
        self,
        leaderboard_df: pd.DataFrame,
        output_path: str,
        format: str = "csv"
    ):
        """
        Export leaderboard to file.
        
        Args:
            leaderboard_df: Leaderboard DataFrame
            output_path: Output file path
            format: Export format (csv, json, excel)
        """
        try:
            if format.lower() == "csv":
                leaderboard_df.to_csv(output_path, index=False)
            elif format.lower() == "json":
                leaderboard_df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == "excel":
                leaderboard_df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Leaderboard exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting leaderboard: {e}")
            raise
    
    def print_leaderboard(
        self,
        leaderboard_df: pd.DataFrame,
        show_columns: List[str] = None
    ):
        """
        Print leaderboard in a formatted way.
        
        Args:
            leaderboard_df: Leaderboard DataFrame
            show_columns: Columns to display
        """
        if leaderboard_df.empty:
            print("No data to display")
            return
        
        if show_columns is None:
            show_columns = ['Rank', 'Model', 'Dataset', 'Value']
        
        # Filter columns
        available_columns = [col for col in show_columns if col in leaderboard_df.columns]
        
        if not available_columns:
            print("No valid columns to display")
            return
        
        print("\n" + "="*80)
        print("MODEL LEADERBOARD")
        print("="*80)
        
        # Display the leaderboard
        display_df = leaderboard_df[available_columns].copy()
        
        # Format numeric columns
        if 'Value' in display_df.columns:
            display_df['Value'] = display_df['Value'].round(4)
        
        print(display_df.to_string(index=False))
        print("="*80)
