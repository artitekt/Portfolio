"""
AI Research Platform

A comprehensive platform for AI experimentation, model evaluation, and research workflows.
"""

__version__ = "2.0.0"
__author__ = "AI Research Platform Team"

# Import key classes for easy access
from .experiments.experiment_runner import ExperimentRunner
from .experiments.experiment_config import ExperimentConfig
from .experiments.experiment_sweeper import ExperimentSweeper
from .research.leaderboard import ModelLeaderboard
from .research.report_generator import ReportGenerator
from .data.dataset_registry import DatasetRegistry

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig", 
    "ExperimentSweeper",
    "ModelLeaderboard",
    "ReportGenerator",
    "DatasetRegistry"
]
