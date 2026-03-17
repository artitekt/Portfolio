"""
Real-time AI Pipeline Package

A clean demonstration of real-time AI processing pipeline architecture.
"""

__version__ = "1.0.0"
__author__ = "Portfolio Project"

# Core components
from .pipeline.pipeline import RealtimePipeline
from .pipeline.event_bus import EventBus, get_event_bus
from .pipeline.message import EventMessage, FeatureMessage, PredictionMessage, ResultMessage
from .utils.config import Config, load_config, setup_logging
from .utils.logger import PipelineLogger, get_logger

__all__ = [
    "RealtimePipeline",
    "EventBus", 
    "get_event_bus",
    "EventMessage",
    "FeatureMessage", 
    "PredictionMessage",
    "ResultMessage",
    "Config",
    "load_config",
    "setup_logging",
    "PipelineLogger",
    "get_logger"
]
