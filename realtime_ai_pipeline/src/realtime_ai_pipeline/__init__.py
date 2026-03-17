"""
Real-time AI Pipeline

A professional demonstration of a real-time AI processing pipeline using event-driven architecture with async Python.
"""

__version__ = "1.0.0"
__author__ = "Real-time AI Pipeline Team"

# Import key classes for easy access
from .pipeline.pipeline import RealtimePipeline
from .utils.config import Config

__all__ = [
    "RealtimePipeline",
    "Config"
]
