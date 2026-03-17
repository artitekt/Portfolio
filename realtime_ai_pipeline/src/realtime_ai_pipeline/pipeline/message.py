"""
Generic message types for real-time AI pipeline.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime
import uuid


@dataclass
class EventMessage:
    """Base event message for pipeline processing."""
    id: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FeatureMessage:
    """Feature extraction result message."""
    event_id: str
    features: Dict[str, float]
    feature_vector: list
    timestamp: datetime
    processing_time_ms: float


@dataclass
class PredictionMessage:
    """AI inference result message."""
    event_id: str
    prediction: float
    confidence: float
    model_version: str
    timestamp: datetime
    inference_time_ms: float


@dataclass
class ResultMessage:
    """Final pipeline result message."""
    event_id: str
    prediction: float
    confidence: float
    features: Dict[str, float]
    processing_summary: Dict[str, float]
    timestamp: datetime
    total_latency_ms: float
