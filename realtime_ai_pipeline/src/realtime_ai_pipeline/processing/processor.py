"""
Component: Feature Processor
Role: Converts raw event data into structured features used by the 
     inference engine. Implements statistical, temporal, and categorical
     feature extraction with sliding window management.
"""

"""
Feature processor for real-time event processing.
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from ..pipeline.message import EventMessage, FeatureMessage

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Generic feature processor for real-time events."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_config = config.get("features", {})
        
        # Feature extraction configuration
        self.enable_statistical_features = self.feature_config.get("statistical", True)
        self.enable_temporal_features = self.feature_config.get("temporal", True)
        self.enable_categorical_features = self.feature_config.get("categorical", True)
        self.feature_window = self.feature_config.get("window_size", 10)
        
        # Feature history for temporal features
        self.feature_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history = self.feature_window
        
        # Feature statistics
        self.feature_stats = {
            "events_processed": 0,
            "features_generated": 0,
            "processing_time_total": 0.0,
            "errors": 0
        }
    
    async def process_event(self, event: EventMessage) -> Dict[str, Any]:
        """Process event and extract features."""
        start_time = time.time()
        
        try:
            # Update history
            self._update_history(event)
            
            # Extract features
            features = {}
            feature_vector = []
            
            # Statistical features
            if self.enable_statistical_features:
                stat_features = self._extract_statistical_features(event)
                features.update(stat_features)
                feature_vector.extend(stat_features.values())
            
            # Temporal features
            if self.enable_temporal_features:
                temporal_features = self._extract_temporal_features(event)
                features.update(temporal_features)
                feature_vector.extend(temporal_features.values())
            
            # Categorical features
            if self.enable_categorical_features:
                categorical_features = self._extract_categorical_features(event)
                features.update(categorical_features)
                feature_vector.extend(categorical_features.values())
            
            # Normalize feature vector
            feature_vector = self._normalize_features(feature_vector)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.feature_stats["events_processed"] += 1
            self.feature_stats["features_generated"] += len(features)
            self.feature_stats["processing_time_total"] += processing_time
            
            result = {
                "features": features,
                "vector": feature_vector,
                "processing_time_ms": processing_time
            }
            
            logger.debug(f"Processed event {event.id}: {len(features)} features")
            return result
            
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")
            self.feature_stats["errors"] += 1
            raise
    
    def _update_history(self, event: EventMessage):
        """Update event history for temporal features."""
        source = event.source
        
        if source not in self.feature_history:
            self.feature_history[source] = []
        
        # Add current event
        self.feature_history[source].append({
            "timestamp": event.timestamp,
            "data": event.data,
            "id": event.id
        })
        
        # Maintain window size
        if len(self.feature_history[source]) > self.max_history:
            self.feature_history[source] = self.feature_history[source][-self.max_history:]
    
    def _extract_statistical_features(self, event: EventMessage) -> Dict[str, float]:
        """Extract statistical features from event data."""
        features = {}
        data = event.data
        
        # Extract numeric values
        numeric_values = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
        
        if numeric_values:
            # Basic statistics
            features["mean_value"] = np.mean(numeric_values)
            features["std_value"] = np.std(numeric_values)
            features["min_value"] = np.min(numeric_values)
            features["max_value"] = np.max(numeric_values)
            features["range_value"] = features["max_value"] - features["min_value"]
            
            # Percentiles
            features["median_value"] = np.median(numeric_values)
            features["q25_value"] = np.percentile(numeric_values, 25)
            features["q75_value"] = np.percentile(numeric_values, 75)
            
            # Distribution features
            features["skewness"] = self._calculate_skewness(numeric_values)
            features["kurtosis"] = self._calculate_kurtosis(numeric_values)
            
            # Count features
            features["numeric_count"] = len(numeric_values)
            features["total_fields"] = len(data)
            features["numeric_ratio"] = len(numeric_values) / len(data)
        
        return features
    
    def _extract_temporal_features(self, event: EventMessage) -> Dict[str, float]:
        """Extract temporal features from event history."""
        features = {}
        source = event.source
        
        if source not in self.feature_history or len(self.feature_history[source]) < 2:
            return features
        
        history = self.feature_history[source]
        current_event = history[-1]
        
        # Time-based features
        features["hour_of_day"] = float(time.strftime("%H", time.localtime(event.timestamp)))
        features["day_of_week"] = float(time.strftime("%w", time.localtime(event.timestamp)))
        
        # Inter-event time features
        if len(history) >= 2:
            prev_event = history[-2]
            time_diff = event.timestamp - prev_event["timestamp"]
            features["inter_event_time"] = time_diff
            features["events_per_minute"] = 60.0 / time_diff if time_diff > 0 else 0.0
        
        # Trend features
        if len(history) >= 3:
            # Extract numeric values from recent events
            recent_values = []
            for hist_event in history[-5:]:  # Last 5 events
                for value in hist_event["data"].values():
                    if isinstance(value, (int, float)):
                        recent_values.append(float(value))
            
            if len(recent_values) >= 2:
                # Simple trend calculation
                x = np.arange(len(recent_values))
                y = np.array(recent_values)
                
                # Linear regression slope
                if len(x) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    features["trend_slope"] = slope
                    features["trend_direction"] = 1.0 if slope > 0 else -1.0 if slope < 0 else 0.0
        
        # Frequency features
        features["source_frequency"] = len(history)
        features["avg_inter_event_time"] = np.mean([
            history[i]["timestamp"] - history[i-1]["timestamp"]
            for i in range(1, len(history))
        ]) if len(history) > 1 else 0.0
        
        return features
    
    def _extract_categorical_features(self, event: EventMessage) -> Dict[str, float]:
        """Extract categorical features from event data."""
        features = {}
        data = event.data
        
        # Categorical encoding
        categorical_fields = ["status", "type", "category", "level", "severity"]
        
        for field in categorical_fields:
            if field in data:
                value = str(data[field]).lower()
                
                # One-hot encoding for common values
                if field == "status":
                    features[f"status_normal"] = 1.0 if value == "normal" else 0.0
                    features[f"status_warning"] = 1.0 if value == "warning" else 0.0
                    features[f"status_critical"] = 1.0 if value == "critical" else 0.0
                elif field == "severity":
                    features[f"severity_info"] = 1.0 if value == "info" else 0.0
                    features[f"severity_warning"] = 1.0 if value == "warning" else 0.0
                    features[f"severity_error"] = 1.0 if value == "error" else 0.0
                    features[f"severity_critical"] = 1.0 if value == "critical" else 0.0
                else:
                    # Generic categorical encoding
                    features[f"{field}_present"] = 1.0
                    features[f"{field}_length"] = float(len(value))
        
        # Source-based features
        features[f"source_{event.source}"] = 1.0
        
        # Data structure features
        features["data_field_count"] = float(len(data))
        features["data_string_fields"] = float(sum(1 for v in data.values() if isinstance(v, str)))
        features["data_numeric_fields"] = float(sum(1 for v in data.values() if isinstance(v, (int, float))))
        features["data_bool_fields"] = float(sum(1 for v in data.values() if isinstance(v, bool)))
        
        return features
    
    def _normalize_features(self, feature_vector: List[float]) -> List[float]:
        """Normalize feature vector."""
        if not feature_vector:
            return feature_vector
        
        # Convert to numpy array
        features = np.array(feature_vector)
        
        # Handle NaN and infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Z-score normalization
        if len(features) > 1 and np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
        
        # Clip extreme values
        features = np.clip(features, -3.0, 3.0)
        
        return features.tolist()
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of values."""
        if len(values) < 3:
            return 0.0
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((values - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of values."""
        if len(values) < 4:
            return 0.0
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((values - mean) / std) ** 4) - 3
        return kurtosis
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that might be generated."""
        feature_names = []
        
        if self.enable_statistical_features:
            feature_names.extend([
                "mean_value", "std_value", "min_value", "max_value", "range_value",
                "median_value", "q25_value", "q75_value", "skewness", "kurtosis",
                "numeric_count", "total_fields", "numeric_ratio"
            ])
        
        if self.enable_temporal_features:
            feature_names.extend([
                "hour_of_day", "day_of_week", "inter_event_time", "events_per_minute",
                "trend_slope", "trend_direction", "source_frequency", "avg_inter_event_time"
            ])
        
        if self.enable_categorical_features:
            feature_names.extend([
                "status_normal", "status_warning", "status_critical",
                "severity_info", "severity_warning", "severity_error", "severity_critical",
                "data_field_count", "data_string_fields", "data_numeric_fields", "data_bool_fields"
            ])
        
        return feature_names
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        stats = self.feature_stats.copy()
        
        if stats["events_processed"] > 0:
            stats["avg_processing_time_ms"] = (
                stats["processing_time_total"] / stats["events_processed"]
            )
            stats["avg_features_per_event"] = (
                stats["features_generated"] / stats["events_processed"]
            )
        
        stats["history_sizes"] = {
            source: len(history) for source, history in self.feature_history.items()
        }
        
        return stats
    
    def is_healthy(self) -> bool:
        """Check if processor is healthy."""
        # Check error rate
        if self.feature_stats["events_processed"] > 0:
            error_rate = self.feature_stats["errors"] / self.feature_stats["events_processed"]
            if error_rate > 0.1:  # More than 10% errors
                return False
        
        # Check processing time
        if self.feature_stats["events_processed"] > 0:
            avg_time = (
                self.feature_stats["processing_time_total"] / 
                self.feature_stats["events_processed"]
            )
            if avg_time > 1000:  # More than 1 second average
                return False
        
        return True
    
    def reset_stats(self):
        """Reset processor statistics."""
        self.feature_stats = {
            "events_processed": 0,
            "features_generated": 0,
            "processing_time_total": 0.0,
            "errors": 0
        }
        self.feature_history.clear()
