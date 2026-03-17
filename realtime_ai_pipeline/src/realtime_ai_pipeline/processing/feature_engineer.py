"""
Advanced feature engineering for complex data patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for complex pattern recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engineering_config = config.get("engineering", {})
        
        # Window configurations
        self.window_sizes = self.engineering_config.get("window_sizes", [5, 10, 20])
        self.max_history = max(self.window_sizes) if self.window_sizes else 20
        
        # Feature groups
        self.enable_lag_features = self.engineering_config.get("lag_features", True)
        self.enable_rolling_features = self.engineering_config.get("rolling_features", True)
        self.enable_interaction_features = self.engineering_config.get("interaction_features", True)
        self.enable_frequency_features = self.engineering_config.get("frequency_features", True)
        
        # Data storage for windowed features
        self.data_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.feature_cache: Dict[str, Any] = {}
        
        # Feature registry
        self.feature_registry = self._build_feature_registry()
    
    def _build_feature_registry(self) -> Dict[str, callable]:
        """Build registry of feature engineering functions."""
        return {
            # Statistical features
            "mean": lambda x: np.mean(x) if len(x) > 0 else 0.0,
            "std": lambda x: np.std(x) if len(x) > 0 else 0.0,
            "min": lambda x: np.min(x) if len(x) > 0 else 0.0,
            "max": lambda x: np.max(x) if len(x) > 0 else 0.0,
            "median": lambda x: np.median(x) if len(x) > 0 else 0.0,
            "range": lambda x: np.max(x) - np.min(x) if len(x) > 0 else 0.0,
            
            # Distribution features
            "skew": lambda x: self._calculate_skewness(x),
            "kurtosis": lambda x: self._calculate_kurtosis(x),
            "q25": lambda x: np.percentile(x, 25) if len(x) > 0 else 0.0,
            "q75": lambda x: np.percentile(x, 75) if len(x) > 0 else 0.0,
            "iqr": lambda x: np.percentile(x, 75) - np.percentile(x, 25) if len(x) > 0 else 0.0,
            
            # Trend features
            "trend_slope": lambda x: self._calculate_trend_slope(x),
            "trend_r2": lambda x: self._calculate_trend_r2(x),
            
            # Change features
            "first_diff": lambda x: self._calculate_first_diff(x),
            "pct_change": lambda x: self._calculate_pct_change(x),
            "acceleration": lambda x: self._calculate_acceleration(x),
        }
    
    def engineer_features(self, event_data: Dict[str, Any], event_source: str) -> Dict[str, float]:
        """Engineer advanced features from event data."""
        features = {}
        
        try:
            # Extract numeric values
            numeric_data = self._extract_numeric_data(event_data)
            
            if not numeric_data:
                return features
            
            # Update data windows
            self._update_data_windows(event_source, numeric_data)
            
            # Generate windowed features
            window_features = self._generate_windowed_features(event_source)
            features.update(window_features)
            
            # Generate lag features
            if self.enable_lag_features:
                lag_features = self._generate_lag_features(event_source)
                features.update(lag_features)
            
            # Generate interaction features
            if self.enable_interaction_features:
                interaction_features = self._generate_interaction_features(numeric_data)
                features.update(interaction_features)
            
            # Generate frequency features
            if self.enable_frequency_features:
                freq_features = self._generate_frequency_features(event_source)
                features.update(freq_features)
            
            # Generate cross-correlation features
            correlation_features = self._generate_correlation_features(event_source)
            features.update(correlation_features)
            
            logger.debug(f"Engineered {len(features)} features for {event_source}")
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
        
        return features
    
    def _extract_numeric_data(self, event_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric values from event data."""
        numeric_data = {}
        
        for key, value in event_data.items():
            if isinstance(value, (int, float)):
                numeric_data[key] = float(value)
            elif isinstance(value, str):
                # Try to parse numeric strings
                try:
                    numeric_data[f"{key}_numeric"] = float(value)
                except ValueError:
                    pass
            elif isinstance(value, bool):
                numeric_data[f"{key}_bool"] = float(value)
        
        return numeric_data
    
    def _update_data_windows(self, source: str, numeric_data: Dict[str, float]):
        """Update sliding windows for feature calculation."""
        for key, value in numeric_data.items():
            window_key = f"{source}_{key}"
            self.data_windows[window_key].append(value)
    
    def _generate_windowed_features(self, source: str) -> Dict[str, float]:
        """Generate features using sliding windows."""
        features = {}
        
        for window_key, data_window in self.data_windows.items():
            if not data_window:
                continue
            
            # Skip if not from this source
            if not window_key.startswith(f"{source}_"):
                continue
            
            feature_name = window_key.replace(f"{source}_", "")
            data_array = np.array(list(data_window))
            
            # Generate features for each window size
            for window_size in self.window_sizes:
                if len(data_array) >= window_size:
                    window_data = data_array[-window_size:]
                    
                    # Apply feature functions
                    for feat_name, feat_func in self.feature_registry.items():
                        try:
                            value = feat_func(window_data)
                            features[f"{feature_name}_{feat_name}_w{window_size}"] = value
                        except Exception as e:
                            logger.debug(f"Error calculating {feat_name}: {e}")
        
        return features
    
    def _generate_lag_features(self, source: str) -> Dict[str, float]:
        """Generate lag features."""
        features = {}
        
        for window_key, data_window in self.data_windows.items():
            if not data_window.startswith(f"{source}_"):
                continue
            
            data_array = list(data_window)
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                if len(data_array) > lag:
                    current_value = data_array[-1]
                    lag_value = data_array[-lag-1]
                    
                    feature_name = window_key.replace(f"{source}_", "")
                    features[f"{feature_name}_lag{lag}"] = current_value - lag_value
                    features[f"{feature_name}_lag{lag}_ratio"] = (
                        current_value / lag_value if lag_value != 0 else 1.0
                    )
        
        return features
    
    def _generate_interaction_features(self, numeric_data: Dict[str, float]) -> Dict[str, float]:
        """Generate interaction features between numeric fields."""
        features = {}
        keys = list(numeric_data.keys())
        
        # Pairwise interactions
        for i, key1 in enumerate(keys):
            for key2 in keys[i+1:]:
                value1 = numeric_data[key1]
                value2 = numeric_data[key2]
                
                # Multiplicative interaction
                features[f"{key1}_x_{key2}"] = value1 * value2
                
                # Additive interaction
                features[f"{key1}_plus_{key2}"] = value1 + value2
                
                # Ratio interaction
                features[f"{key1}_div_{key2}"] = value1 / value2 if value2 != 0 else 0.0
                
                # Difference interaction
                features[f"{key1}_minus_{key2}"] = value1 - value2
        
        return features
    
    def _generate_frequency_features(self, source: str) -> Dict[str, float]:
        """Generate frequency-domain features."""
        features = {}
        
        for window_key, data_window in self.data_windows.items():
            if not window_key.startswith(f"{source}_") or len(data_window) < 4:
                continue
            
            try:
                data_array = np.array(list(data_window))
                feature_name = window_key.replace(f"{source}_", "")
                
                # FFT-based features
                fft = np.fft.fft(data_array)
                fft_freq = np.fft.fftfreq(len(data_array))
                
                # Power spectral density
                psd = np.abs(fft) ** 2
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(psd[1:len(psd)//2]) + 1
                features[f"{feature_name}_dominant_freq"] = fft_freq[dominant_freq_idx]
                
                # Spectral entropy
                psd_norm = psd / np.sum(psd)
                spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                features[f"{feature_name}_spectral_entropy"] = spectral_entropy
                
                # Energy in different frequency bands
                total_energy = np.sum(psd)
                low_freq_energy = np.sum(psd[:len(psd)//4])
                high_freq_energy = np.sum(psd[3*len(psd)//4:])
                
                features[f"{feature_name}_low_freq_ratio"] = low_freq_energy / total_energy
                features[f"{feature_name}_high_freq_ratio"] = high_freq_energy / total_energy
                
            except Exception as e:
                logger.debug(f"Error calculating frequency features for {window_key}: {e}")
        
        return features
    
    def _generate_correlation_features(self, source: str) -> Dict[str, float]:
        """Generate cross-correlation features between different data streams."""
        features = {}
        
        # Get all data windows for this source
        source_windows = {
            key: list(window) for key, window in self.data_windows.items()
            if key.startswith(f"{source}_") and len(window) >= 3
        }
        
        if len(source_windows) < 2:
            return features
        
        # Calculate correlations between all pairs
        window_keys = list(source_windows.keys())
        
        for i, key1 in enumerate(window_keys):
            for key2 in window_keys[i+1:]:
                data1 = np.array(source_windows[key1])
                data2 = np.array(source_windows[key2])
                
                # Ensure same length
                min_len = min(len(data1), len(data2))
                if min_len < 3:
                    continue
                
                data1 = data1[-min_len:]
                data2 = data2[-min_len:]
                
                try:
                    # Pearson correlation
                    correlation = np.corrcoef(data1, data2)[0, 1]
                    if not np.isnan(correlation):
                        name1 = key1.replace(f"{source}_", "")
                        name2 = key2.replace(f"{source}_", "")
                        features[f"corr_{name1}_{name2}"] = correlation
                
                except Exception as e:
                    logger.debug(f"Error calculating correlation: {e}")
        
        return features
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of values."""
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((values - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of values."""
        if len(values) < 4:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((values - mean) / std) ** 4) - 3
        return kurtosis
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate linear trend slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except:
            return 0.0
    
    def _calculate_trend_r2(self, values: np.ndarray) -> float:
        """Calculate R-squared of linear trend."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        try:
            coeffs = np.polyfit(x, values, 1)
            trend = np.polyval(coeffs, x)
            ss_res = np.sum((values - trend) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            return r2
        except:
            return 0.0
    
    def _calculate_first_diff(self, values: np.ndarray) -> float:
        """Calculate first difference."""
        if len(values) < 2:
            return 0.0
        
        return values[-1] - values[-2]
    
    def _calculate_pct_change(self, values: np.ndarray) -> float:
        """Calculate percentage change."""
        if len(values) < 2 or values[-2] == 0:
            return 0.0
        
        return (values[-1] - values[-2]) / values[-2]
    
    def _calculate_acceleration(self, values: np.ndarray) -> float:
        """Calculate acceleration (second derivative)."""
        if len(values) < 3:
            return 0.0
        
        first_diff1 = values[-1] - values[-2]
        first_diff2 = values[-2] - values[-3]
        
        return first_diff1 - first_diff2
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (mock implementation)."""
        # In real implementation, would use feature importance from model
        importance = {}
        
        for feature_name in self.feature_registry.keys():
            importance[feature_name] = np.random.random()  # Mock importance
        
        return importance
    
    def clear_cache(self):
        """Clear feature cache and reset data windows."""
        self.feature_cache.clear()
        self.data_windows.clear()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        total_values = sum(len(window) for window in self.data_windows.values())
        return {
            "data_windows_count": len(self.data_windows),
            "total_stored_values": total_values,
            "cache_size": len(self.feature_cache)
        }
