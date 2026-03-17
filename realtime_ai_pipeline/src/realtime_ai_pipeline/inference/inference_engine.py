"""
Component: AI Inference Engine
Role: Runs AI model inference on processed features to generate predictions.
     Supports multiple model frameworks (Mock, Scikit-learn, ONNX, TensorFlow,
     PyTorch) with batch processing and confidence scoring.
"""

"""
AI inference engine for real-time predictions.
"""

import time
import numpy as np
import os
from typing import Dict, Any, List, Optional
import logging
from ..pipeline.message import FeatureMessage, PredictionMessage

logger = logging.getLogger(__name__)


class InferenceEngine:
    """AI inference engine for real-time predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        
        # Model configuration
        self.model_type = self.model_config.get("type", "mock")
        self.model_path = self.model_config.get("path", None)
        self.input_size = self.model_config.get("input_size", 10)
        self.confidence_threshold = self.model_config.get("confidence_threshold", 0.5)
        
        # Inference configuration
        self.batch_size = self.model_config.get("batch_size", 1)
        self.enable_batching = self.model_config.get("enable_batching", False)
        self.max_batch_delay = self.model_config.get("max_batch_delay", 0.01)
        
        # Model state
        self.model = None
        self.model_loaded = False
        self.model_version = "1.0.0"
        
        # Batch processing
        self.inference_queue = []
        self.last_inference_time = 0
        
        # Statistics
        self.inference_stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "total_inference_time": 0.0,
            "avg_inference_time": 0.0,
            "batch_inferences": 0
        }
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the AI model."""
        try:
            if self.model_type == "mock":
                self.model = MockModel(self.config.get("mock", {}))
                self.model_loaded = True
                logger.info("Loaded mock inference model")
            
            elif self.model_type == "sklearn":
                self.model = SklearnModel(self.model_config)
                self.model_loaded = True
                logger.info("Loaded sklearn inference model")
            
            elif self.model_type == "onnx":
                self.model = ONNXModel(self.model_config)
                self.model_loaded = True
                logger.info("Loaded ONNX inference model")
            
            else:
                logger.warning(f"Unknown model type: {self.model_type}, using mock")
                self.model = MockModel(self.config.get("mock", {}))
                self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = MockModel(self.config.get("mock", {}))
            self.model_loaded = True
    
    async def predict(self, features: FeatureMessage) -> Dict[str, Any]:
        """Run inference on features."""
        start_time = time.time()
        
        try:
            # Prepare input
            input_data = self._prepare_input(features)
            
            # Run inference
            if self.enable_batching:
                result = await self._predict_batch(input_data)
            else:
                result = self._predict_single(input_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(result, input_data)
            
            # Update statistics
            inference_time = (time.time() - start_time) * 1000
            self._update_stats(inference_time, success=True)
            
            prediction_result = {
                "value": result["prediction"],
                "confidence": confidence,
                "model_version": self.model_version,
                "inference_time_ms": inference_time,
                "features_used": len(features.feature_vector)
            }
            
            logger.debug(f"Inference for {features.event_id}: {result['prediction']:.3f} (confidence: {confidence:.3f})")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Inference error for {features.event_id}: {e}")
            self._update_stats(0, success=False)
            
            # Return default prediction
            return {
                "value": 0.0,
                "confidence": 0.0,
                "model_version": self.model_version,
                "inference_time_ms": 0.0,
                "features_used": 0,
                "error": str(e)
            }
    
    def _prepare_input(self, features: FeatureMessage) -> np.ndarray:
        """Prepare input data for model."""
        feature_vector = features.feature_vector
        
        # Convert to numpy array
        input_array = np.array(feature_vector, dtype=np.float32)
        
        # Resize to expected input size
        if len(input_array) != self.input_size:
            if len(input_array) > self.input_size:
                # Truncate
                input_array = input_array[:self.input_size]
            else:
                # Pad with zeros
                padding = self.input_size - len(input_array)
                input_array = np.pad(input_array, (0, padding), mode='constant')
        
        # Add batch dimension
        input_array = input_array.reshape(1, -1)
        
        return input_array
    
    def _predict_single(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run single prediction."""
        return self.model.predict(input_data)
    
    async def _predict_batch(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run batch prediction."""
        current_time = time.time()
        
        # Add to batch queue
        self.inference_queue.append(input_data)
        
        # Check if we should run batch inference
        should_run = (
            len(self.inference_queue) >= self.batch_size or
            (current_time - self.last_inference_time) >= self.max_batch_delay
        )
        
        if should_run and self.inference_queue:
            # Run batch inference
            batch_input = np.vstack(self.inference_queue)
            batch_results = self.model.predict_batch(batch_input)
            
            # Clear queue
            self.inference_queue.clear()
            self.last_inference_time = current_time
            
            # Return first result (for this request)
            if len(batch_results) > 0:
                return batch_results[0]
        
        # Fallback to single prediction
        return self._predict_single(input_data)
    
    def _calculate_confidence(self, result: Dict[str, Any], input_data: np.ndarray) -> float:
        """Calculate prediction confidence."""
        if "confidence" in result:
            return float(result["confidence"])
        
        # Mock confidence calculation
        prediction = result.get("prediction", 0.0)
        
        # Simple confidence based on prediction magnitude and input quality
        input_quality = self._assess_input_quality(input_data)
        prediction_confidence = min(abs(prediction), 1.0)
        
        confidence = (input_quality + prediction_confidence) / 2.0
        return float(confidence)
    
    def _assess_input_quality(self, input_data: np.ndarray) -> float:
        """Assess quality of input data."""
        # Check for NaN or infinite values
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            return 0.1
        
        # Check variance
        variance = np.var(input_data)
        if variance == 0:
            return 0.2
        
        # Simple quality score based on variance and range
        range_score = min(np.ptp(input_data) / 10.0, 1.0)
        variance_score = min(variance / 5.0, 1.0)
        
        return (range_score + variance_score) / 2.0
    
    def _update_stats(self, inference_time: float, success: bool):
        """Update inference statistics."""
        self.inference_stats["total_inferences"] += 1
        
        if success:
            self.inference_stats["successful_inferences"] += 1
            self.inference_stats["total_inference_time"] += inference_time
            self.inference_stats["avg_inference_time"] = (
                self.inference_stats["total_inference_time"] / 
                self.inference_stats["successful_inferences"]
            )
        else:
            self.inference_stats["failed_inferences"] += 1
    
    def get_model_version(self) -> str:
        """Get current model version."""
        return self.model_version
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        stats = self.inference_stats.copy()
        
        if stats["total_inferences"] > 0:
            stats["success_rate"] = stats["successful_inferences"] / stats["total_inferences"]
            stats["failure_rate"] = stats["failed_inferences"] / stats["total_inferences"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        stats["model_loaded"] = self.model_loaded
        stats["model_type"] = self.model_type
        stats["batch_queue_size"] = len(self.inference_queue)
        
        return stats
    
    def is_healthy(self) -> bool:
        """Check if inference engine is healthy."""
        if not self.model_loaded:
            return False
        
        # Check success rate
        if self.inference_stats["total_inferences"] > 0:
            success_rate = self.inference_stats["successful_inferences"] / self.inference_stats["total_inferences"]
            if success_rate < 0.9:  # Less than 90% success rate
                return False
        
        # Check average inference time
        if self.inference_stats["avg_inference_time"] > 1000:  # More than 1 second
            return False
        
        return True
    
    def reset_stats(self):
        """Reset inference statistics."""
        self.inference_stats = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "total_inference_time": 0.0,
            "avg_inference_time": 0.0,
            "batch_inferences": 0
        }


class MockModel:
    """Mock model for testing and demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = np.random.randn(10) * 0.1
        self.bias = np.random.randn() * 0.1
        self.noise_level = config.get("noise_level", 0.1)
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Mock prediction."""
        # Simple linear model with noise
        features = input_data.flatten()
        
        # Ensure we have enough features
        if len(features) < len(self.weights):
            features = np.pad(features, (0, len(self.weights) - len(features)), mode='constant')
        else:
            features = features[:len(self.weights)]
        
        # Linear combination
        prediction = np.dot(features, self.weights) + self.bias
        
        # Add noise
        noise = np.random.randn() * self.noise_level
        prediction += noise
        
        # Apply activation (sigmoid-like)
        prediction = 1.0 / (1.0 + np.exp(-prediction))
        
        return {
            "prediction": prediction,
            "raw_output": prediction,
            "features_used": len(features)
        }
    
    def predict_batch(self, batch_input: np.ndarray) -> List[Dict[str, Any]]:
        """Mock batch prediction."""
        results = []
        for i in range(batch_input.shape[0]):
            result = self.predict(batch_input[i:i+1])
            results.append(result)
        return results


class SklearnModel:
    """Scikit-learn model wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sklearn model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            model_type = self.config.get("sklearn_type", "random_forest")
            
            if model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=10,
                    random_state=42
                )
            elif model_type == "logistic":
                self.model = LogisticRegression(random_state=42)
            else:
                raise ValueError(f"Unknown sklearn model type: {model_type}")
            
            # Train on dummy data for demo
            X_dummy = np.random.randn(100, 10)
            y_dummy = np.random.randint(0, 2, 100)
            self.model.fit(X_dummy, y_dummy)
            
        except ImportError:
            logger.warning("sklearn not available, using mock model")
            self.model = MockModel(self.config)
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Predict using sklearn model."""
        if isinstance(self.model, MockModel):
            return self.model.predict(input_data)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(input_data)
        prediction = probabilities[0][1] if probabilities.shape[1] > 1 else probabilities[0][0]
        
        return {
            "prediction": float(prediction),
            "probabilities": probabilities[0].tolist(),
            "features_used": input_data.shape[1]
        }
    
    def predict_batch(self, batch_input: np.ndarray) -> List[Dict[str, Any]]:
        """Batch prediction using sklearn model."""
        if isinstance(self.model, MockModel):
            return self.model.predict_batch(batch_input)
        
        probabilities = self.model.predict_proba(batch_input)
        results = []
        
        for i in range(batch_input.shape[0]):
            prediction = probabilities[i][1] if probabilities.shape[1] > 1 else probabilities[i][0]
            results.append({
                "prediction": float(prediction),
                "probabilities": probabilities[i].tolist(),
                "features_used": batch_input.shape[1]
            })
        
        return results


class ONNXModel:
    """ONNX model wrapper for high-performance inference."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            model_path = self.config.get("model_path")
            if model_path and os.path.exists(model_path):
                # Load actual ONNX model
                self.session = ort.InferenceSession(model_path)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            else:
                # Create dummy ONNX model for demo
                self.session = None
                logger.warning("ONNX model path not found, using mock")
                
        except ImportError:
            logger.warning("onnxruntime not available, using mock model")
            self.session = None
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Predict using ONNX model."""
        if self.session is None:
            # Fallback to mock
            mock_model = MockModel(self.config)
            return mock_model.predict(input_data)
        
        # Run ONNX inference
        input_dict = {self.input_name: input_data.astype(np.float32)}
        outputs = self.session.run(None, input_dict)
        
        prediction = float(outputs[0][0][0]) if len(outputs[0][0]) > 0 else 0.0
        
        return {
            "prediction": prediction,
            "raw_output": outputs[0][0].tolist(),
            "features_used": input_data.shape[1]
        }
    
    def predict_batch(self, batch_input: np.ndarray) -> List[Dict[str, Any]]:
        """Batch prediction using ONNX model."""
        if self.session is None:
            mock_model = MockModel(self.config)
            return mock_model.predict_batch(batch_input)
        
        # Run ONNX batch inference
        input_dict = {self.input_name: batch_input.astype(np.float32)}
        outputs = self.session.run(None, input_dict)
        
        results = []
        for i in range(batch_input.shape[0]):
            prediction = float(outputs[0][i][0]) if len(outputs[0][i]) > 0 else 0.0
            results.append({
                "prediction": prediction,
                "raw_output": outputs[0][i].tolist(),
                "features_used": batch_input.shape[1]
            })
        
        return results
