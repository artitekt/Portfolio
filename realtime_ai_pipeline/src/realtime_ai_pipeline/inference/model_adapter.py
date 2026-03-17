"""
Model adapter for different AI model types and frameworks.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    @abstractmethod
    def load_model(self, model_path: str, config: Dict[str, Any]) -> bool:
        """Load model from path."""
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run prediction on input data."""
        pass
    
    @abstractmethod
    def predict_batch(self, batch_input: np.ndarray) -> List[Dict[str, Any]]:
        """Run batch prediction."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources."""
        pass


class TensorFlowAdapter(ModelAdapter):
    """TensorFlow/Keras model adapter."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> bool:
        """Load TensorFlow model."""
        try:
            import tensorflow as tf
            
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
            
            logger.info(f"Loaded TensorFlow model from {model_path}")
            return True
            
        except ImportError:
            logger.error("TensorFlow not available")
            return False
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run TensorFlow prediction."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # TensorFlow prediction
        predictions = self.model.predict(input_data, verbose=0)
        
        # Handle different output shapes
        if predictions.ndim == 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        
        if predictions.ndim == 0:
            prediction_value = float(predictions)
        elif predictions.ndim == 1 and len(predictions) == 1:
            prediction_value = float(predictions[0])
        else:
            # For multi-output, take first value or mean
            prediction_value = float(np.mean(predictions))
        
        return {
            "prediction": prediction_value,
            "raw_output": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            "output_shape": predictions.shape
        }
    
    def predict_batch(self, batch_input: np.ndarray) -> List[Dict[str, Any]]:
        """Run batch TensorFlow prediction."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        predictions = self.model.predict(batch_input, verbose=0)
        results = []
        
        for i in range(batch_input.shape[0]):
            if predictions.ndim == 2:
                pred = predictions[i]
            else:
                pred = predictions
            
            if pred.ndim == 0:
                prediction_value = float(pred)
            elif pred.ndim == 1 and len(pred) == 1:
                prediction_value = float(pred[0])
            else:
                prediction_value = float(np.mean(pred))
            
            results.append({
                "prediction": prediction_value,
                "raw_output": pred.tolist() if hasattr(pred, 'tolist') else pred,
                "output_shape": pred.shape
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get TensorFlow model information."""
        if not self.model_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "framework": "tensorflow",
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_parameters": self.model.count_params(),
            "summary": str(self.model.summary())
        }
    
    def cleanup(self):
        """Cleanup TensorFlow resources."""
        if self.model is not None:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            self.model = None
            self.model_loaded = False


class PyTorchAdapter(ModelAdapter):
    """PyTorch model adapter."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.model_loaded = False
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> bool:
        """Load PyTorch model."""
        try:
            import torch
            
            self.device = torch.device("cpu")  # Use CPU for consistency
            
            # Load model state dict
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"Loaded PyTorch model from {model_path}")
            return True
            
        except ImportError:
            logger.error("PyTorch not available")
            return False
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run PyTorch prediction."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        import torch
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        # Run prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert to numpy and extract prediction
        if isinstance(output, torch.Tensor):
            output_np = output.cpu().numpy()
        else:
            output_np = np.array(output)
        
        # Handle different output shapes
        if output_np.ndim == 2 and output_np.shape[0] == 1:
            output_np = output_np[0]
        
        if output_np.ndim == 0:
            prediction_value = float(output_np)
        elif output_np.ndim == 1 and len(output_np) == 1:
            prediction_value = float(output_np[0])
        else:
            prediction_value = float(np.mean(output_np))
        
        return {
            "prediction": prediction_value,
            "raw_output": output_np.tolist(),
            "output_shape": output_np.shape
        }
    
    def predict_batch(self, batch_input: np.ndarray) -> List[Dict[str, Any]]:
        """Run batch PyTorch prediction."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        import torch
        
        input_tensor = torch.FloatTensor(batch_input).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        results = []
        
        if isinstance(outputs, torch.Tensor):
            outputs_np = outputs.cpu().numpy()
        else:
            outputs_np = np.array(outputs)
        
        for i in range(batch_input.shape[0]):
            if outputs_np.ndim == 2:
                output = outputs_np[i]
            else:
                output = outputs_np
            
            if output.ndim == 0:
                prediction_value = float(output)
            elif output.ndim == 1 and len(output) == 1:
                prediction_value = float(output[0])
            else:
                prediction_value = float(np.mean(output))
            
            results.append({
                "prediction": prediction_value,
                "raw_output": output.tolist(),
                "output_shape": output.shape
            })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get PyTorch model information."""
        if not self.model_loaded:
            return {"loaded": False}
        
        import torch
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "loaded": True,
            "framework": "pytorch",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_type": type(self.model).__name__
        }
    
    def cleanup(self):
        """Cleanup PyTorch resources."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False


class ScikitLearnAdapter(ModelAdapter):
    """Scikit-learn model adapter."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
    
    def load_model(self, model_path: str, config: Dict[str, Any]) -> bool:
        """Load scikit-learn model."""
        try:
            import joblib
            import pickle
            
            # Try to load with joblib first, then pickle
            try:
                self.model = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            
            self.model_loaded = True
            
            logger.info(f"Loaded scikit-learn model from {model_path}")
            return True
            
        except ImportError:
            logger.error("scikit-learn not available")
            return False
        except Exception as e:
            logger.error(f"Failed to load scikit-learn model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run scikit-learn prediction."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Flatten input if needed
        if input_data.ndim == 2 and input_data.shape[0] == 1:
            input_flat = input_data[0]
        else:
            input_flat = input_data.flatten()
        
        # Try to get prediction probabilities first
        prediction_value = None
        probabilities = None
        
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_flat.reshape(1, -1))
                prediction_value = probabilities[0][1] if probabilities.shape[1] > 1 else probabilities[0][0]
            else:
                prediction_value = self.model.predict(input_flat.reshape(1, -1))[0]
        except:
            # Fallback to predict
            prediction_value = self.model.predict(input_flat.reshape(1, -1))[0]
        
        return {
            "prediction": float(prediction_value),
            "probabilities": probabilities[0].tolist() if probabilities is not None else None,
            "raw_output": float(prediction_value)
        }
    
    def predict_batch(self, batch_input: np.ndarray) -> List[Dict[str, Any]]:
        """Run batch scikit-learn prediction."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        results = []
        
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(batch_input)
                predictions = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            else:
                predictions = self.model.predict(batch_input)
                probabilities = None
            
            for i in range(batch_input.shape[0]):
                result = {
                    "prediction": float(predictions[i]),
                    "probabilities": probabilities[i].tolist() if probabilities is not None else None,
                    "raw_output": float(predictions[i])
                }
                results.append(result)
        
        except Exception as e:
            # Fallback to individual predictions
            for i in range(batch_input.shape[0]):
                result = self.predict(batch_input[i:i+1])
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get scikit-learn model information."""
        if not self.model_loaded:
            return {"loaded": False}
        
        info = {
            "loaded": True,
            "framework": "scikit-learn",
            "model_type": type(self.model).__name__
        }
        
        # Add model-specific information
        if hasattr(self.model, 'n_features_in_'):
            info["n_features"] = self.model.n_features_in_
        
        if hasattr(self.model, 'n_estimators_'):
            info["n_estimators"] = self.model.n_estimators_
        
        if hasattr(self.model, 'feature_importances_'):
            info["has_feature_importances"] = True
        
        return info
    
    def cleanup(self):
        """Cleanup scikit-learn resources."""
        self.model = None
        self.model_loaded = False


class ModelAdapterFactory:
    """Factory for creating model adapters."""
    
    @staticmethod
    def create_adapter(framework: str) -> ModelAdapter:
        """Create model adapter for specified framework."""
        adapters = {
            "tensorflow": TensorFlowAdapter,
            "keras": TensorFlowAdapter,  # Keras uses TensorFlow adapter
            "pytorch": PyTorchAdapter,
            "torch": PyTorchAdapter,    # Alias for PyTorch
            "sklearn": ScikitLearnAdapter,
            "scikit-learn": ScikitLearnAdapter
        }
        
        adapter_class = adapters.get(framework.lower())
        if adapter_class is None:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return adapter_class()
    
    @staticmethod
    def get_supported_frameworks() -> List[str]:
        """Get list of supported frameworks."""
        return ["tensorflow", "keras", "pytorch", "torch", "sklearn", "scikit-learn"]


class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self):
        self.models: Dict[str, ModelAdapter] = {}
        self.default_model: Optional[str] = None
    
    def register_model(self, name: str, adapter: ModelAdapter, model_path: str, config: Dict[str, Any]) -> bool:
        """Register a model."""
        try:
            if adapter.load_model(model_path, config):
                self.models[name] = adapter
                if self.default_model is None:
                    self.default_model = name
                logger.info(f"Registered model: {name}")
                return True
            else:
                logger.error(f"Failed to load model: {name}")
                return False
        except Exception as e:
            logger.error(f"Error registering model {name}: {e}")
            return False
    
    def get_model(self, name: Optional[str] = None) -> Optional[ModelAdapter]:
        """Get model by name or default."""
        if name is None:
            name = self.default_model
        
        return self.models.get(name)
    
    def set_default_model(self, name: str) -> bool:
        """Set default model."""
        if name in self.models:
            self.default_model = name
            return True
        return False
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())
    
    def get_model_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get model information."""
        model = self.get_model(name)
        if model is None:
            return {"error": "Model not found"}
        
        info = model.get_model_info()
        info["name"] = name or self.default_model
        info["is_default"] = (name or self.default_model) == self.default_model
        
        return info
    
    def cleanup_all(self):
        """Cleanup all models."""
        for model in self.models.values():
            try:
                model.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up model: {e}")
        
        self.models.clear()
        self.default_model = None
