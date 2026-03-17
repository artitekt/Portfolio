"""
Configuration management for real-time AI pipeline.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the pipeline."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        self.config_path = config_path
        self.config_data = {}
        
        # Load configuration
        if config_dict:
            self.config_data = config_dict.copy()
        elif config_path:
            self.load_from_file(config_path)
        else:
            self.config_data = self._get_default_config()
        
        # Override with environment variables
        self._load_from_env()
    
    def load_from_file(self, path: str):
        """Load configuration from file."""
        try:
            config_file = Path(path)
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {path}, using defaults")
                self.config_data = self._get_default_config()
                return
            
            with open(config_file, 'r') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    self.config_data = yaml.safe_load(f)
                elif path.endswith('.json'):
                    self.config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {path}")
            
            logger.info(f"Loaded configuration from {path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            self.config_data = self._get_default_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Pipeline configuration
        if os.getenv('PIPELINE_EVENT_RATE'):
            self.set('data_source.event_rate', float(os.getenv('PIPELINE_EVENT_RATE')))
        
        if os.getenv('PIPELINE_BATCH_SIZE'):
            self.set('publisher.batch_size', int(os.getenv('PIPELINE_BATCH_SIZE')))
        
        if os.getenv('PIPELINE_LOG_LEVEL'):
            self.set('logging.level', os.getenv('PIPELINE_LOG_LEVEL'))
        
        # Model configuration
        if os.getenv('MODEL_TYPE'):
            self.set('inference.model.type', os.getenv('MODEL_TYPE'))
        
        if os.getenv('MODEL_PATH'):
            self.set('inference.model.path', os.getenv('MODEL_PATH'))
        
        # Publisher configuration
        if os.getenv('ENABLE_CONSOLE'):
            self.set('publisher.console', os.getenv('ENABLE_CONSOLE').lower() == 'true')
        
        if os.getenv('ENABLE_FILE'):
            self.set('publisher.file', os.getenv('ENABLE_FILE').lower() == 'true')
        
        if os.getenv('ENABLE_WEBHOOK'):
            self.set('publisher.webhook', os.getenv('ENABLE_WEBHOOK').lower() == 'true')
        
        if os.getenv('WEBHOOK_URL'):
            self.set('publisher.webhook_url', os.getenv('WEBHOOK_URL'))
        
        if os.getenv('METRICS_PORT'):
            self.set('publisher.metrics_port', int(os.getenv('METRICS_PORT')))
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "pipeline": {
                "name": "realtime_ai_pipeline",
                "version": "1.0.0"
            },
            "data_source": {
                "event_rate": 1.0,
                "event_types": ["sensor", "user_action", "system_event"],
                "fields": ["value", "status", "metadata"],
                "type": "simulated"
            },
            "processor": {
                "features": {
                    "statistical": True,
                    "temporal": True,
                    "categorical": True,
                    "window_size": 10
                },
                "engineering": {
                    "window_sizes": [5, 10, 20],
                    "lag_features": True,
                    "rolling_features": True,
                    "interaction_features": True,
                    "frequency_features": True
                }
            },
            "inference": {
                "model": {
                    "type": "mock",
                    "path": None,
                    "input_size": 10,
                    "confidence_threshold": 0.5,
                    "batch_size": 1,
                    "enable_batching": False,
                    "max_batch_delay": 0.01
                },
                "mock": {
                    "noise_level": 0.1
                }
            },
            "publisher": {
                "console": True,
                "file": False,
                "file_path": "results.jsonl",
                "webhook": False,
                "webhook_url": None,
                "webhook_timeout": 5.0,
                "metrics": True,
                "metrics_port": 8080,
                "metrics_endpoint": "/metrics",
                "batch_size": 100,
                "flush_interval": 0.1,
                "max_queue_size": 10000
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (dot notation supported)."""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (dot notation supported)."""
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config_data, updates)
    
    def save_to_file(self, path: str):
        """Save configuration to file."""
        try:
            config_file = Path(path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                elif path.endswith('.json'):
                    json.dump(self.config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {path}")
            
            logger.info(f"Saved configuration to {path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")
    
    def validate(self) -> bool:
        """Validate configuration."""
        try:
            # Check required sections
            required_sections = ['data_source', 'processor', 'inference', 'publisher']
            for section in required_sections:
                if section not in self.config_data:
                    logger.error(f"Missing required section: {section}")
                    return False
            
            # Validate data source
            event_rate = self.get('data_source.event_rate')
            if not isinstance(event_rate, (int, float)) or event_rate <= 0:
                logger.error("Invalid event_rate in data_source")
                return False
            
            # Validate inference
            model_type = self.get('inference.model.type')
            if model_type not in ['mock', 'sklearn', 'onnx', 'tensorflow', 'pytorch']:
                logger.error(f"Invalid model type: {model_type}")
                return False
            
            # Validate publisher
            metrics_port = self.get('publisher.metrics_port')
            if metrics_port is not None and (not isinstance(metrics_port, int) or metrics_port < 1 or metrics_port > 65535):
                logger.error("Invalid metrics_port in publisher")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration."""
        return self.get('data_source', {})
    
    def get_processor_config(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return self.get('processor', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self.get('inference', {})
    
    def get_publisher_config(self) -> Dict[str, Any]:
        """Get publisher configuration."""
        return self.get('publisher', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config_data.copy()
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config_data, indent=2)


class ConfigManager:
    """Global configuration manager."""
    
    def __init__(self):
        self._config: Optional[Config] = None
    
    def load_config(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None) -> Config:
        """Load configuration."""
        self._config = Config(config_path, config_dict)
        
        if not self._config.validate():
            raise ValueError("Invalid configuration")
        
        return self._config
    
    def get_config(self) -> Optional[Config]:
        """Get current configuration."""
        return self._config
    
    def reload_config(self) -> bool:
        """Reload configuration from file."""
        if self._config and self._config.config_path:
            try:
                self._config.load_from_file(self._config.config_path)
                return self._config.validate()
            except Exception as e:
                logger.error(f"Error reloading config: {e}")
                return False
        return False


# Global config manager instance
_config_manager = ConfigManager()


def get_config() -> Optional[Config]:
    """Get global configuration."""
    return _config_manager.get_config()


def load_config(config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None) -> Config:
    """Load global configuration."""
    return _config_manager.load_config(config_path, config_dict)


def setup_logging(config: Config):
    """Setup logging based on configuration."""
    logging_config = config.get_logging_config()
    
    # Set log level
    level = getattr(logging, logging_config.get('level', 'INFO').upper())
    logging.basicConfig(level=level, format=logging_config.get('format'))
    
    # Add file handler if specified
    log_file = logging_config.get('file')
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(logging_config.get('format')))
        logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging setup complete - Level: {logging_config.get('level', 'INFO')}")


# Environment-specific configurations
def get_development_config() -> Dict[str, Any]:
    """Get development configuration."""
    config = Config()._get_default_config()
    config.update({
        "data_source": {
            "event_rate": 0.5,  # Slower for development
            "event_types": ["sensor", "user_action"]
        },
        "publisher": {
            "console": True,
            "file": True,
            "file_path": "dev_results.jsonl",
            "metrics": True,
            "metrics_port": 8080
        },
        "logging": {
            "level": "DEBUG",
            "file": "pipeline.log"
        }
    })
    return config


def get_production_config() -> Dict[str, Any]:
    """Get production configuration."""
    config = Config()._get_default_config()
    config.update({
        "data_source": {
            "event_rate": 10.0,  # Higher throughput for production
            "type": "api"  # Real data source
        },
        "publisher": {
            "console": False,  # No console output in production
            "file": True,
            "file_path": "/var/log/pipeline/results.jsonl",
            "webhook": True,
            "webhook_url": os.getenv("WEBHOOK_URL"),
            "metrics": True,
            "metrics_port": 9090
        },
        "logging": {
            "level": "INFO",
            "file": "/var/log/pipeline/pipeline.log"
        }
    })
    return config


def get_testing_config() -> Dict[str, Any]:
    """Get testing configuration."""
    config = Config()._get_default_config()
    config.update({
        "data_source": {
            "event_rate": 2.0,
            "event_types": ["test_event"]
        },
        "publisher": {
            "console": False,
            "file": False,
            "webhook": False,
            "metrics": False
        },
        "logging": {
            "level": "WARNING"
        }
    })
    return config
