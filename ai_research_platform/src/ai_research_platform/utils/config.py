"""Configuration management for AI Research Platform."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json


class Config:
    """Configuration manager for the AI Research Platform."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or os.getenv("AI_RESEARCH_CONFIG", "config.json")
        self.config = self._load_default_config()
        
        # Load custom config if exists
        if os.path.exists(self.config_path):
            self._load_config()
        
        # Create results directory
        self.results_dir = Path(self.config["results"]["storage_path"])
        self.results_dir.mkdir(exist_ok=True, parents=True)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "results": {
                "storage_path": "results",
                "format": "json"
            },
            "experiments": {
                "default_test_size": 0.2,
                "random_state": 42,
                "cross_validation_folds": 5
            },
            "models": {
                "default_hyperparameters": {
                    "random_forest": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "random_state": 42
                    },
                    "linear_regression": {},
                    "gradient_boosting": {
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "random_state": 42
                    }
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                custom_config = json.load(f)
                self._merge_config(self.config, custom_config)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
    
    def _merge_config(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'models.default_hyperparameters.random_forest')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration (default: self.config_path)
        """
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)


# Global configuration instance
config = Config()
