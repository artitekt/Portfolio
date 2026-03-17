"""
Logging utilities for real-time AI pipeline.
"""

import logging
import sys
import time
from typing import Optional, Dict, Any
from pathlib import Path
import json


class PipelineLogger:
    """Enhanced logger for pipeline operations."""
    
    def __init__(self, name: str = "pipeline", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.setup_logger()
        
        # Performance tracking
        self.performance_data = []
        self.max_performance_entries = 1000
    
    def setup_logger(self):
        """Setup logger with configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        level = self.config.get('level', 'INFO')
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        if self.config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.get('file'):
            file_path = Path(self.config['file'])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # JSON handler for structured logging
        if self.config.get('json_file'):
            json_handler = JSONFileHandler(self.config['json_file'])
            self.logger.addHandler(json_handler)
    
    def log_event(self, event_id: str, stage: str, message: str, level: str = "INFO", **kwargs):
        """Log pipeline event with structured data."""
        log_data = {
            "event_id": event_id,
            "stage": stage,
            "message": message,
            "timestamp": time.time(),
            **kwargs
        }
        
        # Log with standard logger
        log_method = getattr(self.logger, level.lower())
        log_method(f"[{stage}] {message} (event: {event_id})")
        
        # Add to performance tracking
        if stage in ["feature_processing", "inference", "publishing"]:
            self.performance_data.append(log_data)
            if len(self.performance_data) > self.max_performance_entries:
                self.performance_data = self.performance_data[-self.max_performance_entries:]
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics."""
        perf_data = {
            "operation": operation,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
            **kwargs
        }
        
        self.logger.info(f"Performance: {operation} took {duration_ms:.2f}ms")
        self.performance_data.append(perf_data)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context."""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "context": context or {}
        }
        
        self.logger.error(f"Error: {error_data}", exc_info=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_data:
            return {}
        
        # Group by operation
        operations = {}
        for entry in self.performance_data:
            op = entry.get('operation', 'unknown')
            if op not in operations:
                operations[op] = []
            operations[op].append(entry.get('duration_ms', 0))
        
        stats = {}
        for op, durations in operations.items():
            if durations:
                stats[op] = {
                    "count": len(durations),
                    "avg_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "total_ms": sum(durations)
                }
        
        return stats
    
    def clear_performance_data(self):
        """Clear performance data."""
        self.performance_data.clear()


class JSONFileHandler(logging.Handler):
    """JSON file handler for structured logging."""
    
    def __init__(self, filename: str):
        super().__init__()
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record):
        """Emit log record as JSON."""
        try:
            log_entry = {
                "timestamp": time.time(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            
            # Write to file
            with open(self.filename, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                f.flush()
        
        except Exception:
            self.handleError(record)


class PerformanceTracker:
    """Context manager for tracking performance."""
    
    def __init__(self, logger: PipelineLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.logger.log_performance(self.operation, duration_ms, **self.context)
        
        return False  # Don't suppress exceptions


class MetricsLogger:
    """Logger for pipeline metrics."""
    
    def __init__(self, logger: PipelineLogger):
        self.logger = logger
        self.metrics = {}
        self.counters = {}
    
    def increment_counter(self, name: str, value: int = 1, **tags):
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) + value
        
        self.logger.logger.info(f"Metric: {name} = {self.counters[key]} {tags}")
    
    def set_gauge(self, name: str, value: float, **tags):
        """Set a gauge metric."""
        key = self._make_key(name, tags)
        self.metrics[key] = value
        
        self.logger.logger.info(f"Gauge: {name} = {value} {tags}")
    
    def record_timer(self, name: str, duration_ms: float, **tags):
        """Record a timer metric."""
        key = self._make_key(name, tags)
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(duration_ms)
        
        self.logger.logger.info(f"Timer: {name} = {duration_ms:.2f}ms {tags}")
    
    def _make_key(self, name: str, tags: Dict[str, Any]) -> str:
        """Create metric key from name and tags."""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}[{tag_str}]"
        return name
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            "counters": self.counters.copy(),
            "gauges": {},
            "timers": {}
        }
        
        # Process gauge metrics
        for key, value in self.metrics.items():
            if isinstance(value, list):
                # Timer metric
                summary["timers"][key] = {
                    "count": len(value),
                    "avg": sum(value) / len(value),
                    "min": min(value),
                    "max": max(value)
                }
            else:
                # Gauge metric
                summary["gauges"][key] = value
        
        return summary


def setup_logging(config: Dict[str, Any]) -> PipelineLogger:
    """Setup pipeline logging."""
    logger_config = config.get('logging', {})
    
    # Create pipeline logger
    pipeline_logger = PipelineLogger("pipeline", logger_config)
    
    # Setup other loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    return pipeline_logger


def get_logger(name: str = "pipeline") -> PipelineLogger:
    """Get pipeline logger."""
    return PipelineLogger(name)


# Context manager for performance tracking
def track_performance(operation: str, logger: Optional[PipelineLogger] = None, **context):
    """Track performance of an operation."""
    if logger is None:
        logger = get_logger()
    
    return PerformanceTracker(logger, operation, **context)


# Decorator for performance tracking
def performance_logged(operation: str = None):
    """Decorator to log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            logger = get_logger()
            
            with track_performance(op_name, logger):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator
