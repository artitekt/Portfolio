"""
Enhanced logging utilities for AI Agent Framework.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class AgentLogger:
    """
    Enhanced logger with structured logging capabilities.
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: str = 'INFO'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_decision(self, decision_data: dict) -> None:
        """Log a decision with structured data."""
        self.logger.info(f"DECISION: {decision_data}")
    
    def log_cycle_start(self, cycle_number: int) -> None:
        """Log the start of a cycle."""
        self.logger.debug(f"Cycle #{cycle_number} started")
    
    def log_cycle_complete(self, cycle_number: int, outcome: str) -> None:
        """Log the completion of a cycle."""
        self.logger.debug(f"Cycle #{cycle_number} completed: {outcome}")
    
    def log_llm_call(self, provider: str, model: str, tokens: int, latency: float) -> None:
        """Log an LLM call."""
        self.logger.debug(f"LLM Call: {provider}/{model} - {tokens} tokens, {latency:.0f}ms")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with context."""
        self.logger.error(f"Error in {context}: {error}", exc_info=True)
    
    def log_warning(self, message: str, context: dict = None) -> None:
        """Log a warning with optional context."""
        if context:
            self.logger.warning(f"{message} | Context: {context}")
        else:
            self.logger.warning(message)
    
    def log_startup(self, config_summary: str) -> None:
        """Log agent startup."""
        self.logger.info(f"Agent starting up\n{config_summary}")
    
    def log_shutdown(self, stats: dict) -> None:
        """Log agent shutdown."""
        self.logger.info(f"Agent shutting down\n{stats}")


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Setup global logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
