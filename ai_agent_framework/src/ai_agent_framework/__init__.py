"""
AI Agent Framework

A framework for building autonomous reasoning agents with Large Language Models.
"""

__version__ = "1.0.0"
__author__ = "AI Agent Framework Team"

# Import key classes for easy access
from .agent.agent import Agent
from .agent.agent_config import AgentConfig
from .agent.observer import Observer
from .agent.reasoner import Reasoner
from .agent.decision_engine import DecisionEngine

__all__ = [
    "Agent",
    "AgentConfig", 
    "Observer",
    "Reasoner",
    "DecisionEngine"
]
