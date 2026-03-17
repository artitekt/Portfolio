"""
Simple in-memory state store for AI Agent Framework.

Provides basic persistence for agent state between cycles.
"""
import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Agent runtime state."""
    loop_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_decision_time: Optional[float] = None
    total_llm_calls: int = 0
    total_tokens_used: int = 0
    recent_decisions: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'loop_count': self.loop_count,
            'start_time': self.start_time,
            'last_decision_time': self.last_decision_time,
            'total_llm_calls': self.total_llm_calls,
            'total_tokens_used': self.total_tokens_used,
            'recent_decisions': self.recent_decisions,
        }


class StateStore:
    """
    Simple in-memory state store with optional file persistence.
    
    Usage:
        store = StateStore()
        state = store.load_state()
        # modify state
        store.save_state(state)
    """
    
    def __init__(self, persist_file: Optional[str] = None):
        self._persist_file = persist_file
        self._state: Optional[AgentState] = None
        
    def load_state(self) -> AgentState:
        """Load state from memory or file."""
        if self._state is not None:
            return self._state
            
        # Try to load from file
        if self._persist_file:
            try:
                with open(self._persist_file, 'rb') as f:
                    self._state = pickle.load(f)
                logger.info(f"State loaded from {self._persist_file}")
                return self._state
            except (FileNotFoundError, pickle.PickleError) as e:
                logger.debug(f"Could not load state file: {e}")
        
        # Create new state
        self._state = AgentState()
        logger.info("Created new agent state")
        return self._state
    
    def save_state(self, state: AgentState) -> None:
        """Save state to memory and optionally to file."""
        self._state = state
        
        if self._persist_file:
            try:
                with open(self._persist_file, 'wb') as f:
                    pickle.dump(state, f)
                logger.debug(f"State saved to {self._persist_file}")
            except (pickle.PickleError, OSError) as e:
                logger.warning(f"Could not save state file: {e}")
    
    def clear_state(self) -> None:
        """Clear stored state."""
        self._state = None
        if self._persist_file:
            try:
                import os
                os.remove(self._persist_file)
                logger.info(f"State file {self._persist_file} removed")
            except OSError:
                pass
