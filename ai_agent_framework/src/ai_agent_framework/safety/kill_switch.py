"""
Kill switch implementation for AI Agent Framework.

Provides emergency stop functionality.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchState:
    """Kill switch state."""
    is_active: bool = False
    activated_at: Optional[float] = None
    reason: str = ""
    activated_by: str = ""


class KillSwitch:
    """
    Emergency stop mechanism for the agent.
    
    Can be activated manually or automatically based on conditions.
    Once activated, the agent will stop all actions until manually reset.
    """
    
    def __init__(self):
        self._state = KillSwitchState()
        logger.info("KillSwitch initialized")
    
    def activate(self, reason: str = "Manual activation", activated_by: str = "user") -> None:
        """
        Activate the kill switch.
        
        Args:
            reason: Why the kill switch was activated
            activated_by: Who activated it (user, system, etc.)
        """
        if not self._state.is_active:
            self._state.is_active = True
            self._state.activated_at = time.time()
            self._state.reason = reason
            self._state.activated_by = activated_by
            logger.warning(f"KILL SWITCH ACTIVATED by {activated_by}: {reason}")
        else:
            logger.debug("Kill switch already active")
    
    def reset(self, reset_by: str = "user") -> None:
        """
        Reset the kill switch.
        
        Args:
            reset_by: Who reset the kill switch
        """
        if self._state.is_active:
            logger.info(f"Kill switch reset by {reset_by}")
            self._state = KillSwitchState()
        else:
            logger.debug("Kill switch not active")
    
    def is_active(self) -> bool:
        """Check if kill switch is active."""
        return self._state.is_active
    
    def get_state(self) -> KillSwitchState:
        """Get current kill switch state."""
        return self._state
    
    def check_conditions(self, conditions: dict) -> None:
        """
        Check automatic kill switch conditions.
        
        Args:
            conditions: Dictionary of conditions to check
                - 'max_loss': Maximum loss threshold
                - 'max_errors': Maximum error count
                - 'max_latency': Maximum latency threshold
        """
        # This is a placeholder for automatic kill switch logic
        # In a real implementation, you would check various conditions
        # and automatically activate the kill switch if thresholds are exceeded
        pass
    
    def status_string(self) -> str:
        """Get human-readable status."""
        if self._state.is_active:
            duration = time.time() - self._state.activated_at if self._state.activated_at else 0
            return (
                f"KILL SWITCH ACTIVE - {duration:.0f}s ago "
                f"by {self._state.activated_by}: {self._state.reason}"
            )
        return "Kill switch inactive"
