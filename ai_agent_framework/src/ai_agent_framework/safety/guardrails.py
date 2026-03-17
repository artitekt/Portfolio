"""
Safety guardrails for AI Agent Framework.

Provides additional safety checks and constraints.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation."""
    rule_name: str
    severity: str  # 'warning', 'error', 'critical'
    message: str
    timestamp: float
    context: Dict[str, Any]


class Guardrails:
    """
    Safety guardrails for agent operations.
    
    Provides additional validation and constraints beyond
    the basic decision engine.
    """
    
    def __init__(self):
        self._violations: List[GuardrailViolation] = []
        self._rules = {
            'max_llm_calls_per_hour': 100,
            'max_tokens_per_hour': 10000,
            'max_decision_rate': 10,  # decisions per minute
            'required_confidence_threshold': 0.5,
            'forbidden_regimes': {'critical'},
        }
        self._call_times: List[float] = []
        self._decision_times: List[float] = []
        self._hourly_tokens = 0
        self._hourly_calls = 0
        self._last_hour_reset = time.time()
        
        logger.info("Guardrails initialized")
    
    def check_llm_call(self, tokens_used: int) -> List[GuardrailViolation]:
        """
        Check if an LLM call is allowed.
        
        Args:
            tokens_used: Number of tokens used in this call
            
        Returns:
            List of violations (empty if call is allowed)
        """
        violations = []
        now = time.time()
        
        # Reset hourly counters
        if now - self._last_hour_reset > 3600:
            self._hourly_calls = 0
            self._hourly_tokens = 0
            self._last_hour_reset = now
        
        # Check call rate
        self._call_times.append(now)
        # Keep only last hour
        self._call_times = [t for t in self._call_times if now - t < 3600]
        
        if len(self._call_times) > self._rules['max_llm_calls_per_hour']:
            violations.append(GuardrailViolation(
                rule_name='max_llm_calls_per_hour',
                severity='error',
                message=f"Too many LLM calls: {len(self._call_times)} > {self._rules['max_llm_calls_per_hour']}",
                timestamp=now,
                context={'calls': len(self._call_times), 'limit': self._rules['max_llm_calls_per_hour']}
            ))
        
        # Check token usage
        self._hourly_tokens += tokens_used
        if self._hourly_tokens > self._rules['max_tokens_per_hour']:
            violations.append(GuardrailViolation(
                rule_name='max_tokens_per_hour',
                severity='error',
                message=f"Too many tokens: {self._hourly_tokens} > {self._rules['max_tokens_per_hour']}",
                timestamp=now,
                context={'tokens': self._hourly_tokens, 'limit': self._rules['max_tokens_per_hour']}
            ))
        
        # Record violations
        for violation in violations:
            self._violations.append(violation)
            logger.warning(f"Guardrail violation: {violation.message}")
        
        return violations
    
    def check_decision(
        self,
        confidence: float,
        regime: str,
        decision_type: str
    ) -> List[GuardrailViolation]:
        """
        Check if a decision is allowed.
        
        Args:
            confidence: Decision confidence level
            regime: Market regime
            decision_type: Type of decision being made
            
        Returns:
            List of violations (empty if decision is allowed)
        """
        violations = []
        now = time.time()
        
        # Check decision rate
        self._decision_times.append(now)
        # Keep only last minute
        self._decision_times = [t for t in self._decision_times if now - t < 60]
        
        if len(self._decision_times) > self._rules['max_decision_rate']:
            violations.append(GuardrailViolation(
                rule_name='max_decision_rate',
                severity='warning',
                message=f"Too many decisions: {len(self._decision_times)}/min > {self._rules['max_decision_rate']}/min",
                timestamp=now,
                context={'rate': len(self._decision_times), 'limit': self._rules['max_decision_rate']}
            ))
        
        # Check confidence threshold
        if confidence < self._rules['required_confidence_threshold']:
            violations.append(GuardrailViolation(
                rule_name='required_confidence_threshold',
                severity='warning',
                message=f"Low confidence: {confidence:.2f} < {self._rules['required_confidence_threshold']}",
                timestamp=now,
                context={'confidence': confidence, 'threshold': self._rules['required_confidence_threshold']}
            ))
        
        # Check forbidden regimes
        if regime in self._rules['forbidden_regimes']:
            violations.append(GuardrailViolation(
                rule_name='forbidden_regimes',
                severity='critical',
                message=f"Forbidden regime: {regime}",
                timestamp=now,
                context={'regime': regime}
            ))
        
        # Record violations
        for violation in violations:
            self._violations.append(violation)
            logger.warning(f"Guardrail violation: {violation.message}")
        
        return violations
    
    def get_recent_violations(self, minutes: int = 60) -> List[GuardrailViolation]:
        """Get violations from the last N minutes."""
        now = time.time()
        return [v for v in self._violations if now - v.timestamp < minutes * 60]
    
    def clear_violations(self) -> None:
        """Clear all violation records."""
        self._violations.clear()
        logger.info("Guardrail violations cleared")
    
    def update_rule(self, rule_name: str, value: Any) -> None:
        """Update a guardrail rule."""
        if rule_name in self._rules:
            old_value = self._rules[rule_name]
            self._rules[rule_name] = value
            logger.info(f"Guardrail rule updated: {rule_name} {old_value} -> {value}")
        else:
            logger.warning(f"Unknown guardrail rule: {rule_name}")
    
    def get_rules(self) -> Dict[str, Any]:
        """Get current guardrail rules."""
        return self._rules.copy()
    
    def status_summary(self) -> str:
        """Get human-readable status summary."""
        recent_violations = self.get_recent_violations(60)
        critical_count = sum(1 for v in recent_violations if v.severity == 'critical')
        error_count = sum(1 for v in recent_violations if v.severity == 'error')
        warning_count = sum(1 for v in recent_violations if v.severity == 'warning')
        
        return (
            f"Guardrails Status (last hour): "
            f"{critical_count} critical, {error_count} errors, {warning_count} warnings"
        )
