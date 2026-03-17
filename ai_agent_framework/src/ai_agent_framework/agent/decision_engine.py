"""
Decision validator for AI Agent Framework.

Decision sits between the reasoner and action execution.
It is the final safety check before any parameter changes
are applied to the system.
"""
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .agent_config import AgentConfig
from .observer import AgentContext
from .reasoner import ReasoningResult, StrategyParamUpdate

logger = logging.getLogger(__name__)


class DecisionOutcome(str, Enum):
    APPROVED = 'approved'
    REJECTED = 'rejected'
    HELD = 'held'         # hold — no action
    DRY_RUN = 'dry_run'   # would act but dry
    KILLED = 'killed'     # kill switch active
    PAUSED = 'paused'     # agent paused


@dataclass
class DecisionRecord:
    """
    Complete record of one decision cycle.
    """
    timestamp_ns: int
    outcome: DecisionOutcome

    # What the reasoner recommended
    reasoning_result: ReasoningResult

    # What was actually approved (may differ from reasoner output)
    approved_params: Optional[StrategyParamUpdate] = None

    # Why rejected or overridden
    rejection_reason: Optional[str] = None
    override_notes: List[str] = field(default_factory=list)

    # Context snapshot
    mode: str = 'demo'
    system_healthy: bool = False
    anomalies: List[str] = field(default_factory=list)

    @property
    def was_approved(self) -> bool:
        return self.outcome == DecisionOutcome.APPROVED

    @property
    def action_taken(self) -> bool:
        """
        True if params were actually applied.
        False for dry_run, held, rejected, killed, paused.
        """
        return (
            self.outcome == DecisionOutcome.APPROVED
            and self.approved_params is not None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Format for logging."""
        return {
            'timestamp_ns': self.timestamp_ns,
            'outcome': self.outcome.value,
            'decision_type': 'param_update' if self.action_taken else 'hold',
            'action': self.outcome.value,
            'sources_considered': self.reasoning_result.sources_to_watch,
            'reasoning_summary': self.reasoning_result.reasoning_summary,
            'human_readable': self.reasoning_result.human_readable,
            'llm_provider': (
                self.reasoning_result.llm_response.provider
                if self.reasoning_result.llm_response
                else 'none'
            ),
            'confidence': self.reasoning_result.regime_confidence,
            'risk_check': 'PASSED' if self.was_approved else 'FAILED',
            'market_regime': self.reasoning_result.market_regime,
            'approved_params': (
                self.approved_params.to_dict()
                if self.approved_params
                else None
            ),
            'rejection_reason': self.rejection_reason,
            'override_notes': self.override_notes,
            'mode': self.mode,
            'system_healthy': self.system_healthy,
            'anomalies': self.anomalies,
        }


# Hard limits - cannot be overridden by LLM or config
HARD_MIN_CONFIDENCE = 0.50
HARD_MAX_CONFIDENCE = 0.95
HARD_MIN_STRONG_THRESHOLD = 0.65
HARD_MAX_STRONG_THRESHOLD = 0.95
HARD_MIN_HOLD_THRESHOLD = 0.30
HARD_MAX_HOLD_THRESHOLD = 0.60


class DecisionEngine:
    """
    Final safety gate before parameter changes.
    
    Usage:
        decision_engine = DecisionEngine(config=config)
        record = decision_engine.evaluate(
            result=reasoning_result,
            context=agent_context,
            is_killed=False,
            is_paused=False,
        )
        if record.was_approved:
            # apply the approved_params
    """

    def __init__(self, config: AgentConfig):
        self._config = config
        self._approved_count = 0
        self._rejected_count = 0
        self._held_count = 0
        logger.info("DecisionEngine initialised")

    def evaluate(
        self,
        result: ReasoningResult,
        context: AgentContext,
        is_killed: bool = False,
        is_paused: bool = False,
    ) -> DecisionRecord:
        """
        Evaluate a ReasoningResult and return a DecisionRecord.
        
        Evaluation order:
        1. Kill switch — immediate stop
        2. Pause switch — hold
        3. Dry run — record but don't act
        4. Reasoner failed — hold
        5. System unhealthy in live — reject
        6. Anomalies present — warn
        7. Hard limit enforcement
        8. Approve
        """
        now_ns = time.time_ns()

        base_record = DecisionRecord(
            timestamp_ns=now_ns,
            outcome=DecisionOutcome.HELD,
            reasoning_result=result,
            mode=context.mode,
            system_healthy=context.system_health,
            anomalies=context.anomalies,
        )

        # 1. Kill switch
        if is_killed:
            self._rejected_count += 1
            base_record.outcome = DecisionOutcome.KILLED
            base_record.rejection_reason = 'Kill switch active'
            logger.warning("Decision KILLED — kill switch active")
            return base_record

        # 2. Pause switch
        if is_paused:
            self._held_count += 1
            base_record.outcome = DecisionOutcome.PAUSED
            base_record.rejection_reason = 'Agent paused'
            return base_record

        # 3. Reasoner failed
        if not result.success:
            self._held_count += 1
            base_record.outcome = DecisionOutcome.HELD
            base_record.rejection_reason = f'Reasoner failed: {result.reasoning_summary}'
            return base_record

        # 4. Hold — no action from reasoner
        if result.is_hold:
            self._held_count += 1
            base_record.outcome = DecisionOutcome.HELD
            return base_record

        # 5. System unhealthy in live mode — reject param changes
        if context.mode == 'live' and not context.system_health:
            self._rejected_count += 1
            base_record.outcome = DecisionOutcome.REJECTED
            base_record.rejection_reason = 'System unhealthy — refusing param update in live mode'
            logger.warning("Decision REJECTED — system unhealthy in live mode")
            return base_record

        # 6. Anomaly check - critical anomalies block live mode updates
        critical_anomalies = [
            a for a in context.anomalies
            if 'kill switch' in a.lower() or 'spike' in a.lower()
        ]
        if context.mode == 'live' and critical_anomalies:
            self._rejected_count += 1
            base_record.outcome = DecisionOutcome.REJECTED
            base_record.rejection_reason = f'Critical anomaly: {critical_anomalies[0]}'
            return base_record

        # 7. Enforce hard limits on params
        approved_params, override_notes = self._enforce_hard_limits(result.param_update)

        # 8. Dry run check
        if self._config.dry_run:
            self._held_count += 1
            base_record.outcome = DecisionOutcome.DRY_RUN
            base_record.approved_params = approved_params
            base_record.override_notes = override_notes
            logger.info(f"DRY RUN — would apply: {approved_params.to_dict()}")
            return base_record

        # Approved
        self._approved_count += 1
        base_record.outcome = DecisionOutcome.APPROVED
        base_record.approved_params = approved_params
        base_record.override_notes = override_notes

        logger.info(
            f"Decision APPROVED — regime: {result.market_regime} "
            f"params: {approved_params.to_dict()}"
        )
        return base_record

    def _enforce_hard_limits(
        self,
        params: StrategyParamUpdate,
    ) -> tuple:
        """
        Clamp all parameter values within absolute hard limits.
        
        Returns (clamped_params, notes). notes is empty if no clamping needed.
        """
        notes = []

        def clamp(val: float, lo: float, hi: float, name: str) -> float:
            if val < lo:
                notes.append(f"{name} clamped {val:.3f}→{lo:.3f}")
                return lo
            if val > hi:
                notes.append(f"{name} clamped {val:.3f}→{hi:.3f}")
                return hi
            return val

        min_conf = clamp(
            params.min_confidence,
            HARD_MIN_CONFIDENCE,
            HARD_MAX_CONFIDENCE,
            'min_confidence',
        )
        strong = clamp(
            params.strong_signal_threshold,
            HARD_MIN_STRONG_THRESHOLD,
            HARD_MAX_STRONG_THRESHOLD,
            'strong_signal_threshold',
        )
        hold = clamp(
            params.hold_threshold,
            HARD_MIN_HOLD_THRESHOLD,
            HARD_MAX_HOLD_THRESHOLD,
            'hold_threshold',
        )

        clamped = StrategyParamUpdate(
            min_confidence=min_conf,
            strong_signal_threshold=strong,
            hold_threshold=hold,
            max_order_size=params.max_order_size,
            orders_per_minute=params.orders_per_minute,
        )

        if notes:
            logger.warning(f"Hard limits enforced: {'; '.join(notes)}")

        return clamped, notes

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._approved_count + self._rejected_count + self._held_count
        return {
            'approved': self._approved_count,
            'rejected': self._rejected_count,
            'held': self._held_count,
            'total': total,
            'approval_rate': round(self._approved_count / total * 100, 1) if total > 0 else 0.0,
        }
