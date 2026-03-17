"""
LLM reasoning engine for AI Agent Framework.

Reasoner responsibilities:
  - Receive AgentContext from Observer
  - Call LLM provider with context
  - Validate LLM response against safety limits
  - Translate LLM decision into concrete parameter changes
  - Return structured ReasoningResult
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent_config import AgentConfig
from .observer import AgentContext
from ..llm.llm_provider import BaseLLMProvider, LLMResponse, LLMError

logger = logging.getLogger(__name__)


# Valid values for each LLM output field
VALID_REGIMES = {'trending', 'ranging', 'volatile', 'unknown'}
VALID_ADJUSTMENTS = {
    'increase_confidence', 'decrease_confidence', 'hold',
    'tighten_risk', 'loosen_risk'
}


@dataclass
class StrategyParamUpdate:
    """
    Concrete strategy parameter changes derived from LLM decision.
    """
    min_confidence: float
    strong_signal_threshold: float
    hold_threshold: float
    # None means no change
    max_order_size: Optional[int] = None
    orders_per_minute: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'min_confidence': self.min_confidence,
            'strong_signal_threshold': self.strong_signal_threshold,
            'hold_threshold': self.hold_threshold,
        }
        if self.max_order_size is not None:
            d['max_order_size'] = self.max_order_size
        if self.orders_per_minute is not None:
            d['orders_per_minute'] = self.orders_per_minute
        return d


@dataclass
class ReasoningResult:
    """
    Complete output of one reasoning cycle.
    
    Produced by Reasoner.reason(). Consumed by decision.py for final validation.
    """
    # Was the LLM call successful?
    success: bool

    # The raw LLM response
    llm_response: Optional[LLMResponse]

    # Concrete parameter update (None if reasoning failed or adjustment is 'hold')
    param_update: Optional[StrategyParamUpdate] = None

    # Market regime assessment
    market_regime: str = 'unknown'
    regime_confidence: float = 0.0

    # Sources LLM flagged for attention
    sources_to_watch: List[str] = field(default_factory=list)

    # One-line summary for decision log
    reasoning_summary: str = ''

    # Human-readable explanation
    human_readable: str = ''

    # Validation notes — why any LLM suggestions were overridden
    validation_notes: List[str] = field(default_factory=list)

    # Performance
    latency_ms: float = 0.0
    tokens_used: int = 0

    # Was the LLM response well-formed?
    llm_parsed: bool = False

    @property
    def has_param_update(self) -> bool:
        return self.param_update is not None

    @property
    def is_hold(self) -> bool:
        """True if no action recommended."""
        return not self.has_param_update

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'market_regime': self.market_regime,
            'regime_confidence': self.regime_confidence,
            'has_param_update': self.has_param_update,
            'param_update': self.param_update.to_dict() if self.param_update else None,
            'sources_to_watch': self.sources_to_watch,
            'reasoning_summary': self.reasoning_summary,
            'human_readable': self.human_readable,
            'validation_notes': self.validation_notes,
            'latency_ms': self.latency_ms,
            'tokens_used': self.tokens_used,
            'llm_parsed': self.llm_parsed,
        }


class Reasoner:
    """
    LLM reasoning engine.
    
    Usage:
        reasoner = Reasoner(provider=provider, config=config)
        result = await reasoner.reason(context=context)
    """

    # Token budget per day (prevents runaway costs)
    DAILY_TOKEN_BUDGET = 50_000

    # Minimum seconds between LLM calls (prevents thrashing)
    MIN_CALL_INTERVAL = 25.0

    def __init__(self, provider: BaseLLMProvider, config: AgentConfig):
        self._provider = provider
        self._config = config

        # Rate limiting
        self._last_call_time: float = 0.0
        self._call_count: int = 0
        self._tokens_today: int = 0
        self._day_start: float = time.time()

        # Stats
        self._success_count: int = 0
        self._error_count: int = 0
        self._hold_count: int = 0

        logger.info(f"Reasoner initialised — provider: {provider.__class__.__name__}")

    async def reason(self, context: AgentContext) -> ReasoningResult:
        """
        Run one reasoning cycle.
        
        Returns ReasoningResult — never raises. On any failure returns a hold result.
        """
        start = time.monotonic()

        # Check rate limits first
        rate_check = self._check_rate_limits()
        if rate_check:
            self._hold_count += 1
            return ReasoningResult(
                success=False,
                llm_response=None,
                reasoning_summary=rate_check,
                human_readable=f"Rate limit: {rate_check}",
            )

        # Check system health
        if not context.system_health and self._config.mode == 'live':
            self._hold_count += 1
            return ReasoningResult(
                success=True,
                llm_response=None,
                reasoning_summary='System unhealthy — hold',
                human_readable='System not healthy — holding strategy until recovery',
            )

        # Check provider availability
        if not self._provider.is_available():
            self._hold_count += 1
            return ReasoningResult(
                success=False,
                llm_response=None,
                reasoning_summary='LLM unavailable — hold',
                human_readable='LLM provider not available. Check API key configuration.',
            )

        # Call LLM
        try:
            llm_context = context.to_llm_dict()
            response = await self._provider.reason(llm_context)

            self._last_call_time = time.time()
            self._call_count += 1
            self._tokens_today += response.total_tokens
            self._reset_budget_if_new_day()

            latency_ms = (time.monotonic() - start) * 1000

            # Validate and translate
            result = self._process_response(
                response=response,
                context=context,
                latency_ms=latency_ms,
            )

            if result.success:
                self._success_count += 1
            else:
                self._error_count += 1

            return result

        except Exception as e:
            self._error_count += 1
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(f"Reasoner error: {e}")
            return ReasoningResult(
                success=False,
                llm_response=None,
                reasoning_summary=f"Error: {str(e)[:80]}",
                human_readable=f"Reasoning error: {str(e)[:100]} — holding strategy",
                latency_ms=latency_ms,
            )

    def _process_response(
        self,
        response: LLMResponse,
        context: AgentContext,
        latency_ms: float,
    ) -> ReasoningResult:
        """
        Validate LLM response and translate into concrete parameter update.
        """
        notes = []

        # Validate regime
        regime = response.market_regime
        if regime not in VALID_REGIMES:
            notes.append(f"Invalid regime '{regime}' → 'unknown'")
            regime = 'unknown'

        regime_conf = response.regime_confidence

        # Validate adjustment
        adjustment = response.strategy_adjustment
        if adjustment not in VALID_ADJUSTMENTS:
            notes.append(f"Invalid adjustment '{adjustment}' → 'hold'")
            adjustment = 'hold'

        magnitude = float(response.decision.get('adjustment_magnitude', 0.0))
        magnitude = max(0.0, min(1.0, magnitude))

        # Safety: loosen_risk requires high confidence
        if adjustment == 'loosen_risk' and regime_conf < 0.7:
            notes.append(f"loosen_risk blocked: regime_confidence={regime_conf:.2f} < 0.7 → 'hold'")
            adjustment = 'hold'

        # Safety: tighten in volatile markets regardless of LLM
        if regime == 'volatile' and adjustment == 'loosen_risk':
            notes.append("loosen_risk blocked in volatile regime → 'tighten_risk'")
            adjustment = 'tighten_risk'

        # Validate sources_to_watch
        sources = response.decision.get('symbols_to_watch', [])
        if not isinstance(sources, list):
            sources = []
        # Keep only valid source names
        sources = [s for s in sources if isinstance(s, str) and len(s) <= 20]

        # Translate adjustment to params
        param_update = None
        if adjustment != 'hold':
            param_update = self._build_param_update(
                adjustment=adjustment,
                magnitude=magnitude,
                regime=regime,
            )

        # Summary and explanation
        summary = response.decision.get('reasoning_summary', '')
        if not summary:
            summary = f"{regime} market — {adjustment}"
        summary = summary[:100]  # Truncate to 100 chars

        human = response.human_readable
        if not human:
            human = f"Market regime: {regime} (confidence: {regime_conf:.0%}). Strategy: {adjustment}."

        if notes:
            logger.info(f"Validation notes: {'; '.join(notes)}")

        return ReasoningResult(
            success=True,
            llm_response=response,
            param_update=param_update,
            market_regime=regime,
            regime_confidence=regime_conf,
            sources_to_watch=sources,
            reasoning_summary=summary,
            human_readable=human,
            validation_notes=notes,
            latency_ms=latency_ms,
            tokens_used=response.total_tokens,
            llm_parsed=response.parsed_successfully,
        )

    def _build_param_update(
        self,
        adjustment: str,
        magnitude: float,
        regime: str,
    ) -> StrategyParamUpdate:
        """
        Translate adjustment + magnitude into concrete param values.
        """
        # Default baseline values
        base_conf = 0.65
        base_strong = 0.80
        base_hold = 0.45

        # Scale factor from magnitude
        max_shift = 0.15
        shift = magnitude * max_shift

        if adjustment == 'increase_confidence':
            # Raise the bar for signals
            new_conf = min(base_conf + shift, 0.90)
            new_strong = min(base_strong + shift * 0.5, 0.95)
            new_hold = base_hold

        elif adjustment == 'decrease_confidence':
            # Lower the bar — more signals
            new_conf = max(base_conf - shift, 0.50)
            new_strong = max(base_strong - shift * 0.5, 0.65)
            new_hold = base_hold

        elif adjustment == 'tighten_risk':
            # Raise confidence AND reduce order sizes
            new_conf = min(base_conf + shift, 0.90)
            new_strong = min(base_strong + shift, 0.95)
            new_hold = min(base_hold + shift * 0.5, 0.60)

        elif adjustment == 'loosen_risk':
            # Lower confidence AND allow larger orders
            new_conf = max(base_conf - shift * 0.5, 0.55)
            new_strong = max(base_strong - shift * 0.5, 0.70)
            new_hold = base_hold

        else:
            # Should not reach here (hold filtered out above)
            new_conf = base_conf
            new_strong = base_strong
            new_hold = base_hold

        # Clamp all values
        new_conf = round(max(0.50, min(0.95, new_conf)), 3)
        new_strong = round(max(0.65, min(0.95, new_strong)), 3)
        new_hold = round(max(0.30, min(0.60, new_hold)), 3)

        return StrategyParamUpdate(
            min_confidence=new_conf,
            strong_signal_threshold=new_strong,
            hold_threshold=new_hold,
        )

    def _check_rate_limits(self) -> Optional[str]:
        """Check rate limits before LLM call."""
        now = time.time()

        # Minimum interval between calls
        elapsed = now - self._last_call_time
        if self._last_call_time > 0 and elapsed < self.MIN_CALL_INTERVAL:
            remaining = self.MIN_CALL_INTERVAL - elapsed
            return f"Rate limit: wait {remaining:.0f}s"

        # Daily token budget
        self._reset_budget_if_new_day()
        if self._tokens_today >= self.DAILY_TOKEN_BUDGET:
            return f"Daily token budget exhausted: {self._tokens_today}/{self.DAILY_TOKEN_BUDGET}"

        return None

    def _reset_budget_if_new_day(self) -> None:
        """Reset token count at midnight."""
        now = time.time()
        seconds_per_day = 86400
        if now - self._day_start >= seconds_per_day:
            self._tokens_today = 0
            self._day_start = now
            logger.info("Daily token budget reset")

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._success_count + self._error_count + self._hold_count
        return {
            'calls': self._call_count,
            'successes': self._success_count,
            'errors': self._error_count,
            'holds': self._hold_count,
            'tokens_today': self._tokens_today,
            'token_budget': self.DAILY_TOKEN_BUDGET,
            'budget_used_pct': round(self._tokens_today / self.DAILY_TOKEN_BUDGET * 100, 1),
        }
