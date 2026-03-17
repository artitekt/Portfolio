"""
Generic observer for AI Agent Framework.

Observer reads input data and produces a structured context
for the LLM reasoner.
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InputSignal:
    """Generic input signal from data sources."""
    source: str
    value: float
    confidence: float
    timestamp_ns: int
    age_seconds: float

    @property
    def is_fresh(self, max_age: float = 60.0) -> bool:
        """Signal is fresh if < max_age."""
        return self.age_seconds < max_age

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'value': self.value,
            'confidence': self.confidence,
            'age_seconds': self.age_seconds,
            'is_fresh': self.is_fresh,
        }


@dataclass
class AgentContext:
    """
    Structured context for the LLM reasoner.
    
    Built by Observer.observe() on each agent reasoning cycle.
    """
    # When this context was built
    timestamp: str
    timestamp_ns: int

    # Agent operating mode
    mode: str  # demo|live

    # Input signals from various sources
    signals: Dict[str, InputSignal] = field(default_factory=dict)

    # Current system state
    system_health: bool = True
    daily_pnl: Optional[float] = None
    equity: Optional[float] = None

    # Market regime from last LLM cycle
    market_regime_current: str = 'unknown'

    # Last 3 reasoning summaries for LLM context continuity
    recent_decisions: List[str] = field(default_factory=list)

    # Anomalies detected this cycle
    anomalies: List[str] = field(default_factory=list)

    @property
    def actionable_signals(self) -> Dict[str, InputSignal]:
        return {
            source: sig for source, sig in self.signals.items()
            if sig.is_fresh and sig.confidence >= 0.6
        }

    def to_llm_dict(self) -> Dict[str, Any]:
        """Format for LLM provider context."""
        return {
            'timestamp': self.timestamp,
            'mode': self.mode,
            'system_health': self.system_health,
            'input_data': {
                source: sig.to_dict() for source, sig in self.signals.items()
            },
            'daily_pnl': self.daily_pnl,
            'equity': self.equity,
            'market_regime_current': self.market_regime_current,
            'recent_decisions': self.recent_decisions,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full context for logging."""
        d = self.to_llm_dict()
        d.update({
            'timestamp_ns': self.timestamp_ns,
            'anomalies': self.anomalies,
            'actionable_signal_count': len(self.actionable_signals),
        })
        return d


class Observer:
    """
    Reads input data and builds AgentContext for reasoner.
    
    Usage:
        observer = Observer(config)
        context = observer.observe()
        # Pass context to reasoner
    """

    # Max signals to track per source for trend detection
    SIGNAL_HISTORY_SIZE = 20

    def __init__(self, config: Any):
        self._config = config

        # Signal history per source
        self._signal_history: Dict[str, deque] = {}

        # Recent decision summaries
        self._recent_decisions: deque = deque(maxlen=3)

        # Observation count
        self._obs_count = 0
        self._anomaly_count = 0

        logger.info("Observer initialised")

    def observe(self) -> AgentContext:
        """
        Read current system state and return structured AgentContext.
        
        For demo purposes, generates mock data.
        """
        self._obs_count += 1
        now_ns = time.time_ns()
        now_str = datetime.now(timezone.utc).isoformat()

        context = AgentContext(
            timestamp=now_str,
            timestamp_ns=now_ns,
            mode=self._config.mode,
            recent_decisions=list(self._recent_decisions),
        )

        try:
            # Generate mock input signals for demo
            context.signals = self._generate_mock_signals(now_ns)
            
            # Mock system health
            context.system_health = True
            
            # Detect anomalies
            context.anomalies = self._detect_anomalies(context)

            if context.anomalies:
                self._anomaly_count += len(context.anomalies)
                for anomaly in context.anomalies:
                    logger.warning(f"Anomaly: {anomaly}")

            logger.debug(
                f"Observe #{self._obs_count}: "
                f"{len(context.signals)} signals, "
                f"system_healthy={context.system_health}"
            )

        except Exception as e:
            logger.error(f"observe() error: {e}")
            context.anomalies.append(f"Observer error: {str(e)}")

        return context

    def _generate_mock_signals(self, now_ns: int) -> Dict[str, InputSignal]:
        """Generate mock input signals for demonstration."""
        import random

        signals = {}
        
        # Mock signal sources
        sources = ['sensor_a', 'sensor_b', 'data_feed', 'api_endpoint']
        
        for source in sources:
            # Generate random value and confidence
            value = random.uniform(-10, 10)
            confidence = random.uniform(0.3, 0.9)
            
            signal = InputSignal(
                source=source,
                value=value,
                confidence=confidence,
                timestamp_ns=now_ns,
                age_seconds=random.uniform(0, 120),
            )
            signals[source] = signal

            # Update history
            if source not in self._signal_history:
                self._signal_history[source] = deque(maxlen=self.SIGNAL_HISTORY_SIZE)
            self._signal_history[source].append((now_ns, value, confidence))

        return signals

    def _detect_anomalies(self, context: AgentContext) -> List[str]:
        """Detect anomalies in current state."""
        anomalies = []

        # System not healthy
        if not context.system_health:
            anomalies.append("System health check failed")

        # Stale signals
        stale_count = sum(
            1 for sig in context.signals.values()
            if not sig.is_fresh
        )
        if stale_count > 0:
            anomalies.append(f"{stale_count} stale signals (>60s old)")

        return anomalies

    def get_signal_trend(self, source: str, window: int = 5) -> Optional[float]:
        """
        Calculate signal trend for source.
        Returns average value over last N signals, or None if insufficient history.
        """
        history = self._signal_history.get(source)
        if not history or len(history) < window:
            return None
        recent = list(history)[-window:]
        values = [v for _, v, _ in recent]
        return sum(values) / len(values)

    def add_decision_summary(self, summary: str) -> None:
        """
        Add a decision summary to the recent decisions deque.
        Called by agent after each cycle.
        """
        self._recent_decisions.append(summary)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            'observations': self._obs_count,
            'anomalies_total': self._anomaly_count,
            'sources_tracked': len(self._signal_history),
        }
