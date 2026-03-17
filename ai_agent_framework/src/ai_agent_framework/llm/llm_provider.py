"""
Base LLM provider interface.

All providers must implement BaseLLMProvider.
This allows swapping providers without code changes.
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Structured response from any LLM.
    
    The agent works with LLMResponse objects — never raw API responses.
    This keeps agent logic provider-agnostic.
    """
    # The LLM's reasoning text
    reasoning: str

    # Structured decision — parsed from reasoning by the provider
    decision: Dict[str, Any] = field(default_factory=dict)

    # Provider metadata
    provider: str = ''
    model: str = ''
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    # Whether parsing succeeded
    parsed_successfully: bool = True
    parse_error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def market_regime(self) -> str:
        return self.decision.get('market_regime', 'unknown')

    @property
    def regime_confidence(self) -> float:
        return float(self.decision.get('regime_confidence', 0.0))

    @property
    def strategy_adjustment(self) -> str:
        return self.decision.get('strategy_adjustment', 'hold')

    @property
    def human_readable(self) -> str:
        return self.decision.get(
            'human_readable',
            self.reasoning[:200] if self.reasoning else 'No explanation available'
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'reasoning': self.reasoning,
            'decision': self.decision,
            'provider': self.provider,
            'model': self.model,
            'tokens': self.total_tokens,
            'latency_ms': self.latency_ms,
            'parsed_successfully': self.parsed_successfully,
        }


class LLMError(Exception):
    """Raised when LLM call fails."""
    def __init__(
        self,
        message: str,
        provider: str = '',
        retryable: bool = True,
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class BaseLLMProvider(ABC):
    """
    Abstract base for all LLM providers.
    
    Implementations: ClaudeProvider, OpenAIProvider.
    
    The agent calls reason() once per reasoning cycle with a structured
    context dict. The provider formats the prompt, calls the API,
    parses the response, and returns an LLMResponse.
    """

    # System prompt — same for all providers
    SYSTEM_PROMPT = """You are the reasoning engine for an autonomous AI agent.

Your role is to analyse input data conditions and recommend strategy adjustments.

You must respond with a JSON object containing exactly these fields:
{
  "market_regime": "trending" | "ranging" | "volatile" | "unknown",
  "regime_confidence": <float 0.0-1.0>,
  "strategy_adjustment": "increase_confidence" | "decrease_confidence" | "hold" | "tighten_risk" | "loosen_risk",
  "adjustment_magnitude": <float 0.0-1.0>,
  "symbols_to_watch": [<symbol strings>],
  "reasoning_summary": "<one sentence max 100 chars>",
  "human_readable": "<plain English explanation 1-3 sentences>"
}

Rules:
- Default to "hold" when uncertain
- Default to "tighten_risk" in volatile markets
- Never recommend loosen_risk with confidence below 0.7
- Respond ONLY with the JSON object, no preamble or explanation outside it"""

    def __init__(self, config: Any):
        self.config = config
        self._call_count = 0
        self._error_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    @abstractmethod
    async def reason(self, context: Dict[str, Any]) -> LLMResponse:
        """
        Call LLM with context.
        Returns LLMResponse — never raises.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        True if provider is configured and API key is present.
        Does not make an API call.
        """
        ...

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Build user prompt from context."""
        import json

        # Summarise input data
        input_summary = []
        for key, value in context.get('input_data', {}).items():
            if isinstance(value, (int, float)):
                input_summary.append(f"{key}: {value}")
            elif isinstance(value, str):
                input_summary.append(f"{key}: {value[:50]}...")

        prompt_parts = [
            f"INPUT CONTEXT — {context.get('timestamp', 'now')}",
            f"Mode: {context.get('mode', 'demo')}",
            "",
            "CURRENT INPUT DATA:",
        ]

        if input_summary:
            prompt_parts.extend(input_summary)
        else:
            prompt_parts.append("No input data available")

        regime = context.get('market_regime_current', 'unknown')
        prompt_parts.append(f"\nCurrent regime: {regime}")

        recent = context.get('recent_decisions', [])
        if recent:
            prompt_parts.append("\nRECENT DECISIONS:")
            for d in recent[-3:]:
                prompt_parts.append(f"  - {d}")

        prompt_parts.append("\nAnalyse and respond with JSON only.")

        return '\n'.join(prompt_parts)

    def _parse_response(
        self,
        raw: str,
        provider: str,
        model: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> LLMResponse:
        """Parse raw LLM text into LLMResponse."""
        import json
        import re

        reasoning = raw.strip()

        # Extract JSON from response
        json_str = raw.strip()
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)

        try:
            decision = json.loads(json_str)

            # Validate required fields
            required = ['market_regime', 'strategy_adjustment']
            for field_name in required:
                if field_name not in decision:
                    decision[field_name] = 'unknown' if field_name == 'market_regime' else 'hold'

            # Clamp floats to 0-1
            for float_field in ['regime_confidence', 'adjustment_magnitude']:
                if float_field in decision:
                    decision[float_field] = max(0.0, min(1.0, float(decision[float_field])))

            return LLMResponse(
                reasoning=reasoning,
                decision=decision,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                parsed_successfully=True,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}\nRaw: {raw[:200]}")
            return LLMResponse(
                reasoning=reasoning,
                decision={
                    'market_regime': 'unknown',
                    'regime_confidence': 0.0,
                    'strategy_adjustment': 'hold',
                    'adjustment_magnitude': 0.0,
                    'symbols_to_watch': [],
                    'reasoning_summary': 'Parse error — hold',
                    'human_readable': 'Unable to parse LLM response — holding current strategy',
                },
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                parsed_successfully=False,
                parse_error=str(e),
            )

    @property
    def stats(self) -> Dict[str, Any]:
        avg_latency = (
            self._total_latency_ms / self._call_count
            if self._call_count > 0 else 0.0
        )
        return {
            'calls': self._call_count,
            'errors': self._error_count,
            'total_tokens': self._total_tokens,
            'avg_latency_ms': avg_latency,
        }
