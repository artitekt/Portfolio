"""
OpenAI LLM provider implementation.
"""
import asyncio
import logging
import time
from typing import Any, Dict

from .llm_provider import BaseLLMProvider, LLMResponse, LLMError

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: Any):
        super().__init__(config)
        self._client = None

    def is_available(self) -> bool:
        """True if OpenAI API key is configured."""
        return bool(self.config.openai_api_key)

    async def reason(self, context: Dict[str, Any]) -> LLMResponse:
        """Call OpenAI API with context."""
        if not self.is_available():
            return LLMResponse(
                reasoning='OpenAI provider not configured',
                decision={
                    'market_regime': 'unknown',
                    'regime_confidence': 0.0,
                    'strategy_adjustment': 'hold',
                    'adjustment_magnitude': 0.0,
                    'symbols_to_watch': [],
                    'reasoning_summary': 'Provider not configured',
                    'human_readable': 'OpenAI API key not found',
                },
                provider='openai',
                model=self.config.model,
                parsed_successfully=False,
            )

        try:
            # Import here to avoid dependency if not used
            import openai

            if self._client is None:
                self._client = openai.AsyncOpenAI(api_key=self.config.openai_api_key)

            start_time = time.monotonic()
            
            user_prompt = self._build_user_prompt(context)
            
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )

            latency_ms = (time.monotonic() - start_time) * 1000
            
            content = response.choices[0].message.content or ""
            
            llm_response = self._parse_response(
                raw=content,
                provider='openai',
                model=self.config.model,
                latency_ms=latency_ms,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )

            self._call_count += 1
            self._total_tokens += llm_response.total_tokens
            self._total_latency_ms += latency_ms

            logger.debug(f"OpenAI call completed: {llm_response.total_tokens} tokens, {latency_ms:.0f}ms")
            
            return llm_response

        except Exception as e:
            self._error_count += 1
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                reasoning=f'OpenAI API error: {str(e)}',
                decision={
                    'market_regime': 'unknown',
                    'regime_confidence': 0.0,
                    'strategy_adjustment': 'hold',
                    'adjustment_magnitude': 0.0,
                    'symbols_to_watch': [],
                    'reasoning_summary': 'API error',
                    'human_readable': f'OpenAI API call failed: {str(e)[:100]}',
                },
                provider='openai',
                model=self.config.model,
                parsed_successfully=False,
                parse_error=str(e),
            )
