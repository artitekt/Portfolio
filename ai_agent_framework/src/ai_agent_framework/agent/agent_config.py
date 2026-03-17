"""
AI Agent Framework Configuration.

Simplified configuration for autonomous LLM agent.
Removes trading-specific components and focuses on
generic reasoning cycles.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = 'claude'
    claude_api_key: str = ''
    openai_api_key: str = ''
    model: str = ''
    max_tokens: int = 1000
    timeout: float = 30.0

    # Default models per provider
    DEFAULT_MODELS = {
        'claude': 'claude-sonnet-4-20250514',
        'openai': 'gpt-4o',
    }

    def __post_init__(self):
        if not self.model:
            self.model = self.DEFAULT_MODELS.get(
                self.provider, self.DEFAULT_MODELS['claude']
            )

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        provider = os.getenv('AGENT_LLM_PROVIDER', 'claude').lower()
        model = os.getenv('AGENT_LLM_MODEL', '')
        return cls(
            provider=provider,
            claude_api_key=os.getenv('AGENT_CLAUDE_API_KEY', ''),
            openai_api_key=os.getenv('AGENT_OPENAI_API_KEY', ''),
            model=model,
            max_tokens=int(os.getenv('AGENT_LLM_MAX_TOKENS', '1000')),
            timeout=float(os.getenv('AGENT_LLM_TIMEOUT', '30')),
        )

    @property
    def api_key(self) -> str:
        """Get API key for active provider."""
        if self.provider == 'claude':
            return self.claude_api_key
        elif self.provider == 'openai':
            return self.openai_api_key
        return ''

    @property
    def has_credentials(self) -> bool:
        """True if API key is configured."""
        return bool(self.api_key)


@dataclass
class AgentConfig:
    """Master configuration for AI Agent Framework."""
    mode: str = 'demo'
    loop_interval: float = 30.0
    dry_run: bool = True
    log_level: str = 'INFO'
    log_file: Optional[str] = None

    llm: 'LLMConfig' = field(default_factory=LLMConfig)

    def __post_init__(self):
        # Validate mode
        if self.mode not in ('demo', 'live'):
            self.mode = 'demo'

    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Load config from environment."""
        mode = os.getenv('AGENT_MODE', 'demo').lower()
        return cls(
            mode=mode,
            loop_interval=float(os.getenv('AGENT_LOOP_INTERVAL', '30')),
            dry_run=os.getenv('AGENT_DRY_RUN', 'true').lower() == 'true',
            log_level=os.getenv('AGENT_LOG_LEVEL', 'INFO').upper(),
            log_file=os.getenv('AGENT_LOG_FILE'),
            llm=LLMConfig.from_env(),
        )

    @property
    def is_demo(self) -> bool:
        return self.mode == 'demo'

    @property
    def is_live(self) -> bool:
        return self.mode == 'live'

    def summary(self) -> str:
        """Human-readable config summary."""
        return (
            f"AgentConfig(\n"
            f"  mode={self.mode}\n"
            f"  provider={self.llm.provider}\n"
            f"  model={self.llm.model}\n"
            f"  has_credentials={self.llm.has_credentials}\n"
            f"  loop_interval={self.loop_interval}s\n"
            f"  dry_run={self.dry_run}\n"
            f")"
        )
