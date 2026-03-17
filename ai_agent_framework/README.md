# Autonomous LLM Agent Framework

A simplified, standalone framework for building autonomous reasoning agents with Large Language Models.

## Architecture

The framework implements a clean **Observe → Reason → Decide → Act** loop:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Observer  │───▶│  Reasoner   │───▶│  Decision   │───▶│   Action    │
│             │    │             │    │   Engine    │    │   Apply     │
│ - Input     │    │ - LLM Call  │    │ - Safety    │    │ - Params    │
│   Data      │    │ - Context   │    │   Checks    │    │   Update    │
│ - Validation│    │ - Validation│    │ - Hard      │    │ - Logging   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ▲                   ▲                   ▲                   ▲
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                    Sleep & Repeat (configurable interval)
```

## Features

- **Pluggable LLM Providers**: Support for OpenAI and Anthropic Claude
- **Safety First**: Multiple layers of validation and hard limits
- **Async Architecture**: Non-blocking operations with configurable rate limits
- **Generic Input Handling**: Works with any input data (not just trading)
- **Comprehensive Logging**: Detailed audit trails and performance metrics
- **Kill Switch**: Emergency stop functionality
- **Guardrails**: Configurable safety constraints

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set your LLM provider credentials as environment variables:

```bash
# For Anthropic Claude
export AGENT_CLAUDE_API_KEY="your-anthropic-api-key"

# For OpenAI
export AGENT_OPENAI_API_KEY="your-openai-api-key"

# Optional: Choose provider and model
export AGENT_LLM_PROVIDER="claude"  # or "openai"
export AGENT_LLM_MODEL="claude-sonnet-4-20250514"
```

### Run Example

```bash
# Single cycle demo
python examples/run_agent.py --mode single

# Continuous demo (3 cycles)
python examples/run_agent.py --mode continuous
```

## Core Components

### Agent
The main orchestrator that runs the continuous reasoning loop.

```python
from src.agent.agent import Agent
from src.agent.agent_config import AgentConfig
import asyncio

config = AgentConfig.from_env()
agent = Agent.build(config)
asyncio.run(agent.run())
```

### Observer
Collects and validates input data from various sources.

```python
from src.agent.observer import Observer

observer = Observer(config)
context = observer.observe()
```

### Reasoner
Calls the LLM with structured context and validates responses.

```python
from src.agent.reasoner import Reasoner
from src.llm.anthropic_provider import AnthropicProvider

provider = AnthropicProvider(config.llm)
reasoner = Reasoner(provider, config)
result = await reasoner.reason(context)
```

### Decision Engine
Final safety gate before parameter changes are applied.

```python
from src.agent.decision_engine import DecisionEngine

decision_engine = DecisionEngine(config)
record = decision_engine.evaluate(result, context)
```

## Configuration

The framework uses environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODE` | `demo` | Operating mode: `demo` or `live` |
| `AGENT_LLM_PROVIDER` | `claude` | LLM provider: `claude` or `openai` |
| `AGENT_LLM_MODEL` | - | LLM model name |
| `AGENT_LOOP_INTERVAL` | `30` | Seconds between reasoning cycles |
| `AGENT_DRY_RUN` | `true` | If true, observes but doesn't apply changes |
| `AGENT_LOG_LEVEL` | `INFO` | Logging level |

## Safety Features

### Hard Limits
- Confidence thresholds cannot be bypassed
- Parameter ranges are strictly enforced
- Automatic rejection in volatile conditions

### Kill Switch
```python
from src.safety.kill_switch import KillSwitch

kill_switch = KillSwitch()
kill_switch.activate("Manual emergency stop")
```

### Guardrails
```python
from src.safety.guardrails import Guardrails

guardrails = Guardrails()
violations = guardrails.check_decision(confidence, regime, decision_type)
```

## LLM Response Format

The LLM must respond with JSON containing:

```json
{
  "market_regime": "trending" | "ranging" | "volatile" | "unknown",
  "regime_confidence": 0.0-1.0,
  "strategy_adjustment": "increase_confidence" | "decrease_confidence" | "hold" | "tighten_risk" | "loosen_risk",
  "adjustment_magnitude": 0.0-1.0,
  "symbols_to_watch": ["source1", "source2"],
  "reasoning_summary": "One sentence summary",
  "human_readable": "Human explanation"
}
```

## Development

### Project Structure
```
src/
├── agent/          # Core agent logic
│   ├── agent.py
│   ├── agent_config.py
│   ├── observer.py
│   ├── reasoner.py
│   └── decision_engine.py
├── llm/            # LLM providers
│   ├── llm_provider.py
│   ├── openai_provider.py
│   └── anthropic_provider.py
├── memory/         # State management
│   └── state_store.py
├── safety/         # Safety mechanisms
│   ├── kill_switch.py
│   └── guardrails.py
└── utils/          # Utilities
    ├── logger.py
    └── async_helpers.py
```

### Adding New LLM Providers

1. Inherit from `BaseLLMProvider`
2. Implement `reason()` and `is_available()` methods
3. Update provider selection logic

### Testing

```bash
pytest tests/
```

## License

This project is part of a portfolio demonstration. See LICENSE file for details.

## Contributing

This is a simplified portfolio project. For production use, consider adding:
- Persistent state storage
- More sophisticated input validation
- Additional LLM providers
- Performance monitoring
- Unit test coverage
