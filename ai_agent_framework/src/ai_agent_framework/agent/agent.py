"""
AI Agent Framework — Main agent loop.

Orchestrates the complete reasoning cycle:
  1. Observe  — read input data from sources
  2. Reason   — call LLM with context
  3. Decide   — validate against hard limits
  4. Apply    — update parameters
  5. Log      — record decision
  6. Sleep    — wait for next cycle

Usage:
    from ai_agent_framework.agent.agent import Agent
    from ai_agent_framework.agent.agent_config import AgentConfig
    import asyncio

    config = AgentConfig.from_env()
    agent = Agent(config)
    asyncio.run(agent.run())
"""
import asyncio
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from .agent_config import AgentConfig
from .decision_engine import DecisionEngine, DecisionRecord
from .observer import Observer, AgentContext
from .reasoner import Reasoner

logger = logging.getLogger(__name__)


def configure_logging(level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Configure root logging."""
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )


class Agent:
    """
    Autonomous AI reasoning agent.
    
    The agent runs a continuous async loop. Each cycle:
      1. Reads input data from sources
      2. Calls LLM for reasoning
      3. Validates decision
      4. Applies approved parameters
      5. Logs the decision
      6. Prints status periodically
    """

    # Print status every N cycles
    STATUS_EVERY_N_CYCLES = 5

    def __init__(
        self,
        config: AgentConfig,
        observer: Optional[Observer] = None,
        reasoner: Optional[Reasoner] = None,
        decision_engine: Optional[DecisionEngine] = None,
    ):
        self._config = config
        self._observer = observer
        self._reasoner = reasoner
        self._decision_engine = decision_engine

        # Runtime state
        self._loop_count = 0
        self._running = False
        self._shutdown_requested = False
        self._recent_records: List[DecisionRecord] = []

        # Stats
        self._start_time: float = 0.0
        self._cycle_times: List[float] = []

    @classmethod
    def build(cls, config: AgentConfig) -> 'Agent':
        """
        Build agent with all components.
        Returns Agent ready to run.
        """
        configure_logging(config.log_level, config.log_file)

        logger.info(f"Building AI Agent Framework\n{config.summary()}")

        # Create mock LLM provider for demo
        from ..llm.openai_provider import OpenAIProvider
        from ..llm.anthropic_provider import AnthropicProvider

        # Choose provider based on config
        if config.llm.provider == 'openai':
            provider = OpenAIProvider(config.llm)
        else:
            provider = AnthropicProvider(config.llm)

        # Wire all components
        observer = Observer(config)
        reasoner = Reasoner(provider, config)
        decision_engine = DecisionEngine(config)

        agent = cls(
            config=config,
            observer=observer,
            reasoner=reasoner,
            decision_engine=decision_engine,
        )

        logger.info("Agent built successfully")
        return agent

    async def run(self) -> None:
        """
        Run the agent loop until stopped.
        
        Registers signal handlers for graceful shutdown.
        Prints startup banner.
        Runs _cycle() on each iteration.
        Sleeps config.loop_interval between cycles.
        """
        self._running = True
        self._start_time = time.time()

        # Register signal handlers
        self._register_signals()

        # Startup banner
        self._print_banner()

        logger.info(f"Agent loop starting — interval: {self._config.loop_interval}s")

        try:
            while self._running and not self._shutdown_requested:
                cycle_start = time.monotonic()

                await self._cycle()

                cycle_time = time.monotonic() - cycle_start
                self._cycle_times.append(cycle_time)
                if len(self._cycle_times) > 100:
                    self._cycle_times.pop(0)

                # Sleep for remaining interval
                elapsed = cycle_time
                sleep_time = max(0.0, self._config.loop_interval - elapsed)

                if self._running and not self._shutdown_requested:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Agent loop cancelled")
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            raise
        finally:
            await self._shutdown()

    async def _cycle(self) -> None:
        """
        Execute one complete reasoning cycle.
        
        1. Observe input data
        2. Reason with LLM
        3. Decide
        4. Apply parameters
        5. Log decision
        6. Update observer with summary
        7. Print status periodically
        """
        self._loop_count += 1
        logger.debug(f"Cycle #{self._loop_count} start")

        # 1. Observe
        context: AgentContext = self._observer.observe()

        # 2. Reason
        result = await self._reasoner.reason(context)

        # 3. Decide
        record = self._decision_engine.evaluate(
            result=result,
            context=context,
            is_killed=False,  # TODO: implement kill switch
            is_paused=False,  # TODO: implement pause
        )

        # 4. Apply parameters (mock implementation)
        if record.was_approved:
            await self._apply_parameters(record.approved_params)

        # 5. Log decision
        self._log_decision(record)

        # Keep recent records for status
        self._recent_records.append(record)
        if len(self._recent_records) > 10:
            self._recent_records.pop(0)

        # 6. Update observer with summary
        if result.reasoning_summary:
            self._observer.add_decision_summary(result.reasoning_summary)

        # 7. Periodic status print
        if self._loop_count % self.STATUS_EVERY_N_CYCLES == 0:
            self._print_status(context)

        # Log cycle outcome
        logger.debug(
            f"Cycle #{self._loop_count} complete — "
            f"outcome={record.outcome.value} "
            f"applied_params={record.action_taken}"
        )

    async def _apply_parameters(self, params) -> None:
        """Apply approved parameters (mock implementation)."""
        logger.info(f"Applying parameters: {params.to_dict()}")
        # In a real implementation, this would update the target system

    def _log_decision(self, record: DecisionRecord) -> None:
        """Log decision record."""
        logger.info(f"Decision: {record.outcome.value} — {record.reasoning_result.reasoning_summary}")

    def _print_banner(self) -> None:
        """Print startup banner."""
        mode_str = '📄 DEMO MODE' if self._config.is_demo else '🔴 LIVE MODE'
        dry_str = ' [DRY RUN]' if self._config.dry_run else ''
        print(
            f"\n{'═' * 50}\n"
            f"  AI Agent Framework v1.0.0\n"
            f"  {mode_str}{dry_str}\n"
            f"  Provider: {self._config.llm.provider} / {self._config.llm.model}\n"
            f"  Interval: {self._config.loop_interval}s\n"
            f"{'═' * 50}\n"
        )

    def _print_status(self, context: AgentContext) -> None:
        """Print periodic status report."""
        lines = [
            f"Agent Status - Cycle #{self._loop_count}",
            f"  Mode: {context.mode}",
            f"  System healthy: {context.system_health}",
            f"  Active signals: {len(context.actionable_signals)}",
            f"  Recent decisions: {len(context.recent_decisions)}",
        ]
        
        if self._reasoner:
            rs = self._reasoner.stats
            lines.extend([
                f"  LLM calls: {rs['calls']}",
                f"  Tokens today: {rs['tokens_today']}",
                f"  Budget used: {rs['budget_used_pct']}%",
            ])
        
        if self._decision_engine:
            ds = self._decision_engine.stats
            lines.extend([
                f"  Approved: {ds['approved']}",
                f"  Held: {ds['held']}",
                f"  Rejected: {ds['rejected']}",
            ])
        
        lines.append('─' * 40)
        print('\n'.join(lines))

    def _register_signals(self) -> None:
        """Register OS signal handlers."""
        def handle_shutdown(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.info(f"Signal {sig_name} received — shutting down after current cycle")
            self._shutdown_requested = True

        def handle_status(signum: int, frame: Any) -> None:
            logger.info("SIGUSR1 — printing stats")
            print(self.stats_report())

        try:
            signal.signal(signal.SIGTERM, handle_shutdown)
            signal.signal(signal.SIGINT, handle_shutdown)
            # SIGUSR1 not available on Windows
            if hasattr(signal, 'SIGUSR1'):
                signal.signal(signal.SIGUSR1, handle_status)
        except (OSError, ValueError) as e:
            logger.debug(f"Signal registration: {e}")

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Agent shutting down...")
        self._running = False

        # Print final summary
        uptime = time.time() - self._start_time
        logger.info(f"Agent stopped after {self._loop_count} cycles, {uptime:.0f}s uptime")

        # Print final stats
        print(self.stats_report())

    def stats_report(self) -> str:
        """Human-readable stats report."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        avg_cycle = (
            sum(self._cycle_times) / len(self._cycle_times)
            if self._cycle_times else 0
        )
        lines = [
            '─' * 40,
            'Agent Stats',
            f'  Uptime:      {uptime:.0f}s',
            f'  Cycles:      {self._loop_count}',
            f'  Avg cycle:   {avg_cycle:.2f}s',
        ]
        
        if self._reasoner:
            rs = self._reasoner.stats
            lines.extend([
                f'  LLM calls:   {rs["calls"]}',
                f'  Tokens today:{rs["tokens_today"]}',
                f'  Budget used: {rs["budget_used_pct"]}%',
            ])
        
        if self._decision_engine:
            ds = self._decision_engine.stats
            lines.extend([
                f'  Approved:    {ds["approved"]}',
                f'  Held:        {ds["held"]}',
                f'  Rejected:    {ds["rejected"]}',
            ])
        
        lines.append('─' * 40)
        return '\n'.join(lines)

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_requested = True

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def loop_count(self) -> int:
        return self._loop_count
