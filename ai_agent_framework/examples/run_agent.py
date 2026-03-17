#!/usr/bin/env python3
"""
Minimal working example of the AI Agent Framework.

This example demonstrates:
- Creating a mock observation
- Running one agent reasoning cycle
- Printing the decision output
"""
import asyncio
import os
import sys

# Import from proper package structure
from ai_agent_framework.agent.agent import Agent
from ai_agent_framework.agent.agent_config import AgentConfig, LLMConfig
from ai_agent_framework.llm.openai_provider import OpenAIProvider
from ai_agent_framework.llm.anthropic_provider import AnthropicProvider


async def run_single_cycle():
    """Run a single agent reasoning cycle for demonstration."""
    
    # Create configuration (you can also use AgentConfig.from_env() to read from environment)
    config = AgentConfig(
        mode='demo',
        dry_run=True,  # Don't actually apply changes
        loop_interval=30.0,
        log_level='INFO',
        llm=LLMConfig(
            provider='claude',  # or 'openai'
            model='claude-sonnet-4-20250514',
            claude_api_key=os.getenv('AGENT_CLAUDE_API_KEY', ''),
            openai_api_key=os.getenv('AGENT_OPENAI_API_KEY', ''),
            max_tokens=1000,
            timeout=30.0,
        )
    )
    
    print("🚀 AI Agent Framework - Single Cycle Demo")
    print("=" * 50)
    print(f"Mode: {config.mode}")
    print(f"Provider: {config.llm.provider}")
    print(f"Model: {config.llm.model}")
    print(f"Dry Run: {config.dry_run}")
    print("=" * 50)
    
    # Build the agent
    try:
        agent = Agent.build(config)
        print("✅ Agent built successfully")
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        return
    
    # Create a mock observation
    print("\n📊 Creating mock observation...")
    context = agent._observer.observe()
    
    print(f"  Timestamp: {context.timestamp}")
    print(f"  Mode: {context.mode}")
    print(f"  System Health: {context.system_health}")
    print(f"  Input Signals: {len(context.signals)}")
    
    # Show some mock signals
    for source, signal in list(context.signals.items())[:3]:
        print(f"    {source}: value={signal.value:.2f}, confidence={signal.confidence:.2f}")
    
    # Run reasoning
    print("\n🧠 Running LLM reasoning...")
    try:
        reasoning_result = await agent._reasoner.reason(context)
        
        print(f"  Success: {reasoning_result.success}")
        print(f"  Market Regime: {reasoning_result.market_regime}")
        print(f"  Confidence: {reasoning_result.regime_confidence:.2f}")
        print(f"  Strategy: {reasoning_result.reasoning_summary}")
        print(f"  Tokens Used: {reasoning_result.tokens_used}")
        print(f"  Latency: {reasoning_result.latency_ms:.0f}ms")
        
        if reasoning_result.human_readable:
            print(f"  Explanation: {reasoning_result.human_readable}")
        
    except Exception as e:
        print(f"❌ Reasoning failed: {e}")
        return
    
    # Run decision engine
    print("\n⚖️  Running decision engine...")
    try:
        decision_record = agent._decision_engine.evaluate(
            result=reasoning_result,
            context=context,
            is_killed=False,
            is_paused=False,
        )
        
        print(f"  Outcome: {decision_record.outcome.value}")
        print(f"  Approved: {decision_record.was_approved}")
        print(f"  Action Taken: {decision_record.action_taken}")
        
        if decision_record.approved_params:
            params = decision_record.approved_params
            print(f"  Parameter Updates:")
            print(f"    Min Confidence: {params.min_confidence:.3f}")
            print(f"    Strong Threshold: {params.strong_signal_threshold:.3f}")
            print(f"    Hold Threshold: {params.hold_threshold:.3f}")
        
        if decision_record.override_notes:
            print(f"  Override Notes: {', '.join(decision_record.override_notes)}")
            
    except Exception as e:
        print(f"❌ Decision evaluation failed: {e}")
        return
    
    print("\n✅ Single cycle completed successfully!")
    print("\n📈 Summary:")
    print(f"  - Observed {len(context.signals)} input signals")
    print(f"  - LLM identified regime: {reasoning_result.market_regime}")
    print(f"  - Decision outcome: {decision_record.outcome.value}")
    print(f"  - Parameters updated: {decision_record.action_taken}")


async def run_continuous_demo():
    """Run the agent in continuous mode for a few cycles."""
    
    config = AgentConfig.from_env()
    config.mode = 'demo'
    config.dry_run = True
    config.loop_interval = 5.0  # Fast cycle for demo
    
    print("🔄 AI Agent Framework - Continuous Demo (3 cycles)")
    print("=" * 50)
    print("Press Ctrl+C to stop early")
    print("=" * 50)
    
    agent = Agent.build(config)
    
    # Override the run method to limit cycles
    cycle_count = 0
    max_cycles = 3
    
    try:
        while cycle_count < max_cycles and not agent._shutdown_requested:
            print(f"\n--- Cycle {cycle_count + 1}/{max_cycles} ---")
            await agent._cycle()
            cycle_count += 1
            
            if cycle_count < max_cycles:
                print(f"⏳ Waiting {agent._config.loop_interval}s...")
                await asyncio.sleep(agent._config.loop_interval)
                
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    print(f"\n📊 Demo completed: {cycle_count} cycles")
    print(agent.stats_report())


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent Framework Demo")
    parser.add_argument(
        '--mode',
        choices=['single', 'continuous'],
        default='single',
        help='Demo mode: single cycle or continuous'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        asyncio.run(run_single_cycle())
    else:
        asyncio.run(run_continuous_demo())


if __name__ == '__main__':
    main()
