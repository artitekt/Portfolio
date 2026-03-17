#!/usr/bin/env python3
"""
Autonomous AI System Demo using Adapters

This script demonstrates the integration of three AI systems using adapter classes:
1. Real-time AI Pipeline - Event generation and processing
2. AI Agent Framework - Autonomous reasoning and decision making  
3. AI Research Platform - Experiment tracking and evaluation

The adapters provide a unified interface without modifying the original systems.
"""

import sys
import os
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import the adapter classes
from adapters import PipelineAdapter, AgentAdapter, ResearchAdapter
from live_data import LiveDataClient
from metrics import LatencyTracker


class AutonomousAISystem:
    """Main integration class using adapters."""
    
    def __init__(self):
        """Initialize the autonomous AI system with adapters."""
        self.pipeline = None
        self.agent = None
        self.research = None
        self.live_client = None
        self.latency_tracker = LatencyTracker()
    
    async def initialize(self):
        """Initialize all system components using adapters."""
        print("🚀 Initializing Autonomous AI System with Adapters...")
        
        try:
            # Initialize adapters
            print("   📡 Setting up Pipeline Adapter...")
            self.pipeline = PipelineAdapter()
            
            print("   🤖 Setting up Agent Adapter...")
            self.agent = AgentAdapter()
            
            print("   📊 Setting up Research Adapter...")
            self.research = ResearchAdapter()
            
            print("   🌐 Setting up Live Data Client...")
            self.live_client = LiveDataClient()
            
            print("🎉 All adapters initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def run_demo(self):
        """Run the complete autonomous AI system demo using adapters."""
        print("🎯 Starting Autonomous AI System Demo with Adapters")
        print("="*60)
        
        try:
            # Stage 1: Initialize all adapters
            if not await self.initialize():
                return False
            
            # Stage 2: Create live event stream
            print("\n📡 Starting Live Event Stream...")
            print("Fetching real-time Bitcoin price data from CoinGecko API...")
            
            # Process 10 live events
            num_events = 10
            print(f"Processing {num_events} live events...")
            
            for i in range(num_events):
                # Data fetch stage
                self.latency_tracker.start_timer("data")
                event = self.live_client.get_event()
                self.latency_tracker.stop_timer("data")
                
                # Add event_id for compatibility
                event["event_id"] = f"live_event_{i}"
                
                # Pipeline processing stage
                self.latency_tracker.start_timer("pipeline")
                processed_signal = self.pipeline.process(event)
                self.latency_tracker.stop_timer("pipeline")
                
                # Agent decision stage
                self.latency_tracker.start_timer("agent")
                decision = self.agent.decide(processed_signal)
                self.latency_tracker.stop_timer("agent")
                
                # Research logging stage
                self.latency_tracker.start_timer("research")
                self.research.log(decision, processed_signal)
                self.latency_tracker.stop_timer("research")
                
                # Print latest latencies
                self.latency_tracker.print_latest_latencies()
                print()  # Add spacing between events
                
                # Rate limiting - wait 1 second between API calls
                if i < num_events - 1:  # Don't wait after the last event
                    await asyncio.sleep(1)
            
            # Stage 3: Generate summary
            print("\n📊 Generating Final Summary...")
            summary = self.research.summarize()
            
            # Stage 4: Display results
            self.display_results(summary)
            
            # Stage 5: Print latency summary
            self.latency_tracker.print_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_results(self, summary: Dict[str, Any]):
        """Display the final results."""
        print("\n" + "="*60)
        print("AUTONOMOUS AI SYSTEM DEMO RESULTS")
        print("="*60)
        
        print(f"Total Events Processed: {summary['total_decisions']}")
        print(f"Successful Decisions: {summary['successful_decisions']}")
        print(f"Decision Accuracy: {summary['accuracy']:.3f}")
        print(f"Average Confidence: {summary.get('average_confidence', 0):.3f}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        # Reinforcement Learning Metrics
        print(f"\nREINFORCEMENT LEARNING METRICS")
        print(f"Total Reward: {summary.get('total_reward', 0)}")
        print(f"Average Reward: {summary.get('average_reward', 0):.3f}")
        print(f"Training Steps: {summary.get('num_steps', 0)}")
        
        # Show decision distribution
        if summary.get('decision_distribution'):
            print(f"\nDecision Distribution:")
            for decision_type, count in summary['decision_distribution'].items():
                percentage = (count / summary['total_decisions']) * 100
                print(f"  {decision_type}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*60)
        print("SYSTEM INTEGRATION STATUS")
        print("="*60)
        print("✅ Pipeline Adapter: Event processing successful")
        print("✅ Agent Adapter: Decision making functional")
        print("✅ Research Adapter: Evaluation and logging working")
        print("✅ Live Data Client: Real-time API integration active")
        print("✅ End-to-End Integration: All adapters connected with live data")
        
        print("\n🚀 Autonomous AI System Demo with Live Data Completed Successfully!")


async def main():
    """Main entry point for the autonomous AI system demo."""
    print("Autonomous AI Decision System with Adapters")
    print("==========================================")
    print("Integrating Pipeline + Agent + Research via Adapters")
    print("Now with LIVE DATA from CoinGecko API")
    print()
    
    # Create and run the autonomous system
    system = AutonomousAISystem()
    success = await system.run_demo()
    
    if success:
        print("\n✅ All systems integrated successfully via adapters!")
        print("The autonomous AI system demonstrated:")
        print("  • Real-time event processing through Pipeline Adapter")
        print("  • AI agent decisions through Agent Adapter")
        print("  • Performance evaluation through Research Adapter")
        print("  • Live data integration via CoinGecko API")
        print("  • End-to-end system integration with adapter layer")
    else:
        print("\n❌ System integration failed")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)