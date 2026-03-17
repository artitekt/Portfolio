#!/usr/bin/env python3
"""
Real-time AI Pipeline Demo

This script demonstrates a complete real-time AI processing pipeline with:
- Simulated event data generation
- Feature extraction and engineering
- AI inference
- Result publishing

Usage:
    python run_pipeline.py [--config config.yaml] [--duration 60] [--rate 2.0]
"""

import asyncio
import argparse
import signal
import sys
import time
from pathlib import Path

# Import from proper package structure
from realtime_ai_pipeline.pipeline.pipeline import RealtimePipeline
from realtime_ai_pipeline.utils.config import load_config, setup_logging
from realtime_ai_pipeline.utils.logger import get_logger


class PipelineDemo:
    """Real-time AI pipeline demonstration."""
    
    def __init__(self, config_path: str = None, duration: int = 60, event_rate: float = 1.0):
        self.config_path = config_path
        self.duration = duration
        self.event_rate = event_rate
        
        # Pipeline components
        self.pipeline = None
        self.logger = None
        
        # Demo state
        self.running = False
        self.start_time = None
        
        # Statistics
        self.demo_stats = {
            "events_processed": 0,
            "predictions_made": 0,
            "avg_latency": 0.0,
            "errors": 0
        }
    
    async def setup(self):
        """Setup the pipeline."""
        print("🚀 Setting up Real-time AI Pipeline...")
        print("   Pipeline initialized")
        
        # Load configuration
        try:
            if self.config_path:
                config = load_config(self.config_path)
            else:
                # Create demo config
                config_dict = self._create_demo_config()
                config = load_config(config_dict=config_dict)
            
            # Override event rate if specified
            if self.event_rate != 1.0:
                config.set('data_source.event_rate', self.event_rate)
            
            print("   ✅ Configuration loaded")
            
        except Exception as e:
            print(f"   ❌ Configuration error: {e}")
            return False
        
        # Setup logging
        try:
            self.logger = setup_logging(config)
            print("   ✅ Logging setup complete")
        except Exception as e:
            print(f"   ❌ Logging setup error: {e}")
            return False
        
        # Create pipeline
        try:
            self.pipeline = RealtimePipeline(config.to_dict())
            print("   ✅ Pipeline created")
        except Exception as e:
            print(f"   ❌ Pipeline creation error: {e}")
            return False
        
        return True
    
    async def start(self):
        """Start the pipeline demo."""
        if not await self.setup():
            return
        
        print("\\n🎯 Starting Real-time AI Pipeline Demo")
        print("=" * 60)
        
        self.running = True
        self.start_time = time.time()
        
        try:
            # Start pipeline
            print("   Components starting")
            await self.pipeline.start()
            print("   Streaming events")
            print("   Processing features")
            print("   Running AI inference")
            print("   ✅ Pipeline started successfully")
            
            # Show pipeline info
            await self._show_pipeline_info()
            
            # Run demo for specified duration
            await self._run_demo()
            
        except KeyboardInterrupt:
            print("\\n⏹️  Demo stopped by user")
        except Exception as e:
            print(f"❌ Demo error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the pipeline demo."""
        print("\\n🛑 Stopping pipeline...")
        
        self.running = False
        
        if self.pipeline:
            await self.pipeline.stop()
            print("✅ Pipeline stopped")
        
        # Show final statistics
        await self._show_final_stats()
    
    async def _run_demo(self):
        """Run the demo for specified duration."""
        end_time = self.start_time + self.duration
        
        print(f"📊 Running demo for {self.duration} seconds...")
        print("   Press Ctrl+C to stop early\\n")
        
        # Statistics update task
        stats_task = asyncio.create_task(self._update_stats_loop())
        
        try:
            while self.running and time.time() < end_time:
                await asyncio.sleep(1)
                
                # Show progress every 10 seconds
                elapsed = time.time() - self.start_time
                if int(elapsed) % 10 == 0:
                    await self._show_progress(elapsed)
        
        finally:
            stats_task.cancel()
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
    
    async def _update_stats_loop(self):
        """Update statistics periodically."""
        while self.running:
            try:
                if self.pipeline:
                    stats = self.pipeline.get_stats()
                    self.demo_stats["events_processed"] = stats.get("events_processed", 0)
                    self.demo_stats["predictions_made"] = stats.get("predictions_made", 0)
                    self.demo_stats["avg_latency"] = stats.get("avg_latency_ms", 0.0)
                    self.demo_stats["errors"] = stats.get("errors", 0)
                
                await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️  Stats update error: {e}")
                await asyncio.sleep(1)
    
    async def _show_pipeline_info(self):
        """Show pipeline information."""
        health = await self.pipeline.health_check()
        
        print("\\n📋 Pipeline Configuration:")
        print("   • Event Source: Simulated data stream")
        print(f"   • Event Rate: {self.event_rate} events/sec")
        print("   • Processing: Feature extraction + AI inference")
        print("   • Output: Console + optional file/webhook")
        
        print("\\n🔍 Pipeline Health:")
        for component, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {'Healthy' if status else 'Unhealthy'}")
    
    async def _show_progress(self, elapsed: float):
        """Show demo progress."""
        remaining = max(0, self.duration - elapsed)
        
        print(f"⏱️  Progress: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
        print(f"📊 Events: {self.demo_stats['events_processed']}, "
              f"Predictions: {self.demo_stats['predictions_made']}, "
              f"Avg Latency: {self.demo_stats['avg_latency']:.2f}ms")
    
    async def _show_final_stats(self):
        """Show final demo statistics."""
        if not self.pipeline:
            return
        
        print("\\nFINAL PIPELINE SUMMARY")
        print("----------------------")
        
        stats = self.pipeline.get_stats()
        
        # Format stats nicely
        events_processed = stats.get('events_processed', 0)
        features_generated = stats.get('features_generated', 0)
        predictions_made = stats.get('predictions_made', 0)
        results_published = stats.get('results_published', 0)
        errors = stats.get('errors', 0)
        avg_latency = stats.get('avg_latency_ms', 0)
        
        print(f"Events Processed:     {events_processed:5d}")
        print(f"Features Generated:   {features_generated:5d}")
        print(f"Predictions Made:     {predictions_made:5d}")
        print(f"Results Published:    {results_published:5d}")
        print(f"Errors:               {errors:5d}")
        print(f"Average Latency:      {avg_latency:6.2f}ms")
        
        # Additional performance info if available
        if stats.get('uptime_seconds'):
            uptime = stats['uptime_seconds']
            print(f"Uptime:               {uptime:6.2f}s")
            
            if uptime > 0:
                throughput = events_processed / uptime
                print(f"Throughput:           {throughput:6.2f} events/sec")
        
        print("\\n🎉 Demo completed successfully!")
        print("✅ Demonstrated:")
        print("   • Real-time event streaming")
        print("   • Async feature processing")
        print("   • AI inference pipeline")
        print("   • Result publishing")
        print("   • Performance monitoring")
    
    def _create_demo_config(self) -> dict:
        """Create demo configuration."""
        return {
            "data_source": {
                "event_rate": self.event_rate,
                "event_types": ["sensor", "user_action", "system_event"],
                "fields": ["value", "status", "metadata"]
            },
            "processor": {
                "features": {
                    "statistical": True,
                    "temporal": True,
                    "categorical": True,
                    "window_size": 10
                },
                "engineering": {
                    "window_sizes": [5, 10],
                    "lag_features": True,
                    "rolling_features": True,
                    "interaction_features": False,  # Disabled for demo
                    "frequency_features": False     # Disabled for demo
                }
            },
            "inference": {
                "model": {
                    "type": "mock",
                    "input_size": 10,
                    "confidence_threshold": 0.5
                },
                "mock": {
                    "noise_level": 0.1
                }
            },
            "publisher": {
                "console": True,
                "file": False,
                "webhook": False,
                "metrics": False  # Disabled for demo to avoid port conflicts
            },
            "logging": {
                "level": "INFO",
                "console": True
            }
        }


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Real-time AI Pipeline Demo")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--duration", type=int, default=60, help="Demo duration in seconds")
    parser.add_argument("--rate", type=float, default=1.0, help="Event rate (events/sec)")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = PipelineDemo(
        config_path=args.config,
        duration=args.duration,
        event_rate=args.rate
    )
    
    # Setup signal handling
    def signal_handler(signum, frame):
        print("\\n⏹️  Received signal, stopping demo...")
        demo.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run demo
    await demo.start()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required for this demo")
        sys.exit(1)
    
    # Run demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
