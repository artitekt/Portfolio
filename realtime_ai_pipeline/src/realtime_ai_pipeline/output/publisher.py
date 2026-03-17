"""
Component: Result Publisher
Role: Publishes pipeline results to multiple output channels including
     console display, file storage, HTTP webhooks, and monitoring metrics.
     Handles formatting, batching, and delivery confirmation.
"""

"""
Result publisher for real-time AI pipeline outputs.
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
import logging
from ..pipeline.message import ResultMessage

logger = logging.getLogger(__name__)


class ResultPublisher:
    """Publisher for real-time AI pipeline results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.publisher_config = config.get("publisher", {})
        
        # Publisher configuration
        self.enable_console = self.publisher_config.get("console", True)
        self.enable_file = self.publisher_config.get("file", False)
        self.enable_webhook = self.publisher_config.get("webhook", False)
        self.enable_metrics = self.publisher_config.get("metrics", True)
        
        # File configuration
        self.file_path = self.publisher_config.get("file_path", "results.jsonl")
        self.file_handle = None
        
        # Webhook configuration
        self.webhook_url = self.publisher_config.get("webhook_url")
        self.webhook_timeout = self.publisher_config.get("webhook_timeout", 5.0)
        
        # Metrics configuration
        self.metrics_port = self.publisher_config.get("metrics_port", 8080)
        self.metrics_endpoint = self.publisher_config.get("metrics_endpoint", "/metrics")
        
        # Publishing state
        self.running = False
        self.publish_queue = asyncio.Queue()
        self.publisher_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.publish_stats = {
            "total_published": 0,
            "console_published": 0,
            "file_published": 0,
            "webhook_published": 0,
            "webhook_failures": 0,
            "avg_publish_time": 0.0,
            "total_publish_time": 0.0
        }
        
        # Result storage for metrics
        self.recent_results = []
        self.max_recent_results = 1000
    
    async def start(self):
        """Start the result publisher."""
        if self.running:
            logger.warning("Result publisher already running")
            return
        
        logger.info("Starting result publisher")
        self.running = True
        
        # Setup file output
        if self.enable_file:
            await self._setup_file_output()
        
        # Setup metrics server
        if self.enable_metrics:
            await self._setup_metrics_server()
        
        # Start publisher task
        self.publisher_task = asyncio.create_task(self._publisher_loop())
    
    async def stop(self):
        """Stop the result publisher."""
        if not self.running:
            return
        
        logger.info("Stopping result publisher")
        self.running = False
        
        # Cancel publisher task
        if self.publisher_task:
            self.publisher_task.cancel()
            try:
                await self.publisher_task
            except asyncio.CancelledError:
                pass
        
        # Close file handle
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        # Flush remaining results
        await self._flush_queue()
    
    async def publish(self, result: ResultMessage):
        """Publish a result message."""
        try:
            await self.publish_queue.put(result)
        except Exception as e:
            logger.error(f"Error queuing result for publishing: {e}")
    
    async def _publisher_loop(self):
        """Main publisher loop."""
        while self.running:
            try:
                # Wait for result with timeout
                try:
                    result = await asyncio.wait_for(self.publish_queue.get(), timeout=0.1)
                    await self._publish_result(result)
                except asyncio.TimeoutError:
                    continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in publisher loop: {e}")
                await asyncio.sleep(0.1)
        
        # Flush remaining results
        await self._flush_queue()
    
    async def _publish_result(self, result: ResultMessage):
        """Publish a single result to all configured outputs."""
        start_time = time.time()
        
        try:
            # Console output
            if self.enable_console:
                await self._publish_console(result)
                self.publish_stats["console_published"] += 1
            
            # File output
            if self.enable_file:
                await self._publish_file(result)
                self.publish_stats["file_published"] += 1
            
            # Webhook output
            if self.enable_webhook:
                success = await self._publish_webhook(result)
                if success:
                    self.publish_stats["webhook_published"] += 1
                else:
                    self.publish_stats["webhook_failures"] += 1
            
            # Update statistics
            publish_time = (time.time() - start_time) * 1000
            self.publish_stats["total_published"] += 1
            self.publish_stats["total_publish_time"] += publish_time
            self.publish_stats["avg_publish_time"] = (
                self.publish_stats["total_publish_time"] / self.publish_stats["total_published"]
            )
            
            # Store for metrics
            self._store_result(result)
            
            logger.debug(f"Published result {result.event_id} in {publish_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error publishing result {result.event_id}: {e}")
    
    async def _publish_console(self, result: ResultMessage):
        """Publish result to console."""
        timestamp = time.strftime("%H:%M:%S", time.localtime(result.timestamp))
        
        print(f"\n🎯 PREDICTION RESULT [{timestamp}]")
        print(f"   Event ID: {result.event_id}")
        print(f"   Prediction: {result.prediction:.4f}")
        print(f"   Confidence: {result.confidence:.4f}")
        print(f"   Total Latency: {result.total_latency_ms:.2f}ms")
        
        if result.processing_summary:
            print("   Processing Breakdown:")
            for stage, time_ms in result.processing_summary.items():
                print(f"     {stage}: {time_ms:.2f}ms")
        
        if result.features:
            print(f"   Features: {len(result.features)} extracted")
        
        print("-" * 50)
    
    async def _publish_file(self, result: ResultMessage):
        """Publish result to file."""
        if not self.file_handle:
            return
        
        try:
            # Convert result to JSON
            result_dict = {
                "event_id": result.event_id,
                "timestamp": result.timestamp,
                "prediction": result.prediction,
                "confidence": result.confidence,
                "total_latency_ms": result.total_latency_ms,
                "processing_summary": result.processing_summary,
                "features_count": len(result.features) if result.features else 0
            }
            
            # Write to file
            line = json.dumps(result_dict) + "\n"
            self.file_handle.write(line)
            self.file_handle.flush()
            
        except Exception as e:
            logger.error(f"Error writing to file: {e}")
    
    async def _publish_webhook(self, result: ResultMessage) -> bool:
        """Publish result via webhook."""
        if not self.webhook_url:
            return False
        
        try:
            import aiohttp
            
            # Prepare payload
            payload = {
                "event_id": result.event_id,
                "timestamp": result.timestamp,
                "prediction": result.prediction,
                "confidence": result.confidence,
                "latency_ms": result.total_latency_ms,
                "features": result.features,
                "processing_summary": result.processing_summary
            }
            
            # Send webhook
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.webhook_timeout)) as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.warning(f"Webhook failed with status {response.status}")
                        return False
        
        except ImportError:
            logger.error("aiohttp not available for webhook publishing")
            return False
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            return False
    
    async def _setup_file_output(self):
        """Setup file output."""
        try:
            self.file_handle = open(self.file_path, 'a')
            logger.info(f"File output setup: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to setup file output: {e}")
            self.enable_file = False
    
    async def _setup_metrics_server(self):
        """Setup metrics HTTP server."""
        try:
            # Simple HTTP server for metrics
            from aiohttp import web
            
            app = web.Application()
            app.router.add_get(self.metrics_endpoint, self._metrics_handler)
            
            # Start server in background
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', self.metrics_port)
            await site.start()
            
            logger.info(f"Metrics server started on http://localhost:{self.metrics_port}{self.metrics_endpoint}")
            
        except ImportError:
            logger.warning("aiohttp not available, metrics server disabled")
            self.enable_metrics = False
        except Exception as e:
            logger.error(f"Failed to setup metrics server: {e}")
            self.enable_metrics = False
    
    async def _metrics_handler(self, request):
        """Handle metrics HTTP request."""
        try:
            metrics = self.get_metrics()
            return web.json_response(metrics)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    def _store_result(self, result: ResultMessage):
        """Store result for metrics calculation."""
        self.recent_results.append(result)
        
        # Maintain window size
        if len(self.recent_results) > self.max_recent_results:
            self.recent_results = self.recent_results[-self.max_recent_results:]
    
    async def _flush_queue(self):
        """Flush remaining results in queue."""
        while not self.publish_queue.empty():
            try:
                result = self.publish_queue.get_nowait()
                await self._publish_result(result)
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error flushing result: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        stats = self.publish_stats.copy()
        
        if stats["webhook_published"] > 0:
            stats["webhook_success_rate"] = (
                stats["webhook_published"] / (stats["webhook_published"] + stats["webhook_failures"])
            )
        else:
            stats["webhook_success_rate"] = 0.0
        
        stats["queue_size"] = self.publish_queue.qsize()
        stats["running"] = self.running
        
        return stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics."""
        metrics = {
            "publisher": self.get_stats(),
            "recent_performance": self._calculate_recent_metrics(),
            "outputs": {
                "console": self.enable_console,
                "file": self.enable_file,
                "webhook": self.enable_webhook,
                "metrics": self.enable_metrics
            }
        }
        
        return metrics
    
    def _calculate_recent_metrics(self) -> Dict[str, Any]:
        """Calculate metrics from recent results."""
        if not self.recent_results:
            return {}
        
        recent_predictions = [r.prediction for r in self.recent_results]
        recent_confidences = [r.confidence for r in self.recent_results]
        recent_latencies = [r.total_latency_ms for r in self.recent_results]
        
        return {
            "count": len(self.recent_results),
            "prediction_stats": {
                "mean": float(np.mean(recent_predictions)),
                "std": float(np.std(recent_predictions)),
                "min": float(np.min(recent_predictions)),
                "max": float(np.max(recent_predictions))
            },
            "confidence_stats": {
                "mean": float(np.mean(recent_confidences)),
                "std": float(np.std(recent_confidences)),
                "min": float(np.min(recent_confidences)),
                "max": float(np.max(recent_confidences))
            },
            "latency_stats": {
                "mean": float(np.mean(recent_latencies)),
                "std": float(np.std(recent_latencies)),
                "min": float(np.min(recent_latencies)),
                "max": float(np.max(recent_latencies)),
                "p95": float(np.percentile(recent_latencies, 95))
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if publisher is healthy."""
        if not self.running:
            return False
        
        # Check webhook success rate
        if self.enable_webhook and self.publish_stats["webhook_published"] > 0:
            success_rate = self.publish_stats["webhook_published"] / (
                self.publish_stats["webhook_published"] + self.publish_stats["webhook_failures"]
            )
            if success_rate < 0.8:  # Less than 80% success rate
                return False
        
        # Check average publish time
        if self.publish_stats["avg_publish_time"] > 1000:  # More than 1 second
            return False
        
        return True
    
    def reset_stats(self):
        """Reset publisher statistics."""
        self.publish_stats = {
            "total_published": 0,
            "console_published": 0,
            "file_published": 0,
            "webhook_published": 0,
            "webhook_failures": 0,
            "avg_publish_time": 0.0,
            "total_publish_time": 0.0
        }
        self.recent_results.clear()


# Import numpy for metrics calculation
import numpy as np
