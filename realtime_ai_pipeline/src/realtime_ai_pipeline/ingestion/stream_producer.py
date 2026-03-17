"""
Stream producer for high-throughput event publishing.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
import logging
from ..pipeline.event_bus import get_event_bus, EVENT_TOPIC
from ..pipeline.message import EventMessage

logger = logging.getLogger(__name__)


class StreamProducer:
    """High-throughput stream producer for event publishing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_bus = get_event_bus()
        self.running = False
        self.producer_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.batch_size = config.get("batch_size", 100)
        self.flush_interval = config.get("flush_interval", 0.1)  # seconds
        self.max_queue_size = config.get("max_queue_size", 10000)
        
        # Internal queue for batching
        self.event_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Statistics
        self.events_published = 0
        self.batches_published = 0
        self.queues_full = 0
    
    async def start(self):
        """Start the stream producer."""
        if self.running:
            logger.warning("Stream producer already running")
            return
        
        logger.info("Starting stream producer")
        self.running = True
        
        # Start producer task
        self.producer_task = asyncio.create_task(self._producer_loop())
    
    async def stop(self):
        """Stop the stream producer."""
        if not self.running:
            return
        
        logger.info("Stopping stream producer")
        self.running = False
        
        # Flush remaining events
        await self._flush_queue()
        
        # Stop producer task
        if self.producer_task:
            self.producer_task.cancel()
            try:
                await self.producer_task
            except asyncio.CancelledError:
                pass
    
    async def publish_event(self, event: EventMessage) -> bool:
        """Publish a single event."""
        try:
            await self.event_queue.put(event)
            return True
        except asyncio.QueueFull:
            self.queues_full += 1
            logger.warning("Event queue full, dropping event")
            return False
    
    async def publish_batch(self, events: List[EventMessage]) -> int:
        """Publish a batch of events."""
        published_count = 0
        for event in events:
            if await self.publish_event(event):
                published_count += 1
            else:
                break
        
        return published_count
    
    async def _producer_loop(self):
        """Main producer loop for batching and publishing."""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Wait for event or timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), 
                        timeout=self.flush_interval
                    )
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass
                
                # Check if we should flush
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and (current_time - last_flush) >= self.flush_interval)
                )
                
                if should_flush and batch:
                    await self._publish_batch(batch)
                    batch = []
                    last_flush = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in producer loop: {e}")
                await asyncio.sleep(0.1)
        
        # Final flush
        if batch:
            await self._publish_batch(batch)
    
    async def _publish_batch(self, batch: List[EventMessage]):
        """Publish a batch of events to event bus."""
        try:
            # Publish all events in batch
            for event in batch:
                await self.event_bus.publish(EVENT_TOPIC, event)
            
            self.events_published += len(batch)
            self.batches_published += 1
            
            logger.debug(f"Published batch of {len(batch)} events")
            
        except Exception as e:
            logger.error(f"Error publishing batch: {e}")
    
    async def _flush_queue(self):
        """Flush all remaining events in queue."""
        batch = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                batch.append(event)
            except asyncio.QueueEmpty:
                break
        
        if batch:
            await self._publish_batch(batch)
            logger.info(f"Flushed {len(batch)} remaining events")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        return {
            "running": self.running,
            "events_published": self.events_published,
            "batches_published": self.batches_published,
            "queue_size": self.event_queue.qsize(),
            "queues_full": self.queues_full,
            "avg_batch_size": (
                self.events_published / self.batches_published
                if self.batches_published > 0 else 0
            )
        }
    
    def is_healthy(self) -> bool:
        """Check if producer is healthy."""
        if not self.running:
            return False
        
        # Check queue utilization
        queue_utilization = self.event_queue.qsize() / self.max_queue_size
        if queue_utilization > 0.9:
            return False
        
        return True


class BackpressureProducer(StreamProducer):
    """Stream producer with backpressure handling."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.backpressure_threshold = config.get("backpressure_threshold", 0.8)
        self.drop_strategy = config.get("drop_strategy", "oldest")  # "oldest", "newest", "random"
    
    async def publish_event(self, event: EventMessage) -> bool:
        """Publish event with backpressure handling."""
        queue_size = self.event_queue.qsize()
        max_size = self.event_queue.maxsize
        
        # Check if we need to apply backpressure
        if queue_size >= max_size * self.backpressure_threshold:
            await self._apply_backpressure()
        
        return await super().publish_event(event)
    
    async def _apply_backpressure(self):
        """Apply backpressure strategy."""
        if self.drop_strategy == "oldest":
            # Drop oldest events
            drop_count = max(1, self.event_queue.qsize() // 10)
            for _ in range(drop_count):
                try:
                    self.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
        elif self.drop_strategy == "newest":
            # Signal not to accept new events (handled by queue full)
            pass
        
        elif self.drop_strategy == "random":
            # Drop random events
            drop_count = max(1, self.event_queue.qsize() // 10)
            for _ in range(drop_count):
                try:
                    # Simple random drop - in practice would need more sophisticated
                    self.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        logger.warning(f"Applied backpressure: {self.drop_strategy}")


class MetricsProducer(StreamProducer):
    """Stream producer that generates metrics events."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics_config = config.get("metrics", {})
        self.metric_types = self.metrics_config.get("types", ["cpu", "memory", "network"])
    
    async def start(self):
        """Start generating metrics."""
        await super().start()
        
        # Start metrics generation task
        self.metrics_task = asyncio.create_task(self._generate_metrics())
    
    async def stop(self):
        """Stop metrics generation."""
        await super().stop()
        
        if hasattr(self, 'metrics_task'):
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
    
    async def _generate_metrics(self):
        """Generate metrics events."""
        while self.running:
            try:
                metric_event = self._create_metric_event()
                await self.publish_event(metric_event)
                
                await asyncio.sleep(1.0)  # Generate metrics every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating metrics: {e}")
                await asyncio.sleep(1.0)
    
    def _create_metric_event(self) -> EventMessage:
        """Create a metrics event."""
        metric_type = random.choice(self.metric_types)
        
        if metric_type == "cpu":
            data = {
                "metric": "cpu_usage",
                "value": random.uniform(0.1, 1.0),
                "cores": random.randint(1, 8),
                "load_avg": random.uniform(0.5, 2.0)
            }
        elif metric_type == "memory":
            data = {
                "metric": "memory_usage",
                "value": random.uniform(0.3, 0.9),
                "total_gb": random.uniform(8, 64),
                "available_gb": random.uniform(1, 32)
            }
        else:  # network
            data = {
                "metric": "network_throughput",
                "value": random.uniform(1000000, 10000000),  # bytes/sec
                "interface": random.choice(["eth0", "wlan0", "lo"]),
                "packets_per_sec": random.randint(100, 10000)
            }
        
        return EventMessage(
            id=None,
            timestamp=time.time(),
            source="metrics",
            data=data
        )
