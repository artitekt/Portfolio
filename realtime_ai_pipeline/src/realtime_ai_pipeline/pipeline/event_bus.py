"""
Component: Async Event Bus
Role: Provides high-performance message passing between pipeline components
     using asyncio queues with topic-based routing. Handles message
     distribution, subscription management, and queue monitoring.
"""

"""
Async event bus for real-time message passing.
"""

import asyncio
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
import logging
from .message import EventMessage, FeatureMessage, PredictionMessage, ResultMessage

logger = logging.getLogger(__name__)


class EventBus:
    """Async event bus for pipeline communication."""
    
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
        self._tasks: List[asyncio.Task] = []
    
    async def publish(self, topic: str, message: Any) -> None:
        """Publish message to topic."""
        try:
            # Add to queue for async consumers
            await self.queues[topic].put(message)
            
            # Notify direct subscribers
            for callback in self.subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error publishing to {topic}: {e}")
    
    async def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to topic with callback."""
        self.subscribers[topic].append(callback)
        logger.info(f"Subscribed to {topic}")
    
    async def get_queue(self, topic: str) -> asyncio.Queue:
        """Get queue for topic consumption."""
        return self.queues[topic]
    
    async def start_consumer(self, topic: str, callback: Callable) -> asyncio.Task:
        """Start background consumer for topic."""
        async def consumer_loop():
            queue = self.queues[topic]
            while self.running:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Consumer error for {topic}: {e}")
                    await asyncio.sleep(0.01)
        
        task = asyncio.create_task(consumer_loop())
        self._tasks.append(task)
        return task
    
    def start(self) -> None:
        """Start event bus."""
        self.running = True
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop event bus and cleanup."""
        self.running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Event bus stopped")
    
    def get_queue_size(self, topic: str) -> int:
        """Get queue size for monitoring."""
        return self.queues[topic].qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "running": self.running,
            "topics": list(self.queues.keys()),
            "queue_sizes": {topic: queue.qsize() for topic, queue in self.queues.items()},
            "subscribers": {topic: len(calls) for topic, calls in self.subscribers.items()}
        }


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# Topic constants
EVENT_TOPIC = "events"
FEATURE_TOPIC = "features"
PREDICTION_TOPIC = "predictions"
RESULT_TOPIC = "results"
