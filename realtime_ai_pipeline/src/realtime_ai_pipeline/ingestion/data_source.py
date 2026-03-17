"""
Generic data source for real-time event streaming.
"""

import asyncio
import random
import time
from typing import Dict, Any, Optional, AsyncGenerator
import logging
from ..pipeline.event_bus import get_event_bus, EVENT_TOPIC
from ..pipeline.message import EventMessage

logger = logging.getLogger(__name__)


class DataSource:
    """Generic data source for real-time event streaming."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_bus = get_event_bus()
        self.running = False
        self.generation_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.event_rate = config.get("event_rate", 1.0)  # events per second
        self.event_types = config.get("event_types", ["sensor", "user_action", "system_event"])
        self.fields = config.get("fields", ["value", "status", "metadata"])
        
        # Statistics
        self.events_generated = 0
        self.start_time = None
    
    async def start(self):
        """Start generating events."""
        if self.running:
            logger.warning("Data source already running")
            return
        
        logger.info(f"Starting data source with rate: {self.event_rate} events/sec")
        self.running = True
        self.start_time = time.time()
        
        # Start event generation task
        self.generation_task = asyncio.create_task(self._generate_events())
    
    async def stop(self):
        """Stop generating events."""
        if not self.running:
            return
        
        logger.info("Stopping data source")
        self.running = False
        
        if self.generation_task:
            self.generation_task.cancel()
            try:
                await self.generation_task
            except asyncio.CancelledError:
                pass
    
    async def _generate_events(self):
        """Generate events at configured rate."""
        interval = 1.0 / self.event_rate if self.event_rate > 0 else 1.0
        
        while self.running:
            try:
                # Generate event
                event = self._create_event()
                
                # Publish to event bus
                await self.event_bus.publish(EVENT_TOPIC, event)
                self.events_generated += 1
                
                # Log progress periodically
                if self.events_generated % 100 == 0:
                    logger.info(f"Generated {self.events_generated} events")
                
                # Wait for next event
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating event: {e}")
                await asyncio.sleep(0.1)
    
    def _create_event(self) -> EventMessage:
        """Create a simulated event."""
        event_type = random.choice(self.event_types)
        
        # Generate event data based on type
        if event_type == "sensor":
            data = {
                "sensor_id": f"sensor_{random.randint(1, 10)}",
                "value": random.uniform(0, 100),
                "unit": random.choice(["celsius", "percent", "voltage"]),
                "status": random.choice(["normal", "warning", "critical"]),
                "location": random.choice(["zone_a", "zone_b", "zone_c"])
            }
        elif event_type == "user_action":
            data = {
                "user_id": f"user_{random.randint(1, 100)}",
                "action": random.choice(["click", "view", "purchase", "login"]),
                "page": random.choice(["home", "products", "checkout", "profile"]),
                "duration_ms": random.randint(100, 5000),
                "success": random.choice([True, False])
            }
        else:  # system_event
            data = {
                "component": random.choice(["api", "database", "cache", "queue"]),
                "metric": random.choice(["cpu", "memory", "latency", "error_rate"]),
                "value": random.uniform(0.1, 1.0),
                "threshold": random.uniform(0.5, 0.9),
                "severity": random.choice(["info", "warning", "error", "critical"])
            }
        
        return EventMessage(
            id=None,  # Will be auto-generated
            timestamp=time.time(),
            source=event_type,
            data=data
        )
    
    async def generate_batch(self, count: int) -> list:
        """Generate a batch of events for testing."""
        events = []
        for _ in range(count):
            event = self._create_event()
            events.append(event)
            await self.event_bus.publish(EVENT_TOPIC, event)
            self.events_generated += 1
        
        logger.info(f"Generated batch of {count} events")
        return events
    
    def is_healthy(self) -> bool:
        """Check if data source is healthy."""
        if not self.running:
            return False
        
        # Check if we're generating events
        if self.start_time and self.events_generated > 0:
            elapsed = time.time() - self.start_time
            expected_events = int(elapsed * self.event_rate)
            # Allow 20% tolerance
            return self.events_generated >= expected_events * 0.8
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data source statistics."""
        stats = {
            "running": self.running,
            "events_generated": self.events_generated,
            "event_rate": self.event_rate,
            "event_types": self.event_types
        }
        
        if self.start_time:
            stats["uptime_seconds"] = time.time() - self.start_time
            if stats["uptime_seconds"] > 0:
                stats["actual_rate"] = self.events_generated / stats["uptime_seconds"]
        
        return stats


class FileDataSource(DataSource):
    """Data source that reads from file for batch processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.file_path = config.get("file_path")
        self.file_format = config.get("file_format", "json")
    
    async def start(self):
        """Start reading from file."""
        logger.info(f"Starting file data source: {self.file_path}")
        self.running = True
        
        # Read and publish all events from file
        async for event in self._read_file():
            if not self.running:
                break
            await self.event_bus.publish(EVENT_TOPIC, event)
            self.events_generated += 1
        
        logger.info(f"File data source completed: {self.events_generated} events")
    
    async def _read_file(self) -> AsyncGenerator[EventMessage, None]:
        """Read events from file."""
        # This is a placeholder - in real implementation would read actual files
        # For demo, generate some sample events
        for i in range(50):
            event = self._create_event()
            yield event
            await asyncio.sleep(0.01)  # Small delay to prevent overwhelming


class APIDataSource(DataSource):
    """Data source that pulls from API endpoints."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = config.get("api_url")
        self.poll_interval = config.get("poll_interval", 5.0)
        self.headers = config.get("headers", {})
    
    async def start(self):
        """Start polling API."""
        logger.info(f"Starting API data source: {self.api_url}")
        self.running = True
        
        self.generation_task = asyncio.create_task(self._poll_api())
    
    async def _poll_api(self):
        """Poll API for data."""
        while self.running:
            try:
                # This is a placeholder - in real implementation would make HTTP requests
                # For demo, generate events that look like API responses
                event = self._create_api_event()
                await self.event_bus.publish(EVENT_TOPIC, event)
                self.events_generated += 1
                
                await asyncio.sleep(self.poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error polling API: {e}")
                await asyncio.sleep(1.0)
    
    def _create_api_event(self) -> EventMessage:
        """Create event that looks like API response."""
        return EventMessage(
            id=None,
            timestamp=time.time(),
            source="api",
            data={
                "endpoint": random.choice(["/metrics", "/status", "/events"]),
                "response_time_ms": random.randint(50, 500),
                "status_code": random.choice([200, 201, 400, 500]),
                "data_size_bytes": random.randint(100, 10000)
            }
        )
