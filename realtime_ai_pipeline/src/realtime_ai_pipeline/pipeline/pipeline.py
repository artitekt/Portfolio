"""
Component: Real-time AI Pipeline Orchestrator
Role: Coordinates all pipeline components and manages the overall flow
     of events through the processing stages. Handles statistics tracking,
     health monitoring, and component lifecycle management.
"""

"""
Main real-time AI pipeline orchestrator.
"""

import asyncio
import time
from typing import Dict, Any, Optional
import logging
from .event_bus import EventBus, get_event_bus, EVENT_TOPIC, FEATURE_TOPIC, PREDICTION_TOPIC, RESULT_TOPIC
from .message import EventMessage, FeatureMessage, PredictionMessage, ResultMessage
from ..ingestion.data_source import DataSource
from ..processing.processor import FeatureProcessor
from ..inference.inference_engine import InferenceEngine
from ..output.publisher import ResultPublisher

logger = logging.getLogger(__name__)


class RealtimePipeline:
    """Real-time AI processing pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.event_bus = get_event_bus()
        
        # Pipeline components
        self.data_source = DataSource(self.config.get("data_source", {}))
        self.feature_processor = FeatureProcessor(self.config.get("processor", {}))
        self.inference_engine = InferenceEngine(self.config.get("inference", {}))
        self.result_publisher = ResultPublisher(self.config.get("publisher", {}))
        
        # Pipeline state
        self.running = False
        self.stats = {
            "events_processed": 0,
            "features_generated": 0,
            "predictions_made": 0,
            "results_published": 0,
            "errors": 0,
            "start_time": None,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "stage_times": {
                "feature_processing_ms": 0.0,
                "inference_ms": 0.0,
                "publishing_ms": 0.0
            }
        }
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers for pipeline stages."""
        # Event to feature processing
        async def handle_event(event: EventMessage):
            try:
                # Increment event counter when event enters pipeline
                self.stats["events_processed"] += 1
                
                start_time = time.time()
                features = await self.feature_processor.process_event(event)
                processing_time = (time.time() - start_time) * 1000
                
                # Track feature processing time
                self.stats["stage_times"]["feature_processing_ms"] += processing_time
                
                feature_msg = FeatureMessage(
                    event_id=event.id,
                    features=features["features"],
                    feature_vector=features["vector"],
                    timestamp=event.timestamp,
                    processing_time_ms=processing_time
                )
                
                await self.event_bus.publish(FEATURE_TOPIC, feature_msg)
                self.stats["features_generated"] += 1
                
            except Exception as e:
                logger.error(f"Error processing event {event.id}: {e}")
                self.stats["errors"] += 1
        
        # Feature to inference
        async def handle_features(feature: FeatureMessage):
            try:
                start_time = time.time()
                prediction = await self.inference_engine.predict(feature)
                inference_time = (time.time() - start_time) * 1000
                
                # Track inference time
                self.stats["stage_times"]["inference_ms"] += inference_time
                
                prediction_msg = PredictionMessage(
                    event_id=feature.event_id,
                    prediction=prediction["value"],
                    confidence=prediction["confidence"],
                    model_version=self.inference_engine.get_model_version(),
                    timestamp=feature.timestamp,
                    inference_time_ms=inference_time
                )
                
                await self.event_bus.publish(PREDICTION_TOPIC, prediction_msg)
                self.stats["predictions_made"] += 1
                
            except Exception as e:
                logger.error(f"Error in inference for {feature.event_id}: {e}")
                self.stats["errors"] += 1
        
        # Prediction to result
        async def handle_prediction(prediction: PredictionMessage):
            try:
                start_time = time.time()
                
                # Get original features for result
                feature_queue = await self.event_bus.get_queue(FEATURE_TOPIC)
                original_feature = None
                
                # Simple lookup - in production would use proper correlation
                while not feature_queue.empty():
                    feature = await feature_queue.get()
                    if feature.event_id == prediction.event_id:
                        original_feature = feature
                        break
                
                processing_time = (time.time() - start_time) * 1000
                
                # Track publishing time
                self.stats["stage_times"]["publishing_ms"] += processing_time
                
                result_msg = ResultMessage(
                    event_id=prediction.event_id,
                    prediction=prediction.prediction,
                    confidence=prediction.confidence,
                    features=original_feature.features if original_feature else {},
                    processing_summary={
                        "feature_processing_ms": original_feature.processing_time_ms if original_feature else 0,
                        "inference_ms": prediction.inference_time_ms,
                        "publishing_ms": processing_time
                    },
                    timestamp=prediction.timestamp,
                    total_latency_ms=sum([
                        original_feature.processing_time_ms if original_feature else 0,
                        prediction.inference_time_ms,
                        processing_time
                    ])
                )
                
                await self.event_bus.publish(RESULT_TOPIC, result_msg)
                await self.result_publisher.publish(result_msg)
                self.stats["results_published"] += 1
                
                # Update average latency and total latency
                total_latency = result_msg.total_latency_ms
                self.stats["total_latency_ms"] += total_latency
                if self.stats["results_published"] > 0:
                    self.stats["avg_latency_ms"] = self.stats["total_latency_ms"] / self.stats["results_published"]
                
            except Exception as e:
                logger.error(f"Error publishing result for {prediction.event_id}: {e}")
                self.stats["errors"] += 1
        
        # Register handlers
        self.event_handlers = {
            EVENT_TOPIC: handle_event,
            FEATURE_TOPIC: handle_features,
            PREDICTION_TOPIC: handle_prediction
        }
    
    async def start(self):
        """Start the pipeline."""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        logger.info("Starting real-time AI pipeline")
        self.running = True
        self.stats["start_time"] = time.time()
        
        # Start event bus
        self.event_bus.start()
        
        # Start event consumers
        self.consumer_tasks = []
        for topic, handler in self.event_handlers.items():
            task = await self.event_bus.start_consumer(topic, handler)
            self.consumer_tasks.append(task)
        
        # Start data source
        await self.data_source.start()
        
        # Start result publisher
        await self.result_publisher.start()
        
        logger.info("Pipeline started successfully")
    
    async def stop(self):
        """Stop the pipeline."""
        if not self.running:
            return
        
        logger.info("Stopping pipeline")
        self.running = False
        
        # Stop data source
        await self.data_source.stop()
        
        # Stop result publisher
        await self.result_publisher.stop()
        
        # Stop event bus
        await self.event_bus.stop()
        
        # Cancel consumer tasks
        for task in self.consumer_tasks:
            task.cancel()
        
        await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        logger.info("Pipeline stopped")
    
    async def process_event(self, event_data: Dict[str, Any]) -> str:
        """Process a single event through the pipeline."""
        event = EventMessage(
            id=None,
            timestamp=time.time(),
            source="manual",
            data=event_data
        )
        
        await self.event_bus.publish(EVENT_TOPIC, event)
        
        return event.id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        if stats["start_time"]:
            stats["uptime_seconds"] = time.time() - stats["start_time"]
            stats["throughput_events_per_sec"] = (
                stats["events_processed"] / stats["uptime_seconds"]
                if stats["uptime_seconds"] > 0 else 0
            )
        
        # Add event bus stats
        stats["event_bus"] = self.event_bus.get_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all pipeline components."""
        health = {
            "pipeline": self.running,
            "data_source": self.data_source.is_healthy(),
            "feature_processor": self.feature_processor.is_healthy(),
            "inference_engine": self.inference_engine.is_healthy(),
            "result_publisher": self.result_publisher.is_healthy(),
            "event_bus": self.event_bus.running
        }
        
        return health
