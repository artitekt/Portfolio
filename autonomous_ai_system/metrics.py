#!/usr/bin/env python3
"""
Latency and Performance Tracking Module

This module provides the LatencyTracker class for measuring and reporting
execution time for different stages of the autonomous AI system.
"""

import time
from typing import Dict, List, Optional
from collections import defaultdict
import statistics


class LatencyTracker:
    """
    Tracks execution time for different stages of the system.
    
    Responsibilities:
    - Measure execution time of functions
    - Store per-stage timings  
    - Compute averages and statistics
    """
    
    def __init__(self):
        """Initialize the latency tracker."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, stage_name: str) -> None:
        """Start timing a stage."""
        self.start_times[stage_name] = time.perf_counter()
    
    def stop_timer(self, stage_name: str) -> float:
        """
        Stop timing a stage and record the duration.
        
        Args:
            stage_name: Name of the stage to stop timing
            
        Returns:
            Duration in milliseconds
        """
        if stage_name not in self.start_times:
            raise ValueError(f"No start time found for stage: {stage_name}")
        
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_times[stage_name]) * 1000
        
        self.timings[stage_name].append(duration_ms)
        del self.start_times[stage_name]
        
        return duration_ms
    
    def get_average_latency(self, stage_name: str) -> Optional[float]:
        """Get average latency for a stage in milliseconds."""
        if stage_name not in self.timings or not self.timings[stage_name]:
            return None
        return statistics.mean(self.timings[stage_name])
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get average latencies for all stages."""
        return {
            stage: self.get_average_latency(stage)
            for stage in self.timings.keys()
            if self.get_average_latency(stage) is not None
        }
    
    def get_latest_timing(self, stage_name: str) -> Optional[float]:
        """Get the most recent timing for a stage."""
        if stage_name not in self.timings or not self.timings[stage_name]:
            return None
        return self.timings[stage_name][-1]
    
    def print_latest_latencies(self) -> None:
        """Print the latest latency measurements for all stages."""
        stage_names = ["data", "pipeline", "agent", "research"]
        
        for stage in stage_names:
            latest = self.get_latest_timing(stage)
            if latest is not None:
                print(f"[Latency] {stage.capitalize()}: {latest:.1f} ms")
    
    def print_summary(self) -> None:
        """Print summary statistics for all stages."""
        averages = self.get_all_averages()
        
        print("\nLatency Summary:")
        print("=" * 50)
        
        # Individual stage averages
        stage_display_names = {
            "data": "Data Fetch",
            "pipeline": "Pipeline Processing", 
            "agent": "Agent Decision",
            "research": "Research Logging"
        }
        
        for stage_key, display_name in stage_display_names.items():
            if stage_key in averages:
                print(f"average_{stage_key}_latency: {averages[stage_key]:.1f} ms")
        
        # Total average latency
        if averages:
            total_average = sum(averages.values())
            print(f"total_average_latency: {total_average:.1f} ms")
        
        print("=" * 50)
    
    def reset(self) -> None:
        """Reset all timing data."""
        self.timings.clear()
        self.start_times.clear()
