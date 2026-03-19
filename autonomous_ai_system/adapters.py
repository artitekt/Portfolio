#!/usr/bin/env python3
"""
Adapter Layer for Autonomous AI System Integration

This file provides adapter classes that wrap the three AI systems
to create a unified interface without modifying the original systems.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import statistics
import math

# Add src directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / "ai_agent_framework" / "src"))
sys.path.append(str(Path(__file__).parent.parent / "realtime_ai_pipeline" / "src"))
sys.path.append(str(Path(__file__).parent.parent / "ai_research_platform" / "src"))

# Import from the three systems
try:
    # Realtime Pipeline imports
    from realtime_ai_pipeline.pipeline.pipeline import RealtimePipeline
    from realtime_ai_pipeline.utils.config import Config
    
    # AI Agent imports
    from ai_agent_framework.agent.agent import Agent
    from ai_agent_framework.agent.agent_config import AgentConfig
    
    # Research Platform imports
    from ai_research_platform.experiments.experiment_runner import ExperimentRunner
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all three projects are in the parent directory")
    sys.exit(1)


class PipelineAdapter:
    """Adapter for Realtime AI Pipeline."""
    
    def __init__(self, window_size: int = 20, prediction_steps: int = 5):
        """Initialize the pipeline adapter.
        
        Args:
            window_size: Number of prices to store for volatility calculation
            prediction_steps: Number of steps back for trend prediction
        """
        try:
            # Initialize with real pipeline config
            self.config = Config()
            self.pipeline = RealtimePipeline(self.config)
            print("[Pipeline] Initialized with real pipeline components")
        except Exception as e:
            print(f"[Pipeline] Initialization error: {e}")
            # Fallback to minimal initialization
            self.pipeline = None
        
        # Rolling window configuration
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        
        # Initialize rolling window and tracking
        self.price_window: List[float] = []
        self.price_change_window: List[float] = []
        self.previous_price: Optional[float] = None
        
        print(f"[Pipeline] Rolling window configured: size={window_size}, prediction_steps={prediction_steps}")
    
    def process(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single event through the pipeline with rolling window analysis.
        
        Args:
            event: Dictionary containing event data (price, volume, timestamp)
            
        Returns:
            Dictionary with processed features and predictions
        """
        try:
            print(f"[Pipeline] Processing event: {event.get('event_id', 'unknown')}")
            
            # Extract features from event
            current_price = event.get('price', 100)
            volume = event.get('volume', 50)
            
            # Handle first event case
            if self.previous_price is None:
                prediction = 0
                confidence = 0
                price_change = 0
                volatility = 0
                momentum = 0
                print(f"[Pipeline Debug] First event - initializing windows, setting neutral values")
            else:
                # Calculate instantaneous price change
                price_change = current_price - self.previous_price
                price_change_pct = price_change / self.previous_price
                
                # Update rolling windows
                self.price_window.append(current_price)
                self.price_change_window.append(price_change_pct)
                
                # Maintain window size
                if len(self.price_window) > self.window_size:
                    self.price_window.pop(0)
                    self.price_change_window.pop(0)
                
                # Calculate rolling window prediction
                prediction, momentum = self._calculate_rolling_prediction()
                
                # Calculate volatility (standard deviation of price changes)
                volatility = self._calculate_volatility()
                
                # Enhanced confidence based on volatility and prediction magnitude
                confidence = min(1.0, abs(prediction) * 100 / (volatility + 0.0001))
                
                # Debug logging
                print(f"[Pipeline Debug] price_change={price_change:.6f}, price_change_pct={price_change_pct:.6f}")
                print(f"[Pipeline Debug] window_size={len(self.price_window)}, prediction={prediction:.6f}, momentum={momentum:.6f}")
                print(f"[Pipeline Debug] volatility={volatility:.6f}, confidence={confidence:.6f}")
                
                if len(self.price_window) >= 5:
                    print(f"[Pipeline Debug] price_window (last 5): {[f'{p:.2f}' for p in self.price_window[-5:]]}")
                    print(f"[Pipeline Debug] change_window (last 5): {[f'{c:.6f}' for c in self.price_change_window[-5:]]}")
            
            # Create processed signal with enhanced features
            processed_signal = {
                'event_id': event.get('event_id', f"event_{datetime.now().timestamp()}"),
                'timestamp': event.get('timestamp', datetime.now().isoformat()),
                'features': {
                    'price_change': price_change,
                    'volume_normalized': volume / 100,
                    'price_volatility': volatility,
                    'momentum': momentum,
                    'window_size': len(self.price_window),
                    'prediction_steps': self.prediction_steps
                },
                'prediction': prediction,
                'confidence': confidence,
                'volatility': volatility,
                'momentum': momentum
            }
            
            print(f"[Pipeline] Processed: prediction={processed_signal['prediction']:.6f}, "
                  f"confidence={processed_signal['confidence']:.6f}, volatility={volatility:.6f}")
            
            # Store current_price as previous_price for next iteration
            self.previous_price = current_price
            
            return processed_signal
            
        except Exception as e:
            print(f"[Pipeline] Processing error: {e}")
            # Return fallback processed signal
            return {
                'event_id': event.get('event_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'features': {'price_change': 0, 'volume_normalized': 0.5, 'price_volatility': 0, 'momentum': 0},
                'prediction': 0.0,
                'confidence': 0.0,
                'volatility': 0.0,
                'momentum': 0.0
            }

    def _calculate_rolling_prediction(self) -> tuple[float, float]:
        """
        Calculate rolling window prediction and momentum.
        
        Returns:
            Tuple of (prediction, momentum)
        """
        if len(self.price_window) < self.prediction_steps:
            # Not enough data for rolling prediction
            return 0.0, 0.0
        
        try:
            # Rolling prediction: (current_price - price_n_steps_ago) / price_n_steps_ago
            current_price = self.price_window[-1]
            historical_price = self.price_window[-self.prediction_steps]
            
            prediction = (current_price - historical_price) / historical_price
            
            # Momentum: trend direction based on recent price changes
            if len(self.price_change_window) >= 3:
                recent_changes = self.price_change_window[-3:]
                momentum = sum(recent_changes) / len(recent_changes)
            else:
                momentum = 0.0
            
            print(f"[Pipeline Debug] Rolling calculation: current={current_price:.2f}, "
                  f"historical_{self.prediction_steps}={historical_price:.2f}, "
                  f"prediction={prediction:.6f}, momentum={momentum:.6f}")
            
            return prediction, momentum
            
        except Exception as e:
            print(f"[Pipeline] Rolling prediction error: {e}")
            return 0.0, 0.0
    
    def _calculate_volatility(self) -> float:
        """
        Calculate volatility as standard deviation of recent price changes.
        
        Returns:
            Volatility value (standard deviation)
        """
        if len(self.price_change_window) < 2:
            return 0.0
        
        try:
            # Use all available changes in window for volatility calculation
            volatility = statistics.stdev(self.price_change_window) if len(self.price_change_window) > 1 else 0.0
            
            print(f"[Pipeline Debug] Volatility calculation: std_dev of {len(self.price_change_window)} changes = {volatility:.6f}")
            
            return volatility
            
        except Exception as e:
            print(f"[Pipeline] Volatility calculation error: {e}")
            return 0.0


class AgentAdapter:
    """Adapter for AI Agent Framework."""
    
    def __init__(self, volatility_multiplier: float = 1.5):
        """Initialize the agent adapter.
        
        Args:
            volatility_multiplier: Multiplier 'k' for adaptive volatility floor
        """
        try:
            # Initialize with real agent config
            self.agent_config = AgentConfig()
            self.agent = Agent(self.agent_config)
            print("[Agent] Initialized with real agent components")
        except Exception as e:
            print(f"[Agent] Initialization error: {e}")
            # Fallback to minimal initialization
            self.agent = None
        
        # Adaptive volatility configuration
        self.volatility_multiplier = volatility_multiplier
        
        print(f"[Agent] Adaptive volatility floor configured: k={volatility_multiplier}")
    
    def decide(self, signal: Dict[str, Any]) -> str:
        """
        Make a decision based on processed signal with adaptive volatility filtering.
        
        Args:
            signal: Dictionary with processed features and predictions
            
        Returns:
            Decision string: BUY, SELL, or HOLD
        """
        try:
            print(f"[Agent] Analyzing signal: {signal.get('event_id', 'unknown')}")
            
            # Extract signal data
            prediction = signal.get('prediction', 0.0)
            confidence = signal.get('confidence', 0.0)
            volatility = signal.get('volatility', 0.0)
            momentum = signal.get('momentum', 0.0)
            features = signal.get('features', {})
            
            print(f"[Agent Debug] Raw signal: prediction={prediction:.6f}, confidence={confidence:.6f}, volatility={volatility:.6f}, momentum={momentum:.6f}")
            
            # Use real agent decision logic if available
            if self.agent:
                # Try to use real agent methods
                try:
                    # Create context for agent
                    context = {
                        'current_signal': signal,
                        'features': features,
                        'prediction': prediction,
                        'confidence': confidence,
                        'volatility': volatility,
                        'momentum': momentum,
                        'timestamp': signal.get('timestamp'),
                        'system_state': 'operational'
                    }
                    
                    # Use agent's reasoning (simplified call)
                    # In a full implementation, this would call agent.process(context)
                    # For now, we'll use the agent's decision logic pattern
                    decision = self._adaptive_decision_logic(prediction, confidence, volatility, momentum)
                    
                except Exception as agent_error:
                    print(f"[Agent] Agent method error: {agent_error}, using fallback logic")
                    decision = self._adaptive_decision_logic(prediction, confidence, volatility, momentum)
            else:
                # Fallback decision logic
                decision = self._adaptive_decision_logic(prediction, confidence, volatility, momentum)
            
            print(f"[Agent] Decision: {decision} (confidence: {confidence:.3f}, volatility: {volatility:.6f})")
            return decision
            
        except Exception as e:
            print(f"[Agent] Decision error: {e}")
            return "HOLD"
    
    def _adaptive_decision_logic(self, prediction: float, confidence: float, volatility: float, momentum: float) -> str:
        """
        Enhanced decision logic with adaptive volatility floor and momentum consideration.
        
        Args:
            prediction: Signal prediction value
            confidence: Signal confidence score
            volatility: Current market volatility
            momentum: Price momentum indicator
            
        Returns:
            Decision string
        """
        # Log input values for audit
        print(f"[Agent Debug] Input: prediction={prediction:.6f}, confidence={confidence:.6f}, volatility={volatility:.6f}, momentum={momentum:.6f}")
        
        # Decision thresholds calibrated for rolling window predictions
        BUY_THRESHOLD = 0.002  # More conservative for trend detection
        SELL_THRESHOLD = -0.002
        CONFIDENCE_THRESHOLD = 0.05  # Higher confidence threshold
        
        # Adaptive volatility floor: abs(prediction) < k * volatility → HOLD
        adaptive_floor = self.volatility_multiplier * volatility
        
        print(f"[Agent Debug] Thresholds: BUY > {BUY_THRESHOLD}, SELL < {SELL_THRESHOLD}, confidence > {CONFIDENCE_THRESHOLD}")
        print(f"[Agent Debug] Adaptive volatility floor: k={self.volatility_multiplier}, floor={adaptive_floor:.6f}")
        
        # Apply adaptive volatility floor first
        if abs(prediction) < adaptive_floor:
            print(f"[Agent Debug] HOLD: prediction {prediction:.6f} below adaptive floor {adaptive_floor:.6f}")
            decision = "HOLD"
        elif prediction > BUY_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
            # Enhanced BUY condition with momentum confirmation
            momentum_confirm = momentum >= 0  # Positive momentum supports BUY
            if momentum_confirm:
                print(f"[Agent Debug] BUY condition met: {prediction:.6f} > {BUY_THRESHOLD} AND {confidence:.6f} > {CONFIDENCE_THRESHOLD} AND momentum {momentum:.6f} >= 0")
                decision = "BUY"
            else:
                print(f"[Agent Debug] HOLD: prediction meets threshold but momentum {momentum:.6f} contradicts BUY signal")
                decision = "HOLD"
        elif prediction < SELL_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
            # Enhanced SELL condition with momentum confirmation
            momentum_confirm = momentum <= 0  # Negative momentum supports SELL
            if momentum_confirm:
                print(f"[Agent Debug] SELL condition met: {prediction:.6f} < {SELL_THRESHOLD} AND {confidence:.6f} > {CONFIDENCE_THRESHOLD} AND momentum {momentum:.6f} <= 0")
                decision = "SELL"
            else:
                print(f"[Agent Debug] HOLD: prediction meets threshold but momentum {momentum:.6f} contradicts SELL signal")
                decision = "HOLD"
        else:
            # Detailed HOLD reasoning
            if prediction <= BUY_THRESHOLD and prediction >= SELL_THRESHOLD:
                print(f"[Agent Debug] HOLD: prediction {prediction:.6f} within neutral range [{SELL_THRESHOLD}, {BUY_THRESHOLD}]")
            elif confidence <= CONFIDENCE_THRESHOLD:
                print(f"[Agent Debug] HOLD: confidence {confidence:.6f} below threshold {CONFIDENCE_THRESHOLD}")
            else:
                print(f"[Agent Debug] HOLD: other condition")
            decision = "HOLD"
        
        # Add final debug logging
        print(f"[Agent Debug] Final: prediction={prediction:.6f}, confidence={confidence:.6f}, volatility={volatility:.6f}, momentum={momentum:.6f}, decision={decision}")
        
        return decision


class ResearchAdapter:
    """Adapter for AI Research Platform."""
    
    def __init__(self):
        """Initialize the research adapter."""
        try:
            # Initialize with real experiment runner
            self.experiment_runner = ExperimentRunner()
            self.decisions_log: List[Dict[str, Any]] = []
            
            # Reinforcement learning tracking variables
            self.last_price: Optional[float] = None
            self.last_decision: Optional[str] = None
            self.total_reward: int = 0
            self.reward_history: List[int] = []
            
            print("[Research] Initialized with real research components")
        except Exception as e:
            print(f"[Research] Initialization error: {e}")
            # Fallback to minimal initialization
            self.experiment_runner = None
            self.decisions_log = []
            
            # Reinforcement learning tracking variables (fallback)
            self.last_price: Optional[float] = None
            self.last_decision: Optional[str] = None
            self.total_reward: int = 0
            self.reward_history: List[int] = []
    
    def compute_reward(self, previous_decision: str, previous_price: float, current_price: float) -> int:
        """
        Compute reward based on decision and price movement.
        
        Args:
            previous_decision: The decision made at previous step (BUY, SELL, HOLD)
            previous_price: Price at previous step
            current_price: Current price
            
        Returns:
            Reward value: +1, -1, or 0
        """
        price_change = current_price - previous_price
        
        if previous_decision == "BUY":
            # BUY is correct if price goes up
            return 1 if price_change > 0 else -1
        elif previous_decision == "SELL":
            # SELL is correct if price goes down
            return 1 if price_change < 0 else -1
        else:  # HOLD
            # HOLD always gets 0 reward
            return 0
    
    def log(self, decision: str, signal: Dict[str, Any]):
        """
        Log a decision and its corresponding signal.
        
        Args:
            decision: The decision made (BUY, SELL, HOLD)
            signal: The processed signal that led to the decision
        """
        try:
            print(f"[Research] Logging decision: {decision}")
            
            # Extract current price from signal
            current_price = signal.get('features', {}).get('price_change', 0) + 100  # Convert back to absolute price
            
            # Reinforcement learning: compute reward for previous decision
            reward = 0
            if self.last_price is not None and self.last_decision is not None:
                reward = self.compute_reward(self.last_decision, self.last_price, current_price)
                self.total_reward += reward
                self.reward_history.append(reward)
                
                # Console output for reward
                if reward > 0:
                    print(f"[Research] Reward: +{reward} ({self.last_decision} was correct)")
                elif reward < 0:
                    print(f"[Research] Reward: {reward} ({self.last_decision} was incorrect)")
                else:
                    print(f"[Research] Reward: {reward} ({self.last_decision} - no change)")
            
            # Store current decision and price for next step
            self.last_price = current_price
            self.last_decision = decision
            
            # Create log entry
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_id': signal.get('event_id', 'unknown'),
                'decision': decision,
                'signal': signal,
                'outcome': self._simulate_outcome(decision, signal)
            }
            
            self.decisions_log.append(log_entry)
            
            # Try to log with real experiment runner
            if self.experiment_runner:
                try:
                    # Create a simple experiment config for logging
                    from ai_research_platform.experiments.experiment_config import ExperimentConfig
                    experiment_config = ExperimentConfig(
                        experiment_name='autonomous_ai_decisions',
                        description='Autonomous AI system decision logging',
                        tags=['autonomous', 'decisions', 'real-time']
                    )
                    
                    # Use run_experiment for logging
                    experiment_result = self.experiment_runner.run_experiment(experiment_config)
                    experiment_id = experiment_result.get('experiment_id', 'unknown')
                    print(f"[Research] Logged to experiment: {experiment_id}")
                except Exception as research_error:
                    print(f"[Research] Experiment logging error: {research_error}")
            
        except Exception as e:
            print(f"[Research] Logging error: {e}")
    
    def _simulate_outcome(self, decision: str, signal: Dict[str, Any]) -> str:
        """
        Simulate decision outcome for evaluation.
        
        Args:
            decision: The decision made
            signal: The signal that led to the decision
            
        Returns:
            Outcome string: success, neutral, or failure
        """
        confidence = signal.get('confidence', 0.0)
        
        # Simulate outcome based on confidence and decision type
        if confidence > 0.5:
            if decision != "HOLD":
                return "success"
            else:
                return "neutral"
        else:
            return "no_action"
    
    def summarize(self) -> Dict[str, Any]:
        """
        Generate summary metrics from logged decisions.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            print("[Research] Generating summary metrics")
            
            if not self.decisions_log:
                return {
                    'total_decisions': 0,
                    'correct_decisions': 0,
                    'incorrect_decisions': 0,
                    'neutral_decisions': 0,
                    'accuracy': 0.0,
                    'decision_distribution': {},
                    'total_reward': 0,
                    'average_reward': 0.0,
                    'num_steps': 0
                }
            
            # Calculate metrics based on actual rewards (correctness)
            total_decisions = len(self.decisions_log)
            
            # Count correct decisions based on positive rewards
            correct_decisions = sum(1 for reward in self.reward_history if reward > 0)
            incorrect_decisions = sum(1 for reward in self.reward_history if reward < 0)
            neutral_decisions = sum(1 for reward in self.reward_history if reward == 0)
            
            # Calculate accuracy: correct decisions / total decisions with rewards
            decisions_with_rewards = len(self.reward_history)
            accuracy = correct_decisions / decisions_with_rewards if decisions_with_rewards > 0 else 0.0
            
            # Decision distribution
            decision_counts = {}
            for log in self.decisions_log:
                decision = log['decision']
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # Average confidence
            avg_confidence = sum(log['signal'].get('confidence', 0.0) 
                               for log in self.decisions_log) / total_decisions
            
            # Reinforcement learning metrics
            num_steps = len(self.reward_history)
            average_reward = sum(self.reward_history) / num_steps if num_steps > 0 else 0.0
            
            summary = {
                'total_decisions': total_decisions,
                'correct_decisions': correct_decisions,
                'incorrect_decisions': incorrect_decisions,
                'neutral_decisions': neutral_decisions,
                'accuracy': accuracy,
                'average_confidence': avg_confidence,
                'decision_distribution': decision_counts,
                'success_rate': (correct_decisions / decisions_with_rewards * 100) if decisions_with_rewards > 0 else 0,
                # Reinforcement learning metrics
                'total_reward': self.total_reward,
                'average_reward': average_reward,
                'num_steps': num_steps
            }
            
            print(f"[Research] Summary: {summary['total_decisions']} decisions, "
                  f"Accuracy: {summary['accuracy']:.1%} "
                  f"({summary['correct_decisions']}/{decisions_with_rewards} correct)")
            
            return summary
            
        except Exception as e:
            print(f"[Research] Summary error: {e}")
            return {
                'total_decisions': 0,
                'correct_decisions': 0,
                'incorrect_decisions': 0,
                'neutral_decisions': 0,
                'accuracy': 0.0,
                'decision_distribution': {},
                'total_reward': 0,
                'average_reward': 0.0,
                'num_steps': 0
            }
