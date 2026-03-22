#!/usr/bin/env python3
"""
Adapter Layer for Autonomous AI System Integration

This file provides adapter classes that wrap the three AI systems
to create a unified interface without modifying the original systems.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import statistics
import math
from enum import Enum
from dataclasses import dataclass, field

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


class MarketRegime(Enum):
    """Market regime classification"""
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    TREND = "TREND"

@dataclass
class TradeMetrics:
    """Enhanced trade performance metrics with regime-specific expectancy calculation"""
    # Rolling window tracking (last 50 trades)
    max_tracked_trades: int = 50
    rewards_history: List[float] = field(default_factory=list)
    
    # Legacy global stats (for backward compatibility)
    win_count: int = 0
    loss_count: int = 0
    total_wins_value: float = 0.0
    total_losses_value: float = 0.0
    
    # Regime-specific tracking
    # TREND regime stats
    trend_rewards_history: List[float] = field(default_factory=list)
    trend_win_count: int = 0
    trend_loss_count: int = 0
    trend_total_wins_value: float = 0.0
    trend_total_losses_value: float = 0.0
    
    # RANGE regime stats
    range_rewards_history: List[float] = field(default_factory=list)
    range_win_count: int = 0
    range_loss_count: int = 0
    range_total_wins_value: float = 0.0
    range_total_losses_value: float = 0.0
    
    def update_with_reward(self, reward: float, regime: str = "RANGE"):
        """
        Update metrics with reward from new reward system.
        
        Args:
            reward: Reward value from enhanced reward system (>0 = win, <0 = loss)
            regime: Market regime ("TREND", "RANGE", or "BREAKOUT") for regime-specific tracking
        """
        # Legacy global update (for backward compatibility)
        self.rewards_history.append(reward)
        
        # Maintain rolling window of last 50 trades
        if len(self.rewards_history) > self.max_tracked_trades:
            # Remove oldest reward and update stats
            oldest_reward = self.rewards_history.pop(0)
            if oldest_reward > 0:
                self.win_count -= 1
                self.total_wins_value -= oldest_reward
            else:
                self.loss_count -= 1
                self.total_losses_value -= abs(oldest_reward)
        
        # Add new reward to legacy stats
        if reward > 0:
            self.win_count += 1
            self.total_wins_value += reward
        else:
            self.loss_count += 1
            self.total_losses_value += abs(reward)
        
        # Regime-specific tracking
        if regime == MarketRegime.TREND.value:
            self._update_regime_stats(reward, "trend")
        elif regime == MarketRegime.RANGE.value:
            self._update_regime_stats(reward, "range")
        # Note: BREAKOUT regime treated as RANGE for EV tracking (mean-reversion strategy)
        elif regime == MarketRegime.BREAKOUT.value:
            self._update_regime_stats(reward, "range")
    
    def _update_regime_stats(self, reward: float, regime_type: str):
        """
        Update regime-specific statistics.
        
        Args:
            reward: Reward value from enhanced reward system
            regime_type: Either "trend" or "range"
        """
        if regime_type == "trend":
            # Update TREND regime stats
            self.trend_rewards_history.append(reward)
            
            # Maintain rolling window
            if len(self.trend_rewards_history) > self.max_tracked_trades:
                oldest_reward = self.trend_rewards_history.pop(0)
                if oldest_reward > 0:
                    self.trend_win_count -= 1
                    self.trend_total_wins_value -= oldest_reward
                else:
                    self.trend_loss_count -= 1
                    self.trend_total_losses_value -= abs(oldest_reward)
            
            # Add new reward
            if reward > 0:
                self.trend_win_count += 1
                self.trend_total_wins_value += reward
            else:
                self.trend_loss_count += 1
                self.trend_total_losses_value += abs(reward)
                
        elif regime_type == "range":
            # Update RANGE regime stats
            self.range_rewards_history.append(reward)
            
            # Maintain rolling window
            if len(self.range_rewards_history) > self.max_tracked_trades:
                oldest_reward = self.range_rewards_history.pop(0)
                if oldest_reward > 0:
                    self.range_win_count -= 1
                    self.range_total_wins_value -= oldest_reward
                else:
                    self.range_loss_count -= 1
                    self.range_total_losses_value -= abs(oldest_reward)
            
            # Add new reward
            if reward > 0:
                self.range_win_count += 1
                self.range_total_wins_value += reward
            else:
                self.range_loss_count += 1
                self.range_total_losses_value += abs(reward)
    
    def _compute_ev(self, win_count: int, loss_count: int, total_wins_value: float, total_losses_value: float, regime: str) -> float:
        """
        Calculate proper expected value for a specific regime: EV = (win_rate × avg_win) - (loss_rate × avg_loss)
        
        Args:
            win_count: Number of winning trades for this regime
            loss_count: Number of losing trades for this regime  
            total_wins_value: Total value of winning trades for this regime
            total_losses_value: Total value of losing trades for this regime
            regime: Regime name for logging ("TREND" or "RANGE")
            
        Returns:
            Expected value, or 0 if insufficient data (<10 trades)
        """
        total_trades = win_count + loss_count
        
        # Safeguard: insufficient data
        if total_trades < 10:
            return 0.0
        
        # Safeguard: division by zero
        if win_count == 0 or loss_count == 0:
            return 0.0
        
        # Calculate rates
        win_rate = win_count / total_trades
        loss_rate = loss_count / total_trades
        
        # Calculate averages
        avg_win = total_wins_value / win_count
        avg_loss = total_losses_value / loss_count
        
        # Enhanced EV calculation
        ev = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        # Enhanced logging
        print(f"[EV][{regime}] win_rate={win_rate:.3f}, avg_win={avg_win:.6f}, avg_loss={avg_loss:.6f}, EV={ev:.6f}")
        
        return ev
    
    def expected_value_trend(self) -> float:
        """
        Calculate TREND regime expected value.
        
        Returns:
            Expected value for TREND regime, or 0 if insufficient data (<10 trades)
        """
        return self._compute_ev(
            self.trend_win_count, self.trend_loss_count,
            self.trend_total_wins_value, self.trend_total_losses_value,
            "TREND"
        )
    
    def expected_value_range(self) -> float:
        """
        Calculate RANGE regime expected value.
        
        Returns:
            Expected value for RANGE regime, or 0 if insufficient data (<10 trades)
        """
        return self._compute_ev(
            self.range_win_count, self.range_loss_count,
            self.range_total_wins_value, self.range_total_losses_value,
            "RANGE"
        )
    
    def expected_value(self) -> float:
        """
        Legacy global expected value calculation (for backward compatibility).
        
        Returns:
            Expected value, or 0 if insufficient data (<10 trades)
        """
        total_trades = self.win_count + self.loss_count
        
        # Safeguard: insufficient data
        if total_trades < 10:
            return 0.0
        
        # Safeguard: division by zero
        if self.win_count == 0 or self.loss_count == 0:
            return 0.0
        
        # Calculate rates
        win_rate = self.win_count / total_trades
        loss_rate = self.loss_count / total_trades
        
        # Calculate averages
        avg_win = self.total_wins_value / self.win_count
        avg_loss = self.total_losses_value / self.loss_count
        
        # Enhanced EV calculation
        ev = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        return ev
    
    # Legacy methods for backward compatibility
    @property
    def total_trades(self) -> int:
        return self.win_count + self.loss_count
    
    @property
    def winning_trades(self) -> int:
        return self.win_count
    
    @property
    def losing_trades(self) -> int:
        return self.loss_count
    
    @property
    def win_probability(self) -> float:
        total = self.total_trades
        return self.win_count / total if total > 0 else 0.5

class PositionSize(Enum):
    """Position sizing levels"""
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"

@dataclass
class CooldownTracker:
    """Trade cooldown tracking"""
    last_trade_time: Optional[datetime] = None
    cooldown_window: timedelta = timedelta(seconds=30)
    
    def can_trade(self, current_time: datetime) -> bool:
        """Check if enough time has passed since last trade"""
        if self.last_trade_time is None:
            return True
        return current_time - self.last_trade_time >= self.cooldown_window
    
    def update_last_trade(self, trade_time: datetime):
        """Update last trade timestamp"""
        self.last_trade_time = trade_time
    
    def cooldown_factor(self, current_time: datetime) -> float:
        """Return signal strength reduction factor during cooldown"""
        if self.last_trade_time is None:
            return 1.0
        
        time_since_last = current_time - self.last_trade_time
        if time_since_last >= self.cooldown_window:
            return 1.0
        
        # Linear reduction from 0.1 to 1.0 over cooldown period
        progress = time_since_last / self.cooldown_window
        return 0.1 + 0.9 * progress


class PipelineAdapter:
    """Adapter for Realtime AI Pipeline."""
    
    def __init__(self, window_size: int = 20, prediction_steps: int = 5, 
                 window_size_slow: int = 60, prediction_steps_slow: int = 15):
        """Initialize the pipeline adapter with dual timescale support.
        
        Args:
            window_size: Number of prices to store for fast volatility calculation
            prediction_steps: Number of steps back for fast trend prediction
            window_size_slow: Number of prices to store for slow volatility calculation
            prediction_steps_slow: Number of steps back for slow trend prediction
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
        self.window_size_slow = window_size_slow
        self.prediction_steps_slow = prediction_steps_slow
        
        # Initialize rolling windows and tracking
        self.price_window: List[float] = []
        self.price_change_window: List[float] = []
        self.previous_price: Optional[float] = None
        
        # V3 Regime detection tracking
        self.volatility_history: List[float] = []
        self.trend_strength_history: List[float] = []
        self.current_regime: MarketRegime = MarketRegime.RANGE
        
        print(f"[Pipeline] Dual timescale configured:")
        print(f"  Fast: window={window_size}, steps={prediction_steps}")
        print(f"  Slow: window={window_size_slow}, steps={prediction_steps_slow}")
    
    def process(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single event through the pipeline with dual timescale rolling window analysis.
        
        Args:
            event: Dictionary containing event data (price, volume, timestamp)
            
        Returns:
            Dictionary with processed features and predictions for both fast and slow timescales
        """
        try:
            print(f"[Pipeline] Processing event: {event.get('event_id', 'unknown')}")
            
            # Extract features from event
            current_price = event.get('price', 100)
            volume = event.get('volume', 50)
            
            # Handle first event case
            if self.previous_price is None:
                # Fast signal initialization
                prediction_fast = 0
                momentum_fast = 0
                volatility_fast = 0
                
                # Slow signal initialization
                prediction_slow = 0
                momentum_slow = 0
                volatility_slow = 0
                
                confidence = 0
                price_change = 0
                print(f"[Pipeline Debug] First event - initializing dual timescale windows")
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
                
                # Calculate rolling window predictions for both timescales
                prediction_fast, momentum_fast = self._calculate_rolling_prediction(self.prediction_steps)
                prediction_slow, momentum_slow = self._calculate_rolling_prediction(self.prediction_steps_slow)
                
                # Calculate volatility for both timescales
                volatility_fast = self._calculate_volatility(self.window_size)
                volatility_slow = self._calculate_volatility(self.window_size_slow)
                
                # V3 Regime Detection
                self.current_regime = self._detect_market_regime(volatility_fast, volatility_slow, prediction_fast, momentum_fast)
                
                # Enhanced confidence calculation with proper weighting
                trend_aligned = (prediction_fast * prediction_slow) > 0
                momentum_confirmed = (prediction_fast * momentum_fast) > 0
                
                # Calculate base confidence as ratio of signal to threshold with volatility compression
                volatility_factor = max(volatility_fast, 0.00005)
                
                # Apply compression for low-volatility conditions
                compression = 1.0
                if volatility_fast < 0.00008:
                    compression = 0.5   # more aggressive trading in quiet markets
                elif volatility_fast < 0.00012:
                    compression = 0.75
                
                # Calculate dynamic threshold with compression and regime multiplier
                regime_multiplier = 1.0  # Could be regime-specific in future
                dynamic_threshold = volatility_factor * 1.5 * regime_multiplier * compression
                
                # Apply minimum threshold safety constraint
                dynamic_threshold = max(dynamic_threshold, 0.00005)
                
                # Debug logging for threshold adjustments
                print(f"[Agent] Threshold Adjusted:")
                print(f"  volatility={volatility_fast:.6f}")
                print(f"  compression={compression:.2f}")
                print(f"  final_threshold={dynamic_threshold:.6f}")
                
                base_signal_strength = abs(prediction_fast) / dynamic_threshold
                
                # Apply weighting factors
                alignment_weight = 1.0 if trend_aligned else 0.6
                momentum_weight = 1.0 if momentum_confirmed else 0.7
                
                # Final confidence calculation
                confidence = base_signal_strength * alignment_weight * momentum_weight
                
                # Clamp confidence between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
                
                alignment_str = "ALIGNED" if trend_aligned else "MISALIGNED"
                momentum_str = "CONFIRMED" if momentum_confirmed else "NOT_CONFIRMED"
                
                # Debug logging for dual timescale
                print(f"[Pipeline Debug] V3 Regime Detection: {self.current_regime.value}")
                print(f"[Pipeline Debug] Fast signal: pred={prediction_fast:.6f}, mom={momentum_fast:.6f}, vol={volatility_fast:.6f}")
                print(f"[Pipeline Debug] Slow signal: pred={prediction_slow:.6f}, mom={momentum_slow:.6f}, vol={volatility_slow:.6f}")
                print(f"[Pipeline Debug] Signal alignment: {alignment_str}, momentum: {momentum_str}")
                print(f"[Pipeline Debug] Confidence: base={base_signal_strength:.3f} × align={alignment_weight:.1f} × mom={momentum_weight:.1f} = {confidence:.3f}")
                print(f"[Pipeline Debug] Confidence scaling: pred/threshold={abs(prediction_fast):.6f}/{dynamic_threshold:.6f}={base_signal_strength:.3f}")
                print(f"[Pipeline Debug] price_change={price_change:.6f}, price_change_pct={price_change_pct:.6f}")
                print(f"[Pipeline Debug] window_size={len(self.price_window)}, confidence={confidence:.6f}")
                
                if len(self.price_window) >= 5:
                    print(f"[Pipeline Debug] price_window (last 5): {[f'{p:.2f}' for p in self.price_window[-5:]]}")
                    print(f"[Pipeline Debug] change_window (last 5): {[f'{c:.6f}' for c in self.price_change_window[-5:]]}")
            
            # Create processed signal with dual timescale features and V3 regime data
            processed_signal = {
                'event_id': event.get('event_id', f"event_{datetime.now().timestamp()}"),
                'timestamp': event.get('timestamp', datetime.now().isoformat()),
                'features': {
                    'price_change': price_change,
                    'volume_normalized': volume / 100,
                    'window_size': len(self.price_window)
                },
                # Fast signals
                'prediction_fast': prediction_fast,
                'momentum_fast': momentum_fast,
                'volatility_fast': volatility_fast,
                # Slow signals
                'prediction_slow': prediction_slow,
                'momentum_slow': momentum_slow,
                'volatility_slow': volatility_slow,
                # Combined metrics
                'confidence': confidence,
                # V3 Intelligence features
                'regime': self.current_regime.value,
                'trend_strength': abs(prediction_slow),
                'volatility_ratio': volatility_fast / (volatility_slow + 0.0001)
            }
            
            print(f"[Pipeline] Processed dual timescale: fast_pred={processed_signal['prediction_fast']:.6f}, "
                  f"slow_pred={processed_signal['prediction_slow']:.6f}, confidence={confidence:.6f}")
            
            # Store current_price as previous_price for next iteration
            self.previous_price = current_price
            
            return processed_signal
            
        except Exception as e:
            print(f"[Pipeline] Processing error: {e}")
            # Return fallback processed signal with dual timescale structure and V3 features
            return {
                'event_id': event.get('event_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'features': {'price_change': 0, 'volume_normalized': 0.5},
                # Fast signals
                'prediction_fast': 0.0,
                'momentum_fast': 0.0,
                'volatility_fast': 0.0,
                # Slow signals
                'prediction_slow': 0.0,
                'momentum_slow': 0.0,
                'volatility_slow': 0.0,
                # Combined metrics
                'confidence': 0.0,
                # V3 Intelligence features
                'regime': MarketRegime.RANGE.value,
                'trend_strength': 0.0,
                'volatility_ratio': 1.0
            }

    def _calculate_rolling_prediction(self, prediction_steps: int) -> tuple[float, float]:
        """
        Calculate rolling window prediction and momentum for specified timescale.
        
        Args:
            prediction_steps: Number of steps back for prediction calculation
            
        Returns:
            Tuple of (prediction, momentum)
        """
        if len(self.price_window) < prediction_steps:
            # Not enough data for rolling prediction
            return 0.0, 0.0
        
        try:
            # Rolling prediction: (current_price - price_n_steps_ago) / price_n_steps_ago
            current_price = self.price_window[-1]
            historical_price = self.price_window[-prediction_steps]
            
            prediction = (current_price - historical_price) / historical_price
            
            # Momentum: trend direction based on recent price changes
            if len(self.price_change_window) >= 3:
                recent_changes = self.price_change_window[-3:]
                momentum = sum(recent_changes) / len(recent_changes)
            else:
                momentum = 0.0
            
            print(f"[Pipeline Debug] Rolling calculation ({prediction_steps} steps): current={current_price:.2f}, "
                  f"historical={historical_price:.2f}, prediction={prediction:.6f}, momentum={momentum:.6f}")
            
            return prediction, momentum
            
        except Exception as e:
            print(f"[Pipeline] Rolling prediction error: {e}")
            return 0.0, 0.0
    
    def _calculate_volatility(self, window_size: int) -> float:
        """
        Calculate volatility as standard deviation of recent price changes for specified window.
        
        Args:
            window_size: Size of window to use for volatility calculation
            
        Returns:
            Volatility value (standard deviation)
        """
        if len(self.price_change_window) < 2:
            return 0.0
        
        try:
            # Use specified window size for volatility calculation
            changes_for_volatility = self.price_change_window[-window_size:] if len(self.price_change_window) >= window_size else self.price_change_window
            volatility = statistics.stdev(changes_for_volatility) if len(changes_for_volatility) > 1 else 0.0
            
            print(f"[Pipeline Debug] Volatility calculation ({window_size} window): std_dev of {len(changes_for_volatility)} changes = {volatility:.6f}")
            
            return volatility
            
        except Exception as e:
            print(f"[Pipeline] Volatility calculation error: {e}")
            return 0.0
    
    def _detect_market_regime(self, volatility_fast: float, volatility_slow: float, 
                            prediction_fast: float, momentum_fast: float) -> MarketRegime:
        """
        V3 Market regime detection based on volatility and trend characteristics.
        
        Args:
            volatility_fast: Current fast timescale volatility
            volatility_slow: Slow timescale volatility (baseline)
            prediction_fast: Fast timescale prediction (trend direction)
            momentum_fast: Fast timescale momentum
            
        Returns:
            MarketRegime enum value
        """
        try:
            # Update volatility history for trend analysis
            self.volatility_history.append(volatility_fast)
            if len(self.volatility_history) > 20:  # Keep last 20 measurements
                self.volatility_history.pop(0)
            
            # Update trend strength history
            trend_strength = abs(prediction_fast)
            self.trend_strength_history.append(trend_strength)
            if len(self.trend_strength_history) > 20:
                self.trend_strength_history.pop(0)
            
            # Regime classification logic
            if len(self.volatility_history) < 5:
                return MarketRegime.RANGE  # Default until enough data
            
            # Calculate volatility trend (rising/falling/stable)
            recent_vol = sum(self.volatility_history[-5:]) / 5
            older_vol = sum(self.volatility_history[-10:-5]) / 5 if len(self.volatility_history) >= 10 else recent_vol
            volatility_trend = (recent_vol - older_vol) / (older_vol + 0.0001)
            
            # Calculate average trend strength
            avg_trend_strength = sum(self.trend_strength_history[-5:]) / 5
            
            # Regime thresholds
            volatility_low_threshold = volatility_slow * 0.8  # Low volatility = < 80% of baseline
            volatility_rising_threshold = 0.2  # Volatility rising > 20%
            trend_strong_threshold = 0.003  # Strong trend > 0.3%
            
            print(f"[Pipeline Debug] Regime Analysis:")
            print(f"  Volatility: fast={volatility_fast:.6f}, slow={volatility_slow:.6f}, trend={volatility_trend:.3f}")
            print(f"  Trend strength: avg={avg_trend_strength:.6f}, current={trend_strength:.6f}")
            print(f"  Thresholds: vol_low={volatility_low_threshold:.6f}, vol_rising={volatility_rising_threshold:.3f}, trend_strong={trend_strong_threshold:.6f}")
            
            # Classification logic
            if volatility_fast < volatility_low_threshold:
                regime = MarketRegime.RANGE
                reason = "Low volatility - range bound"
            elif volatility_trend > volatility_rising_threshold:
                regime = MarketRegime.BREAKOUT
                reason = "Volatility rising - breakout potential"
            elif avg_trend_strength > trend_strong_threshold and abs(momentum_fast) > volatility_fast * 0.5:
                regime = MarketRegime.TREND
                reason = "Strong trend with momentum"
            else:
                regime = MarketRegime.RANGE
                reason = "Default to range - unclear signals"
            
            print(f"[Pipeline Debug] Regime Classification: {regime.value} ({reason})")
            return regime
            
        except Exception as e:
            print(f"[Pipeline] Regime detection error: {e}")
            return MarketRegime.RANGE


class AgentAdapter:
    """Adapter for AI Agent Framework."""
    
    def __init__(self, volatility_multiplier: float = 1.5):
        """Initialize the agent adapter with V3 intelligence features.
        
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
        
        # V3 Intelligence components
        self.trade_metrics = TradeMetrics()
        self.cooldown_tracker = CooldownTracker()
        self.last_signal_prediction: Optional[float] = None
        self.last_signal_time: Optional[datetime] = None
        
        # Trade gating for overtrading prevention
        self.recent_signals: List[bool] = []  # Track recent signal accuracy (True=correct, False=wrong)
        self.max_recent_signals = 10  # Track last 10 signals
        self.consecutive_losses = 0
        self.threshold_multiplier_boost = 0.0  # Temporary threshold increase
        
        # Warm-up risk limiter tracking
        self.recent_trade_results: List[bool] = []  # Track last 5 trades (True=win, False=loss)
        self.max_recent_trades = 5  # Track last 5 trades for early loss protection
        self.warmup_loss_protection_active = False  # Flag for loss protection mode
        self.warmup_threshold_boost = 0.0  # Additional threshold during loss protection
        
        # Signal frequency calibration
        self.calibration_window = 100  # Track last 100 events for frequency calculation
        self.recent_events: List[bool] = []  # Track if each event resulted in a trade (True=trade, False=hold)
        self.calibration_adjustment_factor = 1.0  # Dynamic threshold multiplier for calibration
        
        # Target frequencies by regime
        self.target_frequency_range = {
            MarketRegime.RANGE.value: (0.08, 0.15),  # 8-15% for RANGE
            MarketRegime.TREND.value: (0.15, 0.25),  # 15-25% for TREND
            MarketRegime.BREAKOUT.value: (0.15, 0.25)  # 15-25% for BREAKOUT (same as TREND)
        }
        
        # Learning parameters (adaptive)
        self.k_threshold = volatility_multiplier  # Adaptive threshold multiplier
        self.momentum_weight = 0.5  # Momentum signal weight
        self.volatility_floor = 0.0001  # Minimum volatility floor
        
        print(f"[Agent] V3 Intelligence initialized:")
        print(f"  Adaptive volatility floor: k={volatility_multiplier}")
        print(f"  Trade metrics: EV={self.trade_metrics.expected_value():.6f}")
        print(f"  Cooldown window: {self.cooldown_tracker.cooldown_window.total_seconds()}s")
    
    def decide(self, signal: Dict[str, Any]) -> str:
        """
        V3 Enhanced decision making with Expected Value calculation and position sizing.
        
        Args:
            signal: Dictionary with processed features and V3 regime data
            
        Returns:
            Decision string: BUY_SMALL/MEDIUM/LARGE, SELL_SMALL/MEDIUM/LARGE, or HOLD
        """
        try:
            current_time = datetime.now()
            print(f"[Agent] V3 Decision Analysis: {signal.get('event_id', 'unknown')}")
            
            # Extract V3 enhanced signal data
            prediction_fast = signal.get('prediction_fast', 0.0)
            prediction_slow = signal.get('prediction_slow', 0.0)
            confidence = signal.get('confidence', 0.0)
            volatility_fast = signal.get('volatility_fast', 0.0)
            volatility_slow = signal.get('volatility_slow', 0.0)
            momentum_fast = signal.get('momentum_fast', 0.0)
            momentum_slow = signal.get('momentum_slow', 0.0)
            
            # V3 Regime data
            regime = signal.get('regime', MarketRegime.RANGE.value)
            trend_strength = signal.get('trend_strength', 0.0)
            volatility_ratio = signal.get('volatility_ratio', 1.0)
            
            print(f"[Agent Debug] V3 Signal Input:")
            print(f"  Fast: pred={prediction_fast:.6f}, vol={volatility_fast:.6f}, mom={momentum_fast:.6f}")
            print(f"  Slow: pred={prediction_slow:.6f}, vol={volatility_slow:.6f}, mom={momentum_slow:.6f}")
            print(f"  Regime: {regime}, trend_strength={trend_strength:.6f}, vol_ratio={volatility_ratio:.3f}")
            print(f"  Confidence: {confidence:.6f}")
            
            # Step 1: Regime-Specific Expected Value (EV) Filtering with Warm-up Bypass
            # Calculate both regime-specific EVs for logging
            ev_trend = self.trade_metrics.expected_value_trend()
            ev_range = self.trade_metrics.expected_value_range()
            
            # Determine active EV based on current regime
            if regime == MarketRegime.TREND.value:
                active_ev = ev_trend
                regime_trades = self.trade_metrics.trend_win_count + self.trade_metrics.trend_loss_count
                regime_name = "TREND"
            elif regime == MarketRegime.RANGE.value:
                active_ev = ev_range
                regime_trades = self.trade_metrics.range_win_count + self.trade_metrics.range_loss_count
                regime_name = "RANGE"
            else:  # BREAKOUT treated as RANGE
                active_ev = ev_range
                regime_trades = self.trade_metrics.range_win_count + self.trade_metrics.range_loss_count
                regime_name = "RANGE (BREAKOUT)"
            
            min_trades_for_ev = 10
            is_warmup_phase = regime_trades < min_trades_for_ev
            
            # Log both EVs every cycle for visibility
            print(f"[EV] TREND: trades={self.trade_metrics.trend_win_count + self.trade_metrics.trend_loss_count} EV={ev_trend:.6f}")
            print(f"[EV] RANGE: trades={self.trade_metrics.range_win_count + self.trade_metrics.range_loss_count} EV={ev_range:.6f}")
            
            if is_warmup_phase:
                # Warm-up period: BYPASS EV filtering completely for this regime
                print(f"[EV] {regime_name} BYPASSED: insufficient trades ({regime_trades}/{min_trades_for_ev})")
                print(f"[Agent] Warm-up mode: allowing trades based on signal logic only")
                print(f"[Agent] WARMUP MODE: size forced to SMALL")
            else:
                # Normal operation: Apply regime-specific EV filtering
                if active_ev < 0:
                    print(f"[EV] {regime_name} BLOCKED: EV={active_ev:.6f}")
                    return "HOLD"
                print(f"[EV] {regime_name} ACTIVE: EV={active_ev:.6f}, decision gated")
            
            # Step 2: Warm-up Early Loss Protection
            self._check_warmup_loss_protection()
            
            # Step 2.5: RANGE Regime Override (BEFORE alignment checks)
            if regime == MarketRegime.RANGE.value and not is_warmup_phase:
                # Calculate dynamic threshold with volatility compression (needed for override logic)
                threshold_multiplier = 1.5  # Same as used in hybrid logic
                
                # Apply volatility-aware compression
                volatility_factor = max(volatility_fast, 0.00005)
                compression = 1.0
                if volatility_fast < 0.00008:
                    compression = 0.5   # more aggressive trading in quiet markets
                elif volatility_fast < 0.00012:
                    compression = 0.75
                
                regime_multiplier = 1.0
                dynamic_threshold = volatility_factor * threshold_multiplier * regime_multiplier * compression
                dynamic_threshold = max(dynamic_threshold, 0.00005)
                
                # Check for strong counter-trend conditions
                signals_opposite = (prediction_fast * prediction_slow) < 0  # Fast and slow opposite signs
                signal_strong = abs(prediction_fast) > (dynamic_threshold * 1.5)  # Strong signal requirement
                momentum_confirms = (prediction_fast * momentum_fast) > 0  # Momentum confirms fast direction
                confidence_high = confidence > 0.7  # High confidence requirement
                
                print(f"[Agent Debug] RANGE OVERRIDE CHECK:")
                print(f"  Signals opposite: {signals_opposite}")
                print(f"  Signal strong: {signal_strong} (pred={abs(prediction_fast):.6f} > {dynamic_threshold * 1.5:.6f})")
                print(f"  Momentum confirms: {momentum_confirms}")
                print(f"  Confidence high: {confidence_high}")
                
                if signals_opposite and signal_strong and momentum_confirms and confidence_high:
                    # Strong counter-trend with momentum confirmation - ALLOW TRADE
                    if prediction_fast > 0:
                        print(f"[Agent] RANGE OVERRIDE: strong counter-trend BUY accepted (pred={prediction_fast:.6f}, mom={momentum_fast:.6f})")
                        return "BUY_SMALL"
                    else:
                        print(f"[Agent] RANGE OVERRIDE: strong counter-trend SELL accepted (pred={prediction_fast:.6f}, mom={momentum_fast:.6f})")
                        return "SELL_SMALL"
                else:
                    # Log why override was blocked
                    if not signals_opposite:
                        print(f"[Agent] RANGE OVERRIDE: blocked (reason=signals_aligned)")
                    elif not signal_strong:
                        print(f"[Agent] RANGE OVERRIDE: blocked (reason=weak_signal: {abs(prediction_fast):.6f} <= {dynamic_threshold * 1.5:.6f})")
                    elif not momentum_confirms:
                        print(f"[Agent] RANGE OVERRIDE: blocked (reason=no_momentum: pred={prediction_fast:.6f}, mom={momentum_fast:.6f})")
                    else:
                        print(f"[Agent] RANGE OVERRIDE: blocked (reason=low_confidence: {confidence:.3f})")
            
            # Step 3: Trade Gating for Overtrading Prevention
            if self.consecutive_losses >= 3:
                self.threshold_multiplier_boost = 0.5  # Increase threshold temporarily
                print(f"[Agent] Trade Gating Active: {self.consecutive_losses} consecutive losses, increasing threshold by {self.threshold_multiplier_boost}")
            elif self.consecutive_losses == 0:
                self.threshold_multiplier_boost = 0.0  # Reset boost when recovering
                print(f"[Agent] Trade Gating Reset: No consecutive losses")
            
            # Step 4: Cooldown System
            cooldown_factor = self.cooldown_tracker.cooldown_factor(current_time)
            if cooldown_factor < 1.0:
                print(f"[Agent] Cooldown Active: factor={cooldown_factor:.2f}")
            
            # Step 5: Regime-Adjusted Decision Logic (with adjusted threshold)
            base_decision = self._v3_regime_aware_decision(
                prediction_fast, prediction_slow, confidence, 
                volatility_fast, momentum_fast, momentum_slow,
                regime, trend_strength, volatility_ratio,
                threshold_boost=self.threshold_multiplier_boost + self.warmup_threshold_boost
            )
            
            if base_decision == "HOLD":
                return "HOLD"
            
            # Step 6: Position Sizing based on confidence, regime-specific EV, regime, cooldown, trend, and warm-up rules
            position_size = self._calculate_position_size(
                confidence, active_ev, regime, cooldown_factor, trend_strength, 
                is_warmup_phase  # Pass warm-up phase flag
            )
            
            # Step 7: Apply cooldown to signal strength
            if cooldown_factor < 0.5:  # Strong cooldown - reduce position size
                if position_size == PositionSize.LARGE:
                    position_size = PositionSize.MEDIUM
                elif position_size == PositionSize.MEDIUM:
                    position_size = PositionSize.SMALL
            
            final_decision = f"{base_decision}_{position_size.value}"
            
            # Step 8: Update tracking
            if base_decision in ["BUY", "SELL"]:
                self.cooldown_tracker.update_last_trade(current_time)
                self.last_signal_prediction = prediction_fast
                self.last_signal_time = current_time
            
            print(f"[Agent] V3 Decision: {final_decision} ({regime_name} EV={active_ev:.6f}, cooldown={cooldown_factor:.2f})")
            return final_decision
            
        except Exception as e:
            print(f"[Agent] V3 Decision error: {e}")
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
    
    def _hybrid_decision_logic(self, prediction_fast: float, prediction_slow: float, confidence: float, 
                              volatility_fast: float, momentum_fast: float, momentum_slow: float) -> str:
        """
        Enhanced hybrid decision logic with adaptive thresholds, relaxed volatility floor, and soft decision zone.
        
        Args:
            prediction_fast: Fast timescale prediction value
            prediction_slow: Slow timescale prediction value  
            confidence: Signal confidence score
            volatility_fast: Fast timescale market volatility
            momentum_fast: Fast timescale price momentum indicator
            momentum_slow: Slow timescale price momentum indicator
            
        Returns:
            Decision string: BUY, SELL, WEAK_BUY, WEAK_SELL, or HOLD
        """
        print(f"[Agent Debug] === ENHANCED HYBRID DECISION LOGIC ===")
        
        # Adaptive thresholds based on volatility with compression
        threshold_multiplier = 1.5  # tunable parameter
        
        # Apply volatility-aware compression
        volatility_factor = max(volatility_fast, 0.00005)
        compression = 1.0
        if volatility_fast < 0.00008:
            compression = 0.5   # more aggressive trading in quiet markets
        elif volatility_fast < 0.00012:
            compression = 0.75
        
        regime_multiplier = 1.0
        dynamic_threshold = volatility_factor * threshold_multiplier * regime_multiplier * compression
        dynamic_threshold = max(dynamic_threshold, 0.00005)
        
        # REMOVED: Momentum weighting boost - now using momentum as confirmation filter only
        momentum_weight = 0.5
        adjusted_prediction_fast = prediction_fast  # No momentum boost
        
        # Momentum confirmation filter
        momentum_confirmed = (prediction_fast * momentum_fast) > 0  # Same sign = confirmed
        
        # Enhanced trend alignment logic with weak disagreement handling
        pred_diff = abs(prediction_fast - prediction_slow)
        weak_disagreement_threshold = volatility_fast * 0.5  # Allow small differences
        strong_disagreement_threshold = volatility_fast * 1.5  # Hold for large differences
        
        if (prediction_fast * prediction_slow) > 0:
            trend_aligned = True
            alignment_strength = "STRONG"
        elif pred_diff < weak_disagreement_threshold:
            trend_aligned = True  # Allow weak disagreement
            alignment_strength = "WEAK"
        elif pred_diff > strong_disagreement_threshold:
            trend_aligned = False  # Strong disagreement - hold
            alignment_strength = "HOLD"
        else:
            trend_aligned = True  # Moderate disagreement - allow with reduced confidence
            alignment_strength = "MODERATE"
            
        trend_alignment_str = f"{alignment_strength} (diff={pred_diff:.6f})"
        
        # Minimum signal strength requirement
        min_signal = 0.75 * volatility_fast
        below_min_signal = abs(prediction_fast) < min_signal
        
        # Check for minimum signal bypass with exceptional conditions
        bypass_min_signal = trend_aligned and momentum_confirmed and confidence > 0.9
        
        print(f"[Agent Debug] Signal Strength:")
        print(f"  Minimum signal: {min_signal:.6f} (0.75 * vol={volatility_fast:.6f})")
        print(f"  Prediction: {prediction_fast:.6f}, abs={abs(prediction_fast):.6f}")
        print(f"  Below minimum: {below_min_signal}")
        print(f"  Bypass allowed: {bypass_min_signal}")
        
        print(f"[Agent Debug] Enhanced Parameters:")
        print(f"  Dynamic threshold: {dynamic_threshold:.6f} (k={threshold_multiplier} * vol={volatility_fast:.6f})")
        print(f"  Adjusted prediction: {prediction_fast:.6f} (no momentum boost)")
        print(f"  Minimum signal: {min_signal:.6f}")
        print(f"  Trend alignment: {trend_alignment_str}")
        print(f"  Momentum confirmed: {momentum_confirmed}")
        print(f"  Bypass minimum signal: {bypass_min_signal}")
        
        # Step 1: Minimum Signal Strength Check
        if below_min_signal and not bypass_min_signal:
            print(f"[Agent Debug] HOLD: Prediction {prediction_fast:.6f} below minimum signal {min_signal:.6f}")
            return "HOLD"
        
        # Step 2: Enhanced Trend Filter Check (alignment-based)
        print(f"[Agent Debug] Step 2 - Checking alignment: {alignment_strength}")
        if alignment_strength == "HOLD":
            print(f"[Agent Debug] HOLD: Strong trend disagreement - fast ({prediction_fast:.6f}) and slow ({prediction_slow:.6f}) signals conflict")
            return "HOLD"
        
        # Step 3: Enhanced Decision Logic with Momentum Confirmation
        print(f"[Agent Debug] Step 3 - Enhanced Decision Logic:")
        print(f"  Dynamic thresholds: BUY > {dynamic_threshold:.6f}, SELL < {-dynamic_threshold:.6f}")
        print(f"  Prediction: {prediction_fast:.6f}")
        print(f"  Fast momentum: {momentum_fast:.6f}, Slow momentum: {momentum_slow:.6f}")
        print(f"  Momentum confirmed: {momentum_confirmed}")
        
        decision = "HOLD"
        decision_reason = ""
        
        # Strong signals - exceed dynamic thresholds
        if prediction_fast > dynamic_threshold:
            if prediction_slow > 0 and momentum_confirmed:
                decision = "BUY"
                decision_reason = f"STRONG BUY: pred={prediction_fast:.6f} > {dynamic_threshold:.6f} AND slow_pred>0 AND momentum_confirmed"
            else:
                decision_reason = f"HOLD: Strong positive signal but confirmation failed (slow={prediction_slow:.6f}, momentum_confirmed={momentum_confirmed})"
        elif prediction_fast < -dynamic_threshold:
            if prediction_slow < 0 and momentum_confirmed:
                decision = "SELL"
                decision_reason = f"STRONG SELL: pred={prediction_fast:.6f} < {-dynamic_threshold:.6f} AND slow_pred<0 AND momentum_confirmed"
            else:
                decision_reason = f"HOLD: Strong negative signal but confirmation failed (slow={prediction_slow:.6f}, momentum_confirmed={momentum_confirmed})"
        
        # Soft decision zone - weak signals with trend alignment
        elif abs(prediction_fast) < dynamic_threshold:
            if trend_aligned and momentum_confirmed and alignment_strength in ["WEAK", "MODERATE"]:
                if prediction_fast > 0:
                    decision = "WEAK_BUY"
                    decision_reason = f"WEAK BUY: pred={prediction_fast:.6f} < threshold but {alignment_strength} alignment and momentum confirmed"
                elif prediction_fast < 0:
                    decision = "WEAK_SELL"
                    decision_reason = f"WEAK SELL: pred={prediction_fast:.6f} > -threshold but {alignment_strength} alignment and momentum confirmed"
                else:
                    decision_reason = f"HOLD: Weak signal but prediction is zero"
            else:
                decision_reason = f"HOLD: Weak signal (pred={prediction_fast:.6f}) without sufficient alignment/momentum confirmation"
        else:
            decision_reason = f"HOLD: Signal in neutral zone (pred={prediction_fast:.6f})"
        
        print(f"[Agent Debug] Decision Reason: {decision_reason}")
        print(f"[Agent Debug] Final Decision: {decision}")
        print(f"[Agent Debug] === END ENHANCED LOGIC ===")
        
        return decision
    
    def _v3_regime_aware_decision(self, prediction_fast: float, prediction_slow: float, confidence: float, 
                                 volatility_fast: float, momentum_fast: float, momentum_slow: float,
                                 regime: str, trend_strength: float, volatility_ratio: float,
                                 threshold_boost: float = 0.0) -> str:
        """
        V3 Regime-aware decision logic that adapts behavior based on market conditions.
        
        Args:
            prediction_fast: Fast timescale prediction
            prediction_slow: Slow timescale prediction
            confidence: Signal confidence
            volatility_fast: Fast volatility
            momentum_fast: Fast momentum
            momentum_slow: Slow momentum
            regime: Current market regime (RANGE/BREAKOUT/TREND)
            trend_strength: Trend strength indicator
            volatility_ratio: Fast/slow volatility ratio
            threshold_boost: Additional threshold multiplier for trade gating
            
        Returns:
            Base decision: BUY, SELL, or HOLD
        """
        print(f"[Agent Debug] V3 Regime-Aware Logic: regime={regime}")
        
        # Adaptive thresholds based on regime
        if regime == MarketRegime.RANGE.value:
            # Mean reversion bias in range markets
            threshold_multiplier = 2.0  # Higher threshold for range markets
            momentum_bias = -momentum_fast  # Contrarian approach
            print(f"[Agent Debug] RANGE regime: mean reversion bias")
        elif regime == MarketRegime.TREND.value:
            # Trend following in strong trends
            threshold_multiplier = 1.0  # Lower threshold for trend markets
            momentum_bias = momentum_fast  # Momentum following
            print(f"[Agent Debug] TREND regime: trend following bias")
        elif regime == MarketRegime.BREAKOUT.value:
            # Aggressive entry in breakout markets
            threshold_multiplier = 0.5  # Lowest threshold for breakout
            momentum_bias = momentum_fast * 1.5  # Amplified momentum
            print(f"[Agent Debug] BREAKOUT regime: aggressive entry bias")
        else:
            # Default behavior
            threshold_multiplier = 1.5
            momentum_bias = momentum_fast
            print(f"[Agent Debug] DEFAULT regime behavior")
        
        # Apply trade gating threshold boost
        threshold_multiplier += threshold_boost
        if threshold_boost > 0:
            print(f"[Agent Debug] Trade gating boost: +{threshold_boost} applied (new multiplier: {threshold_multiplier})")
        
        # Calculate dynamic threshold with volatility compression
        volatility_factor = max(volatility_fast, 0.00005)
        compression = 1.0
        if volatility_fast < 0.00008:
            compression = 0.5   # more aggressive trading in quiet markets
        elif volatility_fast < 0.00012:
            compression = 0.75
        
        dynamic_threshold = threshold_multiplier * volatility_factor * compression
        dynamic_threshold = max(dynamic_threshold, 0.00005)
        
        # REMOVED: Momentum-adjusted prediction - using raw prediction with momentum confirmation only
        adjusted_prediction = prediction_fast  # No momentum boost
        
        print(f"[Agent Debug] Regime-Adjusted Parameters:")
        print(f"  Threshold multiplier: {threshold_multiplier} (base + {threshold_boost} boost)")
        print(f"  Dynamic threshold: {dynamic_threshold:.6f}")
        print(f"  Prediction: {prediction_fast:.6f} (no momentum boost)")
        
        # Decision logic with regime adjustment and momentum confirmation
        momentum_confirmed = (prediction_fast * momentum_fast) > 0
        
        if prediction_fast > dynamic_threshold and trend_strength > 0.001 and momentum_confirmed:
            return "BUY"
        elif prediction_fast < -dynamic_threshold and trend_strength > 0.001 and momentum_confirmed:
            return "SELL"
        
        # RANGE TREND MODE: Allow controlled trend-following trades in RANGE regime
        if regime == MarketRegime.RANGE.value:
            aligned = (prediction_fast * prediction_slow) > 0
            strong_signal = abs(prediction_fast) > dynamic_threshold * 1.5
            momentum_ok = abs(momentum_fast) > 0
            confidence_ok = confidence > 0.6
            
            print(f"[Agent Debug] RANGE TREND MODE CHECK:")
            print(f"  Aligned: {aligned}")
            print(f"  Strong signal: {strong_signal} (pred={abs(prediction_fast):.6f} > {dynamic_threshold * 1.5:.6f})")
            print(f"  Momentum OK: {momentum_ok}")
            print(f"  Confidence OK: {confidence_ok}")
            
            if aligned and strong_signal and momentum_ok and confidence_ok:
                print(f"[Agent] RANGE TREND MODE: aligned signal accepted")
                if prediction_fast > 0:
                    return "BUY_SMALL"
                else:
                    return "SELL_SMALL"
            else:
                if not aligned:
                    print(f"[Agent] RANGE TREND MODE: blocked (reason=signals_not_aligned)")
                elif not strong_signal:
                    print(f"[Agent] RANGE TREND MODE: blocked (reason=weak_signal: {abs(prediction_fast):.6f} <= {dynamic_threshold * 1.5:.6f})")
                elif not momentum_ok:
                    print(f"[Agent] RANGE TREND MODE: blocked (reason=no_momentum: {momentum_fast:.6f})")
                else:
                    print(f"[Agent] RANGE TREND MODE: blocked (reason=low_confidence: {confidence:.3f})")
        
        return "HOLD"
    
    def _check_warmup_loss_protection(self):
        """
        Check recent trade results for warm-up loss protection.
        If 3+ losses in last 5 trades, increase threshold by 20% and reduce trade frequency.
        """
        total_trades = self.trade_metrics.total_trades
        
        # Only apply loss protection during warm-up phase (first 20 trades)
        if total_trades >= 20:
            # Reset warm-up protection when past warm-up phase
            if self.warmup_loss_protection_active:
                self.warmup_loss_protection_active = False
                self.warmup_threshold_boost = 0.0
                print(f"[Agent] Warm-up loss protection deactivated: past warm-up phase")
            return
        
        # Check if we have enough trades to evaluate
        if len(self.recent_trade_results) < 3:
            return
        
        # Count losses in last 5 trades
        recent_losses = sum(1 for result in self.recent_trade_results if not result)
        
        if recent_losses >= 3 and not self.warmup_loss_protection_active:
            # Activate loss protection
            self.warmup_loss_protection_active = True
            self.warmup_threshold_boost = 0.2  # Increase threshold by 20%
            print(f"[Agent] WARMUP PROTECTION: loss streak detected ({recent_losses}/5 losses), tightening thresholds by +20%")
            
            # Increase cooldown to reduce trade frequency
            self.cooldown_tracker.cooldown_window = timedelta(seconds=60)  # Increase from 30s to 60s
            print(f"[Agent] WARMUP PROTECTION: reducing trade frequency (cooldown: 30s → 60s)")
            
        elif recent_losses < 3 and self.warmup_loss_protection_active:
            # Deactivate loss protection if conditions improve
            self.warmup_loss_protection_active = False
            self.warmup_threshold_boost = 0.0
            self.cooldown_tracker.cooldown_window = timedelta(seconds=30)  # Reset to normal
            print(f"[Agent] WARMUP PROTECTION: loss streak resolved, returning to normal thresholds")
    
    def _calculate_position_size(self, confidence: float, ev: float, regime: str, 
                               cooldown_factor: float, trend_strength: float, is_warmup_phase: bool = False) -> PositionSize:
        """
        V3 Position sizing based on confidence, EV, and market conditions.
        
        Args:
            confidence: Signal confidence (0-1)
            ev: Expected value
            regime: Current market regime
            cooldown_factor: Current cooldown reduction factor
            trend_strength: Trend strength
            is_warmup_phase: Whether we're in warm-up phase (first 20 trades)
            
        Returns:
            PositionSize enum
        """
        # WARM-UP RULE: Force SMALL position size during EV bypass phase
        if is_warmup_phase:
            print(f"[Agent] WARMUP MODE: size forced to SMALL (warm-up phase)")
            return PositionSize.SMALL
        
        # Base size calculation from confidence and EV
        confidence_score = confidence
        ev_score = min(ev * 100, 1.0)  # Scale EV to 0-1 range
        trend_score = min(trend_strength * 100, 1.0)  # Scale trend strength
        
        # Combined score
        combined_score = (confidence_score + ev_score + trend_score) / 3
        
        # Apply cooldown factor
        combined_score *= cooldown_factor
        
        # Regime adjustments
        if regime == MarketRegime.BREAKOUT.value:
            combined_score *= 1.2  # Boost size in breakout
        elif regime == MarketRegime.RANGE.value:
            combined_score *= 0.8  # Reduce size in range
        
        print(f"[Agent Debug] Position Sizing:")
        print(f"  Scores: conf={confidence_score:.3f}, ev={ev_score:.3f}, trend={trend_score:.3f}")
        print(f"  Combined: {combined_score:.3f} (cooldown={cooldown_factor:.2f})")
        
        # Size thresholds
        if combined_score >= 0.7:
            return PositionSize.LARGE
        elif combined_score >= 0.4:
            return PositionSize.MEDIUM
        else:
            return PositionSize.SMALL
    
    def update_trade_metrics(self, decision: str, entry_price: float, exit_price: float):
        """
        V3 Learning: Update trade metrics with actual trade results.
        
        Args:
            decision: Trade decision (BUY/SELL)
            entry_price: Entry price
            exit_price: Exit price
        """
        try:
            pnl = (exit_price - entry_price) / entry_price
            
            # Determine if trade was profitable
            if decision.startswith("BUY"):
                won = pnl > 0
            elif decision.startswith("SELL"):
                won = pnl < 0
            else:
                return  # HOLD decisions don't update metrics
            
            # Update metrics
            self.trade_metrics.update(won, abs(pnl))
            
            # Update trade gating: track consecutive losses
            if won:
                self.consecutive_losses = 0  # Reset on win
                print(f"[Agent] Trade Gating: Win detected, consecutive losses reset to 0")
            else:
                self.consecutive_losses += 1  # Increment on loss
                print(f"[Agent] Trade Gating: Loss detected, consecutive losses increased to {self.consecutive_losses}")
            
            # Update warm-up loss protection tracking
            self.recent_trade_results.append(won)
            if len(self.recent_trade_results) > self.max_recent_trades:
                self.recent_trade_results.pop(0)
            
            # Track recent signal accuracy
            self.recent_signals.append(won)
            if len(self.recent_signals) > self.max_recent_signals:
                self.recent_signals.pop(0)
            
            # Adaptive learning: adjust parameters based on performance
            self._adaptive_learning_update(won, pnl)
            
            print(f"[Agent] Trade Metrics Updated:")
            print(f"  Result: {'WIN' if won else 'LOSS'}, PnL: {pnl:.6f}")
            print(f"  New EV: {self.trade_metrics.expected_value():.6f}")
            print(f"  Win Rate: {self.trade_metrics.win_probability:.3f}")
            
        except Exception as e:
            print(f"[Agent] Trade metrics update error: {e}")
    
    def _adaptive_learning_update(self, won: bool, pnl: float):
        """
        V3 Learning feedback loop to adapt system parameters.
        
        Args:
            won: Whether the trade was profitable
            pnl: Profit/loss percentage
        """
        try:
            # Adjust threshold multiplier based on recent performance
            if won and pnl > 0.005:  # Big win
                self.k_threshold = max(0.5, self.k_threshold * 0.95)  # Be more aggressive
            elif not won and pnl < -0.005:  # Big loss
                self.k_threshold = min(3.0, self.k_threshold * 1.05)  # Be more conservative
            
            # Adjust momentum weight based on trend following success
            if abs(pnl) > 0.003:  # Significant move
                if won:
                    self.momentum_weight = min(1.0, self.momentum_weight * 1.02)  # Increase momentum emphasis
                else:
                    self.momentum_weight = max(0.1, self.momentum_weight * 0.98)  # Decrease momentum emphasis
            
            # Adjust volatility floor based on false signals
            if not won and abs(pnl) < 0.001:  # False signal
                self.volatility_floor = min(0.01, self.volatility_floor * 1.01)  # Increase floor
            
            print(f"[Agent] Adaptive Learning Update:")
            print(f"  k_threshold: {self.k_threshold:.3f}")
            print(f"  momentum_weight: {self.momentum_weight:.3f}")
            print(f"  volatility_floor: {self.volatility_floor:.6f}")
            
        except Exception as e:
            print(f"[Agent] Adaptive learning error: {e}")


class ResearchAdapter:
    """Adapter for AI Research Platform with logging fail-safe system."""
    
    def __init__(self):
        """Initialize the research adapter with fail-safe logging."""
        try:
            # Initialize with real experiment runner
            self.experiment_runner = ExperimentRunner()
            self.decisions_log: List[Dict[str, Any]] = []
            
            # Reinforcement learning tracking variables
            self.last_price: Optional[float] = None
            self.last_decision: Optional[str] = None
            self.total_reward: float = 0
            self.reward_history: List[float] = []
            
            # New multi-timescale metrics tracking
            self.trend_alignments: List[bool] = []  # Track when fast & slow signals align
            self.signal_quality_count: Dict[str, int] = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'WEAK_BUY': 0, 'WEAK_SELL': 0}
            self.total_signals: int = 0
            
            # Logging fail-safe parameters
            self.logging_enabled = True
            self.experiment_logging_enabled = True
            
            print("[Research] Initialized with real research components and fail-safe logging")
        except Exception as e:
            print(f"[Research] Initialization error: {e}")
            # Fallback to minimal initialization
            self.experiment_runner = None
            self.decisions_log = []
            
            # Reinforcement learning tracking variables (fallback)
            self.last_price: Optional[float] = None
            self.last_decision: Optional[str] = None
            self.total_reward: float = 0
            self.reward_history: List[float] = []
            
            # New multi-timescale metrics tracking (fallback)
            self.trend_alignments: List[bool] = []  # Track when fast & slow signals align
            self.signal_quality_count: Dict[str, int] = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'WEAK_BUY': 0, 'WEAK_SELL': 0}
            self.total_signals: int = 0
            
            # Logging fail-safe parameters (fallback)
            self.logging_enabled = True
            self.experiment_logging_enabled = False  # Disable experiment logging on error
            print("[Research] Fallback mode: experiment logging disabled")
    
    def compute_reward(self, previous_decision: str, previous_price: float, current_price: float, 
                      prediction: float = 0.0, confidence: float = 1.0) -> float:
        """
        Compute continuous directional reward with graded outcome strength.
        
        Args:
            previous_decision: The decision made at previous step (BUY, SELL, WEAK_BUY, WEAK_SELL, HOLD)
            previous_price: Price at previous step
            current_price: Current price
            prediction: Prediction value (>0 for LONG bias, <0 for SHORT bias)
            confidence: Confidence score for scaling (0.0 to 1.0)
            
        Returns:
            Continuous reward value clipped to [-0.01, +0.01] with outcome-based multipliers
        """
        # Calculate short-term future return
        future_return = (current_price - previous_price) / previous_price if previous_price > 0 else 0.0
        
        # Determine decision direction
        if previous_decision in ["BUY", "WEAK_BUY"]:
            decision_direction = 1  # LONG bias
            decision_type = "TRADE"
        elif previous_decision in ["SELL", "WEAK_SELL"]:
            decision_direction = -1  # SHORT bias
            decision_type = "TRADE"
        else:  # HOLD
            decision_direction = 1 if prediction > 0 else -1  # Use prediction for direction
            decision_type = "HOLD"
        
        # Determine if direction is correct
        direction_correct = (decision_direction * future_return) > 0
        
        # Calculate base reward
        if decision_type == "HOLD":
            # Case A: HOLD decisions - smaller rewards
            base_reward = abs(future_return) * 0.5
        else:
            # Case B: TRADE decisions - full rewards
            base_reward = abs(future_return)
        
        # Apply direction sign
        reward = base_reward if direction_correct else -base_reward
        
        # Case A: HOLD decisions - apply prediction direction check
        if decision_type == "HOLD":
            prediction_correct = (prediction * future_return) > 0
            if not prediction_correct:
                reward = -abs(future_return) * 0.5
        
        # === ENHANCED: 5-TIER OUTCOME SYSTEM ===
        
        # Define thresholds for outcome categorization
        strong_threshold = 0.0005
        weak_threshold = 0.0001
        
        # Categorize outcome based on magnitude
        if abs(future_return) >= strong_threshold:
            outcome = "STRONG"
        elif abs(future_return) >= weak_threshold:
            outcome = "WEAK"
        else:
            outcome = "NEUTRAL"
        
        # Apply outcome multipliers
        if direction_correct:
            # Correct predictions
            if outcome == "STRONG":
                multiplier = 1.5
            elif outcome == "WEAK":
                multiplier = 1.0
            else:  # NEUTRAL
                multiplier = 0.2
        else:
            # Wrong predictions - stronger penalties for strong moves
            if outcome == "STRONG":
                multiplier = 1.5
            elif outcome == "WEAK":
                multiplier = 1.0
            else:  # NEUTRAL
                multiplier = 0.2
        
        # Apply outcome multiplier
        reward *= multiplier
        
        # Apply confidence scaling (preserved from original)
        reward *= confidence
        
        # Clip reward to prevent spikes
        reward = max(min(reward, 0.01), -0.01)
        
        # Enhanced debug logging with outcome information
        direction_str = "correct" if direction_correct else "wrong"
        print(f"[Reward] type={decision_type}, direction={direction_str}, outcome={outcome}, multiplier={multiplier:.1f}, raw={base_reward:.6f}, scaled={reward:.6f}")
        
        return reward
    
    def compute_reward_only(self, decision: str, signal: Dict[str, Any]) -> Optional[float]:
        """
        Compute reward only without any logging - used for minimal logging mode.
        
        Args:
            decision: The decision made (BUY, SELL, HOLD)
            signal: The processed signal that led to decision
            
        Returns:
            Computed reward for EV tracking, or None if computation fails
        """
        try:
            if not self.logging_enabled:
                return None
                
            # Extract current price from signal
            current_price = signal.get('features', {}).get('price_change', 0) + 100  # Convert back to absolute price
            
            # Reinforcement learning: compute reward for previous decision
            reward = 0.0
            if self.last_price is not None and self.last_decision is not None:
                # Use combined prediction for reward calculation
                prediction_fast = signal.get('prediction_fast', 0.0)
                prediction_slow = signal.get('prediction_slow', 0.0)
                combined_prediction = (prediction_fast + prediction_slow) / 2.0
                confidence = signal.get('confidence', 1.0)
                reward = self.compute_reward(self.last_decision, self.last_price, current_price, combined_prediction, confidence)
                self.total_reward += reward
                self.reward_history.append(reward)
            
            # Store current decision and price for next step
            self.last_price = current_price
            self.last_decision = decision
            
            return reward if self.last_price is not None and self.last_decision is not None else None
            
        except Exception as e:
            print(f"[Logging Warning] Reward computation failed: {e}")
            return None
    
    def log(self, decision: str, signal: Dict[str, Any]) -> Optional[float]:
        """
        Log a decision and its corresponding signal with comprehensive fail-safe handling.
        
        Args:
            decision: The decision made (BUY, SELL, HOLD)
            signal: The processed signal that led to decision
            
        Returns:
            Computed reward for EV tracking, or None if no reward computed
        """
        try:
            if not self.logging_enabled:
                print("[Logging Warning] Logging disabled")
                return None
                
            print(f"[Research] Logging decision: {decision}")
            
            # Extract dual timescale signal data
            prediction_fast = signal.get('prediction_fast', 0.0)
            prediction_slow = signal.get('prediction_slow', 0.0)
            
            # Track trend alignment
            trend_aligned = (prediction_fast * prediction_slow) > 0
            self.trend_alignments.append(trend_aligned)
            
            # Track signal quality (BUY+SELL vs total)
            self.signal_quality_count[decision] = self.signal_quality_count.get(decision, 0) + 1
            self.total_signals += 1
            
            # Extract current price from signal
            current_price = signal.get('features', {}).get('price_change', 0) + 100  # Convert back to absolute price
            
            # Reinforcement learning: compute reward for previous decision
            reward = 0.0
            if self.last_price is not None and self.last_decision is not None:
                # Use combined prediction for reward calculation
                combined_prediction = (prediction_fast + prediction_slow) / 2.0
                confidence = signal.get('confidence', 1.0)
                reward = self.compute_reward(self.last_decision, self.last_price, current_price, combined_prediction, confidence)
                self.total_reward += reward
                self.reward_history.append(reward)
            
            # Console output for new metrics
            alignment_str = "ALIGNED" if trend_aligned else "MISALIGNED"
            print(f"[Research] Trend signals: {alignment_str} (fast={prediction_fast:.6f}, slow={prediction_slow:.6f})")
            
            # Store current decision and price for next step
            self.last_price = current_price
            self.last_decision = decision
            
            # Create log entry with enhanced dual timescale data
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_id': signal.get('event_id', 'unknown'),
                'decision': decision,
                'signal': signal,
                'outcome': self._simulate_outcome(decision, signal),
                # New multi-timescale metrics
                'trend_aligned': trend_aligned,
                'prediction_fast': prediction_fast,
                'prediction_slow': prediction_slow,
                'signal_quality': decision if decision in ['BUY', 'SELL'] else 'HOLD'
            }
            
            self.decisions_log.append(log_entry)
            
            # Try to log with real experiment runner (with fail-safe)
            if self.experiment_runner and self.experiment_logging_enabled:
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
                    print(f"[Logging Warning] Experiment logging failed: {research_error}")
                    # Disable experiment logging after failure to prevent repeated errors
                    self.experiment_logging_enabled = False
                    print("[Logging Warning] Experiment logging disabled due to repeated failures")
            
            # Return computed reward for EV tracking
            return reward if self.last_price is not None and self.last_decision is not None else None
            
        except Exception as e:
            print(f"[Logging Warning] Logging error: {e}")
            # Try to at least compute reward for EV tracking
            try:
                return self.compute_reward_only(decision, signal)
            except:
                return None
    
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
            
            # New multi-timescale metrics
            # Trend alignment rate: % of time fast & slow signals agree
            trend_alignment_rate = sum(self.trend_alignments) / len(self.trend_alignments) if self.trend_alignments else 0.0
            
            # Signal quality: % of BUY+SELL+WEAK_BUY+WEAK_SELL vs total signals
            trading_signals = (self.signal_quality_count.get('BUY', 0) + 
                           self.signal_quality_count.get('SELL', 0) +
                           self.signal_quality_count.get('WEAK_BUY', 0) +
                           self.signal_quality_count.get('WEAK_SELL', 0))
            signal_quality_rate = trading_signals / self.total_signals if self.total_signals > 0 else 0.0
            
            # False signal rate (optional): % of HOLD signals when trend was aligned
            aligned_holds = sum(1 for i, log in enumerate(self.decisions_log) 
                              if log['decision'] == 'HOLD' and self.trend_alignments[i])
            false_signal_rate = aligned_holds / len(self.trend_alignments) if self.trend_alignments else 0.0
            
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
                'num_steps': num_steps,
                # New multi-timescale metrics
                'trend_alignment_rate': trend_alignment_rate,
                'signal_quality_rate': signal_quality_rate,
                'false_signal_rate': false_signal_rate,
                'signal_quality_count': self.signal_quality_count,
                'total_signals': self.total_signals
            }
            
            print(f"[Research] Summary: {summary['total_decisions']} decisions, "
                  f"Accuracy: {summary['accuracy']:.1%} "
                  f"({summary['correct_decisions']}/{decisions_with_rewards} correct)")
            print(f"[Research] Multi-timescale metrics:")
            print(f"  Trend Alignment Rate: {summary['trend_alignment_rate']:.1%}")
            print(f"  Signal Quality Rate: {summary['signal_quality_rate']:.1%} (trading signals)")
            print(f"  False Signal Rate: {summary['false_signal_rate']:.1%} (aligned holds)")
            
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
                'num_steps': 0,
                # New multi-timescale metrics (fallback)
                'trend_alignment_rate': 0.0,
                'signal_quality_rate': 0.0,
                'false_signal_rate': 0.0,
                'signal_quality_count': {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'WEAK_BUY': 0, 'WEAK_SELL': 0},
                'total_signals': 0
            }
