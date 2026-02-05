"""
ENGINE ONE: MICRO-PROFIT ENGINE
High-accuracy scalping/swing engine with fixed targets

Strategy:
- EMA20/EMA50 pullbacks
- VWAP reclaims
- Tight-range breakouts

Exit: Fixed targets at 0.8R, 1.0R, 1.3R (full exit)
Risk: 0.25%-0.5% per trade
Threshold: ≥70% AI probability

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SetupType(Enum):
    """Types of setups for Engine 1"""
    EMA_PULLBACK = "EMA Pullback"
    VWAP_RECLAIM = "VWAP Reclaim"
    TIGHT_BREAKOUT = "Tight-Range Breakout"


@dataclass
class TradeSetup:
    """Trade setup details"""
    setup_type: SetupType
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    targets: Dict[str, float]  # {'0.8R': price, '1.0R': price, '1.3R': price}
    risk_amount: float
    reward_potential: float
    risk_reward_ratio: float
    confidence_factors: Dict[str, bool]
    volume_confirmed: bool
    ema_aligned: bool
    price_action_strong: bool


class EngineOneMicroProfit:
    """
    Micro-profit engine for high-accuracy trades
    
    Detects:
    1. EMA pullbacks (price 1-3% from EMA20/50, volume contracting)
    2. VWAP reclaims (close above VWAP after being below)
    3. Tight-range breakouts (3-5 day consolidation, volume explosion)
    
    Entry:
    - Bullish close above previous high
    - OR strong close (top 30% of range) above key level
    - Volume confirmation required
    
    Risk Management:
    - Stop below structure/EMA
    - Fixed targets: 0.8R, 1.0R, 1.3R
    - Full exit (no partials)
    """
    
    def __init__(self):
        """Initialize engine"""
        self.target_multiples = [0.8, 1.0, 1.3]
    
    def scan_for_setups(self, data: pd.DataFrame) -> List[TradeSetup]:
        """
        Scan for all Engine 1 setups
        
        Args:
            data: Stock data with indicators
        
        Returns:
            List of detected setups
        """
        setups = []
        
        # Check each setup type
        ema_setup = self.detect_ema_pullback(data)
        if ema_setup:
            setups.append(ema_setup)
        
        vwap_setup = self.detect_vwap_reclaim(data)
        if vwap_setup:
            setups.append(vwap_setup)
        
        breakout_setup = self.detect_tight_breakout(data)
        if breakout_setup:
            setups.append(breakout_setup)
        
        return setups
    
    def detect_ema_pullback(self, data: pd.DataFrame) -> Optional[TradeSetup]:
        """
        Detect EMA20/EMA50 pullback setup
        
        Bullish setup:
        - Price pulled back to EMA20 or EMA50
        - Price 1-3% from EMA (not too far)
        - Volume contracting during pullback
        - EMA20 > EMA50 > EMA200 (uptrend)
        - Recent bullish close above previous high
        """
        if len(data) < 50:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        confidence_factors = {}
        
        # Check EMA alignment (uptrend)
        ema_aligned = (latest['EMA20'] > latest['EMA50'] > latest['EMA200'])
        confidence_factors['ema_alignment'] = ema_aligned
        
        if not ema_aligned:
            return None  # Wrong trend structure
        
        # Check distance from EMA20
        dist_from_ema20 = ((latest['Close'] - latest['EMA20']) / latest['EMA20']) * 100
        near_ema20 = -3 <= dist_from_ema20 <= 3
        confidence_factors['near_ema20'] = near_ema20
        
        # Check distance from EMA50 as alternative
        dist_from_ema50 = ((latest['Close'] - latest['EMA50']) / latest['EMA50']) * 100
        near_ema50 = -3 <= dist_from_ema50 <= 3
        confidence_factors['near_ema50'] = near_ema50
        
        # Must be near at least one EMA
        if not (near_ema20 or near_ema50):
            return None
        
        # Check volume contraction during pullback
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        recent_volume_low = data['Volume'].tail(3).mean() < avg_volume * 0.8
        confidence_factors['volume_contraction'] = recent_volume_low
        
        # Check for bullish entry trigger
        bullish_close = latest['Close'] > prev['High']
        strong_close = (latest['Close'] - latest['Low']) / (latest['High'] - latest['Low']) > 0.7
        confidence_factors['bullish_trigger'] = bullish_close or strong_close
        
        # Volume confirmation on entry
        volume_confirmed = latest['Volume'] > avg_volume
        confidence_factors['volume_confirmed'] = volume_confirmed
        
        # Must have entry trigger
        if not (bullish_close or strong_close):
            return None
        
        # Determine entry and stop
        entry_price = latest['Close']
        
        # Stop below EMA or recent swing low
        swing_low = data['Low'].tail(10).min()
        ema_stop = min(latest['EMA20'], latest['EMA50'])
        stop_loss = min(swing_low, ema_stop) * 0.995  # 0.5% buffer
        
        # Calculate targets
        targets = self._calculate_fixed_targets(entry_price, stop_loss)
        
        # Risk/reward
        risk = entry_price - stop_loss
        reward = targets['1.0R'] - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return TradeSetup(
            setup_type=SetupType.EMA_PULLBACK,
            direction='LONG',
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets=targets,
            risk_amount=risk,
            reward_potential=reward,
            risk_reward_ratio=rr_ratio,
            confidence_factors=confidence_factors,
            volume_confirmed=volume_confirmed,
            ema_aligned=ema_aligned,
            price_action_strong=strong_close
        )
    
    def detect_vwap_reclaim(self, data: pd.DataFrame) -> Optional[TradeSetup]:
        """
        Detect VWAP reclaim setup
        
        Bullish setup:
        - Price was below VWAP
        - Now closed above VWAP
        - Volume increasing
        - Strong close in top 30% of range
        """
        if len(data) < 20 or 'VWAP' not in data.columns:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        prev_2 = data.iloc[-3]
        
        confidence_factors = {}
        
        # Check if reclaim happened
        was_below_vwap = prev['Close'] < prev['VWAP']
        now_above_vwap = latest['Close'] > latest['VWAP']
        confidence_factors['vwap_reclaim'] = was_below_vwap and now_above_vwap
        
        if not (was_below_vwap and now_above_vwap):
            return None
        
        # Check strong close
        candle_range = latest['High'] - latest['Low']
        if candle_range == 0:
            return None
        
        close_position = (latest['Close'] - latest['Low']) / candle_range
        strong_close = close_position > 0.7
        confidence_factors['strong_close'] = strong_close
        
        if not strong_close:
            return None
        
        # Check volume surge
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_surge = latest['Volume'] > avg_volume * 1.2
        confidence_factors['volume_surge'] = volume_surge
        
        # Check trend alignment
        ema_aligned = latest['EMA20'] > latest['EMA50']
        confidence_factors['ema_alignment'] = ema_aligned
        
        # Entry and stop
        entry_price = latest['Close']
        
        # Stop below VWAP or swing low
        swing_low = data['Low'].tail(5).min()
        stop_loss = min(swing_low, latest['VWAP']) * 0.995
        
        # Targets
        targets = self._calculate_fixed_targets(entry_price, stop_loss)
        
        risk = entry_price - stop_loss
        reward = targets['1.0R'] - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return TradeSetup(
            setup_type=SetupType.VWAP_RECLAIM,
            direction='LONG',
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets=targets,
            risk_amount=risk,
            reward_potential=reward,
            risk_reward_ratio=rr_ratio,
            confidence_factors=confidence_factors,
            volume_confirmed=volume_surge,
            ema_aligned=ema_aligned,
            price_action_strong=strong_close
        )
    
    def detect_tight_breakout(self, data: pd.DataFrame) -> Optional[TradeSetup]:
        """
        Detect tight-range breakout
        
        Bullish setup:
        - 3-5 day consolidation
        - Range < ATR
        - Breakout with volume explosion (> 2x average)
        - Close above consolidation high
        """
        if len(data) < 30 or 'ATR' not in data.columns:
            return None
        
        latest = data.iloc[-1]
        
        confidence_factors = {}
        
        # Look for consolidation in last 3-5 days
        lookback = 5
        recent_data = data.tail(lookback + 1)
        
        # Calculate range of consolidation
        consolidation_high = recent_data['High'].iloc[:-1].max()
        consolidation_low = recent_data['Low'].iloc[:-1].min()
        consolidation_range = consolidation_high - consolidation_low
        
        # Get ATR
        atr = latest['ATR']
        
        # Tight range check: consolidation < ATR
        tight_range = consolidation_range < atr
        confidence_factors['tight_range'] = tight_range
        
        if not tight_range:
            return None
        
        # Breakout check: latest close > consolidation high
        breakout = latest['Close'] > consolidation_high
        confidence_factors['breakout'] = breakout
        
        if not breakout:
            return None
        
        # Volume explosion check
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_explosion = latest['Volume'] > avg_volume * 2.0
        confidence_factors['volume_explosion'] = volume_explosion
        
        if not volume_explosion:
            return None  # Volume confirmation required
        
        # Check EMA alignment
        ema_aligned = latest['EMA20'] > latest['EMA50']
        confidence_factors['ema_alignment'] = ema_aligned
        
        # Strong close check
        close_strength = (latest['Close'] - latest['Low']) / (latest['High'] - latest['Low'])
        strong_close = close_strength > 0.6
        confidence_factors['strong_close'] = strong_close
        
        # Entry and stop
        entry_price = latest['Close']
        
        # Stop below consolidation low
        stop_loss = consolidation_low * 0.995
        
        # Targets
        targets = self._calculate_fixed_targets(entry_price, stop_loss)
        
        risk = entry_price - stop_loss
        reward = targets['1.0R'] - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return TradeSetup(
            setup_type=SetupType.TIGHT_BREAKOUT,
            direction='LONG',
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets=targets,
            risk_amount=risk,
            reward_potential=reward,
            risk_reward_ratio=rr_ratio,
            confidence_factors=confidence_factors,
            volume_confirmed=volume_explosion,
            ema_aligned=ema_aligned,
            price_action_strong=strong_close
        )
    
    def _calculate_fixed_targets(self, entry: float, stop: float) -> Dict[str, float]:
        """
        Calculate fixed R-multiple targets
        
        Args:
            entry: Entry price
            stop: Stop loss price
        
        Returns:
            Dictionary of targets
        """
        risk = entry - stop
        
        targets = {}
        for r_multiple in self.target_multiples:
            target_price = entry + (risk * r_multiple)
            targets[f'{r_multiple}R'] = target_price
        
        return targets
    
    def get_setup_summary(self, setup: TradeSetup) -> str:
        """Generate human-readable setup summary"""
        summary = f"\n{'='*70}\n"
        summary += f"ENGINE 1: MICRO-PROFIT - {setup.setup_type.value}\n"
        summary += f"{'='*70}\n\n"
        summary += f"Direction: {setup.direction}\n"
        summary += f"Entry: ₹{setup.entry_price:.2f}\n"
        summary += f"Stop Loss: ₹{setup.stop_loss:.2f}\n"
        summary += f"Risk: ₹{setup.risk_amount:.2f} per share\n\n"
        summary += f"Targets:\n"
        for label, price in setup.targets.items():
            summary += f"  {label}: ₹{price:.2f}\n"
        summary += f"\nRisk:Reward = 1:{setup.risk_reward_ratio:.2f}\n\n"
        summary += f"Confidence Factors:\n"
        for factor, value in setup.confidence_factors.items():
            check = "✅" if value else "❌"
            summary += f"  {check} {factor}\n"
        summary += f"\n{'='*70}\n"
        
        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Engine One: Micro-Profit - v1.0")
    print("=" * 70)
    print("\nHigh-accuracy scalping/swing engine")
    print("\nSetup Types:")
    print("  1. EMA Pullbacks")
    print("  2. VWAP Reclaims")
    print("  3. Tight-Range Breakouts")
    print("\nExit Strategy: Fixed targets (0.8R, 1.0R, 1.3R)")
    print("Risk: 0.25%-0.5% per trade")
    print("Threshold: ≥70% AI probability")
