"""
ENGINE TWO: BIG-RUNNER WEALTH ENGINE
Position trading engine for 40-80%+ returns

Strategy:
- Multi-week resistance breakouts with retest
- Volatility contraction breakouts (Bollinger squeeze)
- High-volume trend continuation pullbacks

Exit: Partial profits + trailing stops
Risk: 0.5%-0.75% per trade
Threshold: ≥65% AI probability

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SetupType(Enum):
    """Types of setups for Engine 2"""
    BREAKOUT_RETEST = "Breakout + Retest"
    VOLATILITY_CONTRACTION = "Volatility Contraction Breakout"
    TREND_CONTINUATION = "Trend Continuation Pullback"


class TrailingMethod(Enum):
    """Trailing stop methods"""
    EMA20 = "EMA20"
    STRUCTURE = "STRUCTURE"
    WEEKLY = "WEEKLY"


@dataclass
class BigRunnerSetup:
    """Trade setup for Engine 2"""
    setup_type: SetupType
    direction: str
    entry_price: float
    stop_loss: float
    partial_exit_target: float  # 1.5R for 40-60% exit
    partial_exit_percentage: float  # 40-60%
    trailing_method: TrailingMethod
    current_trailing_stop: Optional[float]
    risk_amount: float
    reward_potential: float
    risk_reward_ratio: float
    confidence_factors: Dict[str, bool]
    breakout_strength: float  # 0-100
    volume_strength: float  # 0-100
    trend_quality: float  #0-100


class EngineTwoBigRunner:
    """
    Big-runner wealth engine for position trades
    
    Detects:
    1. Multi-week breakouts with retest (4+ week consolidation)
    2. Volatility contraction (Bollinger Band squeeze)
    3. Trend continuation pullbacks (pullback to EMA50 in uptrend)
    
    Entry:
    - Strong retest confirmation
    - Volume spike
    - Bullish close above key level
    
    Exit Strategy:
    - Book 40-60% at +1.5R
    - Move stop to breakeven
    - Trail remaining:
      * EMA20 (daily close below)
      * Structure (break of swing low)
      * Weekly close (below EMA50)
    """
    
    def __init__(self):
        """Initialize engine"""
        self.partial_target_r = 1.5
        self.partial_exit_pct = 0.5  # 50% by default
    
    def scan_for_setups(self, data: pd.DataFrame) -> List[BigRunnerSetup]:
        """
        Scan for all Engine 2 setups
        
        Args:
            data: Stock data with indicators
        
        Returns:
            List of detected setups
        """
        setups = []
        
        # Check each setup type
        breakout_setup = self.detect_breakout_retest(data)
        if breakout_setup:
            setups.append(breakout_setup)
        
        vol_setup = self.detect_volatility_contraction(data)
        if vol_setup:
            setups.append(vol_setup)
        
        trend_setup = self.detect_trend_continuation_pullback(data)
        if trend_setup:
            setups.append(trend_setup)
        
        return setups
    
    def detect_breakout_retest(self, data: pd.DataFrame) -> Optional[BigRunnerSetup]:
        """
        Detect multi-week resistance breakout with retest
        
        Bullish setup:
        - 4+ week consolidation (20+ days)
        - Breakout with volume > 3× average
        - Retest of breakout level held
        - Now showing bullish continuation
        """
        if len(data) < 60:  # Need at least 3 months of data
            return None
        
        latest = data.iloc[-1]
        confidence_factors = {}
        
        # Find consolidation range (last 20-40 days, excluding recent 3)
        consolidation_start = -43
        consolidation_end = -3
        consolidation_data = data.iloc[consolidation_start:consolidation_end]
        
        resistance_level = consolidation_data['High'].max()
        support_level = consolidation_data['Low'].min()
        consolidation_range = resistance_level - support_level
        
        # Check if consolidation is meaningful (> 5% range)
        if consolidation_range / support_level < 0.05:
            return None  # Range too tight
        
        # Check for breakout in recent days
        recent_data = data.tail(10)
        breakout_occurred = recent_data['Close'].max() > resistance_level
        confidence_factors['breakout_occurred'] = breakout_occurred
        
        if not breakout_occurred:
            return None
        
        # Find breakout candle
        breakout_idx = recent_data[recent_data['Close'] > resistance_level].index[0]
        breakout_candle = data.loc[breakout_idx]
        
        # Check breakout volume
        avg_volume = data['Volume'].rolling(20).mean().loc[breakout_idx]
        breakout_volume_strong = breakout_candle['Volume'] > avg_volume * 3.0
        confidence_factors['strong_breakout_volume'] = breakout_volume_strong
        
        # Check for retest (price came back near resistance, but held)
        retest_occurred = False
        after_breakout = data.loc[breakout_idx:].tail(5)
        for idx, row in after_breakout.iterrows():
            distance_from_resistance = abs(row['Low'] - resistance_level) / resistance_level
            if distance_from_resistance < 0.02:  # Within 2%
                retest_occurred = True
                if row['Close'] > resistance_level:  # Held above
                    confidence_factors['retest_held'] = True
                break
        
        confidence_factors['retest_occurred'] = retest_occurred
        
        # Current price should be building momentum again
        bullish_momentum = latest['Close'] > latest['EMA20']
        confidence_factors['bullish_momentum'] = bullish_momentum
        
        # Volume on current candle
        current_avg_vol = data['Volume'].rolling(20).mean().iloc[-1]
        volume_confirmed = latest['Volume'] > current_avg_vol
        confidence_factors['volume_confirmed'] = volume_confirmed
        
        # Must have key factors
        if not (breakout_occurred and (retest_occurred or breakout_volume_strong)):
            return None
        
        # Entry and stop
        entry_price = latest['Close']
        
        # Stop below retest low or resistance level
        recent_low = data.tail(5)['Low'].min()
        stop_loss = min(recent_low, resistance_level * 0.99)
        
        # Calculate partial exit target
        risk = entry_price - stop_loss
        partial_target = entry_price + (risk * self.partial_target_r)
        
        # Calculate scores
        breakout_strength = self._score_breakout(breakout_volume_strong, retest_occurred)
        volume_strength = min(100, (breakout_candle['Volume'] / avg_volume) * 25)
        trend_quality = self._score_trend_quality(data)
        
        # Risk/reward
        reward = partial_target - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return BigRunnerSetup(
            setup_type=SetupType.BREAKOUT_RETEST,
            direction='LONG',
            entry_price=entry_price,
            stop_loss=stop_loss,
            partial_exit_target=partial_target,
            partial_exit_percentage=self.partial_exit_pct,
            trailing_method=TrailingMethod.STRUCTURE,
            current_trailing_stop=stop_loss,
            risk_amount=risk,
            reward_potential=reward,
            risk_reward_ratio=rr_ratio,
            confidence_factors=confidence_factors,
            breakout_strength=breakout_strength,
            volume_strength=volume_strength,
            trend_quality=trend_quality
        )
    
    def detect_volatility_contraction(self, data: pd.DataFrame) -> Optional[BigRunnerSetup]:
        """
        Detect volatility contraction breakout (Bollinger squeeze)
        
        Bullish setup:
        - Bollinger Bands squeezed (ATR < 50% of 30-day avg)
        - Breakout above upper band
        - Volume explosion
        - Strong trend emerging
        """
        if len(data) < 50 or 'ATR' not in data.columns:
            return None
        
        latest = data.iloc[-1]
        confidence_factors = {}
        
        # Check Bollinger Band squeeze
        bb_width = ((latest['BB_Upper'] - latest['BB_Lower']) / latest['BB_Middle']) * 100
        avg_bb_width = ((data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle'] * 100).rolling(30).mean().iloc[-1]
        
        squeeze = bb_width < avg_bb_width * 0.6
        confidence_factors['bb_squeeze'] = squeeze
        
        # Check ATR contraction
        atr = latest['ATR']
        avg_atr = data['ATR'].rolling(30).mean().iloc[-1]
        atr_contracted = atr < avg_atr * 0.5
        confidence_factors['atr_contraction'] = atr_contracted
        
        if not (squeeze or atr_contracted):
            return None
        
        # Check breakout above upper band
        breakout = latest['Close'] > latest['BB_Upper']
        confidence_factors['bb_breakout'] = breakout
        
        if not breakout:
            return None
        
        # Volume confirmation
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_explosion = latest['Volume'] > avg_volume * 2.0
        confidence_factors['volume_explosion'] = volume_explosion
        
        if not volume_explosion:
            return None
        
        # Trend check
        ema_aligned = latest['EMA20'] > latest['EMA50'] > latest['EMA200']
        confidence_factors['ema_alignment'] = ema_aligned
        
        # Entry and stop
        entry_price = latest['Close']
        
        # Stop below BB middle or recent low
        recent_low = data.tail(10)['Low'].min()
        stop_loss = min(recent_low, latest['BB_Middle']) * 0.995
        
        # Partial target
        risk = entry_price - stop_loss
        partial_target = entry_price + (risk * self.partial_target_r)
        
        # Scores
        breakout_strength = 75 if (squeeze and atr_contracted) else 60
        volume_strength = min(100, (latest['Volume'] / avg_volume) * 33)
        trend_quality = self._score_trend_quality(data)
        
        reward = partial_target - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return BigRunnerSetup(
            setup_type=SetupType.VOLATILITY_CONTRACTION,
            direction='LONG',
            entry_price=entry_price,
            stop_loss=stop_loss,
            partial_exit_target=partial_target,
            partial_exit_percentage=self.partial_exit_pct,
            trailing_method=TrailingMethod.EMA20,
            current_trailing_stop=stop_loss,
            risk_amount=risk,
            reward_potential=reward,
            risk_reward_ratio=rr_ratio,
            confidence_factors=confidence_factors,
            breakout_strength=breakout_strength,
            volume_strength=volume_strength,
            trend_quality=trend_quality
        )
    
    def detect_trend_continuation_pullback(self, data: pd.DataFrame) -> Optional[BigRunnerSetup]:
        """
        Detect high-volume trend continuation pullback
        
        Bullish setup:
        - Strong uptrend (EMA20 > EMA50 > EMA200)
        - Pullback to EMA50 (price within 3%)
        - Volume spike on bounce
        - Bullish reversal candle
        """
        if len(data) < 50:
            return None
        
        latest = data.iloc[-1]
        confidence_factors = {}
        
        # Check strong uptrend
        ema_uptrend = (latest['EMA20'] > latest['EMA50'] > latest['EMA200'])
        confidence_factors['uptrend'] = ema_uptrend
        
        if not ema_uptrend:
            return None
        
        # Check pullback to EMA50
        dist_from_ema50 = ((latest['Close'] - latest['EMA50']) / latest['EMA50']) * 100
        near_ema50 = -3 <= dist_from_ema50 <= 5  # Within 3% below or 5% above
        confidence_factors['near_ema50'] = near_ema50
        
        if not near_ema50:
            return None
        
        # Check for bounce (bullish reversal)
        prev = data.iloc[-2]
        bounce = latest['Close'] > prev['High']
        strong_close = (latest['Close'] - latest['Low']) / (latest['High'] - latest['Low']) > 0.7
        confidence_factors['bullish_bounce'] = bounce or strong_close
        
        if not (bounce or strong_close):
            return None
        
        # Volume spike
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_spike = latest['Volume'] > avg_volume * 1.5
        confidence_factors['volume_spike'] = volume_spike
        
        # Recent higher highs
        recent_high = data['High'].tail(20).max()
        making_hh = latest['Close'] > data['Close'].tail(20).iloc[-5]
        confidence_factors['higher_highs'] = making_hh
        
        # Entry and stop
        entry_price = latest['Close']
        
        # Stop below EMA50
        swing_low = data['Low'].tail(10).min()
        stop_loss = min(swing_low, latest['EMA50']) * 0.995
        
        # Partial target
        risk = entry_price - stop_loss
        partial_target = entry_price + (risk * self.partial_target_r)
        
        # Scores
        breakout_strength = 60  # Medium (pullback, not breakout)
        volume_strength = min(100, (latest['Volume'] / avg_volume) * 40)
        trend_quality = self._score_trend_quality(data)
        
        reward = partial_target - entry_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return BigRunnerSetup(
            setup_type=SetupType.TREND_CONTINUATION,
            direction='LONG',
            entry_price=entry_price,
            stop_loss=stop_loss,
            partial_exit_target=partial_target,
            partial_exit_percentage=self.partial_exit_pct,
            trailing_method=TrailingMethod.EMA20,
            current_trailing_stop=stop_loss,
            risk_amount=risk,
            reward_potential=reward,
            risk_reward_ratio=rr_ratio,
            confidence_factors=confidence_factors,
            breakout_strength=breakout_strength,
            volume_strength=volume_strength,
            trend_quality=trend_quality
        )
    
    def get_trailing_stop(self, 
                         data: pd.DataFrame,
                         entry_price: float,
                         method: TrailingMethod) -> float:
        """
        Calculate trailing stop based on method
        
        Args:
            data: Current stock data
            entry_price: Original entry price
            method: Trailing method to use
        
        Returns:
            Trailing stop price
        """
        latest = data.iloc[-1]
        
        if method == TrailingMethod.EMA20:
            # Trail with EMA20
            return latest['EMA20']
        
        elif method == TrailingMethod.STRUCTURE:
            # Trail with swing lows
            swing_low = data['Low'].tail(10).min()
            return swing_low * 0.995
        
        elif method == TrailingMethod.WEEKLY:
            # Trail with weekly EMA50 (need weekly data)
            # For now, use daily EMA50
            return latest['EMA50']
        
        return entry_price  # Fallback to breakeven
    
    def _score_breakout(self, strong_volume: bool, retest: bool) -> float:
        """Score breakout strength 0-100"""
        score = 50  # Base
        if strong_volume:
            score += 30
        if retest:
            score += 20
        return min(100, score)
    
    def _score_trend_quality(self, data: pd.DataFrame) -> float:
        """Score trend quality 0-100"""
        latest = data.iloc[-1]
        score = 0
        
        # EMA alignment
        if latest['EMA20'] > latest['EMA50'] > latest['EMA200']:
            score += 40
        
        # Price above all EMAs
        if latest['Close'] > latest['EMA200']:
            score += 30
        
        # Angle of EMA20 (rising)
        ema20_slope = (latest['EMA20'] - data['EMA20'].iloc[-5]) / data['EMA20'].iloc[-5]
        if ema20_slope > 0:
            score += 30
        
        return min(100, score)
    
    def get_setup_summary(self, setup: BigRunnerSetup) -> str:
        """Generate human-readable setup summary"""
        summary = f"\n{'='*70}\n"
        summary += f"ENGINE 2: BIG-RUNNER - {setup.setup_type.value}\n"
        summary += f"{'='*70}\n\n"
        summary += f"Direction: {setup.direction}\n"
        summary += f"Entry: ₹{setup.entry_price:.2f}\n"
        summary += f"Stop Loss: ₹{setup.stop_loss:.2f}\n"
        summary += f"Risk: ₹{setup.risk_amount:.2f} per share\n\n"
        summary += f"Partial Exit ({setup.partial_exit_percentage*100:.0f}%): ₹{setup.partial_exit_target:.2f} (+1.5R)\n"
        summary += f"Trailing Method: {setup.trailing_method.value}\n"
        summary += f"Risk:Reward = 1:{setup.risk_reward_ratio:.2f}\n\n"
        summary += f"Strength Scores:\n"
        summary += f"  Breakout: {setup.breakout_strength:.0f}/100\n"
        summary += f"  Volume: {setup.volume_strength:.0f}/100\n"
        summary += f"  Trend Quality: {setup.trend_quality:.0f}/100\n\n"
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
    print("Engine Two: Big-Runner Wealth Engine - v1.0")
    print("=" * 70)
    print("\nPosition trading for 40-80%+ returns")
    print("\nSetup Types:")
    print("  1. Breakout + Retest")
    print("  2. Volatility Contraction")
    print("  3. Trend Continuation Pullback")
    print("\nExit Strategy: Partial profits + trailing stops")
    print("Risk: 0.5%-0.75% per trade")
    print("Threshold: ≥65% AI probability")
