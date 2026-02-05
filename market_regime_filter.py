"""
MARKET REGIME FILTER
Mandatory pre-trade filter for both engines

This module ensures trades are only taken in favorable market regimes:
- LONG only when price structure is bullish
- SHORT only when price structure is bearish
- NO TRADE in ambiguous/ranging markets

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum


class MarketDirection(Enum):
    """Market direction classification"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class RegimeResult:
    """Result of regime analysis"""
    direction: MarketDirection
    score: int  # 0-100
    eligible_for_long: bool
    eligible_for_short: bool
    reasons: Dict[str, bool]
    index_aligned: bool
    atr_expanding: bool
    volume_above_average: bool


class MarketRegimeFilter:
    """
    Implements strict regime filtering before any trade consideration
    
    LONG eligibility requires:
    - Price > EMA200
    - EMA50 > EMA200
    - Index (NIFTY/BANKNIFTY) > EMA200
    - ATR expanding (current > 5-day average)
    - Volume > 20-day average
    
    SHORT eligibility requires inverse conditions
    """
    
    def __init__(self):
        """Initialize regime filter"""
        self.ema_periods = {
            'ema20': 20,
            'ema50': 50,
            'ema200': 200
        }
    
    def analyze_regime(self, 
                      stock_data: pd.DataFrame,
                      index_data: Optional[pd.DataFrame] = None) -> RegimeResult:
        """
        Perform complete regime analysis
        
        Args:
            stock_data: DataFrame with OHLCV data and pre-calculated EMAs
            index_data: Optional DataFrame with index OHLCV and EMAs
        
        Returns:
            RegimeResult with eligibility and scoring
        """
        # Ensure we have required columns
        required = ['Close', 'High', 'Low', 'Volume']
        if not all(col in stock_data.columns for col in required):
            raise ValueError(f"Stock data must contain: {required}")
        
        # Calculate EMAs if not present
        stock_data = self._ensure_indicators(stock_data)
        
        # Check long conditions
        long_eligible, long_reasons = self._check_long_eligibility(
            stock_data, index_data
        )
        
        # Check short conditions
        short_eligible, short_reasons = self._check_short_eligibility(
            stock_data, index_data
        )
        
        # Calculate regime score
        score = self._calculate_regime_score(stock_data, long_reasons, short_reasons)
        
        # Determine direction
        if long_eligible:
            direction = MarketDirection.BULLISH
        elif short_eligible:
            direction = MarketDirection.BEARISH
        else:
            direction = MarketDirection.NEUTRAL
        
        # Extract key metrics
        atr_expanding = long_reasons.get('atr_expanding', False) or \
                       short_reasons.get('atr_contracting', False)
        volume_above = long_reasons.get('volume_above_avg', False)
        index_aligned = long_reasons.get('index_above_ema200', False) or \
                       short_reasons.get('index_below_ema200', False)
        
        return RegimeResult(
            direction=direction,
            score=score,
            eligible_for_long=long_eligible,
            eligible_for_short=short_eligible,
            reasons=long_reasons if long_eligible else short_reasons,
            index_aligned=index_aligned if index_data is not None else True,
            atr_expanding=atr_expanding,
            volume_above_average=volume_above
        )
    
    def _check_long_eligibility(self, 
                               stock_data: pd.DataFrame,
                               index_data: Optional[pd.DataFrame]) -> Tuple[bool, Dict]:
        """Check if long trades are eligible"""
        reasons = {}
        
        # Get latest values
        latest = stock_data.iloc[-1]
        
        # 1. Price > EMA200
        reasons['price_above_ema200'] = latest['Close'] > latest['EMA200']
        
        # 2. EMA50 > EMA200
        reasons['ema50_above_ema200'] = latest['EMA50'] > latest['EMA200']
        
        # 3. Index aligned (if available)
        if index_data is not None:
            index_data = self._ensure_indicators(index_data)
            index_latest = index_data.iloc[-1]
            reasons['index_above_ema200'] = index_latest['Close'] > index_latest['EMA200']
        else:
            reasons['index_above_ema200'] = True  # Assume aligned if no data
        
        # 4. ATR expanding
        atr = self._calculate_atr(stock_data)
        atr_5day_avg = atr.rolling(5).mean().iloc[-1]
        reasons['atr_expanding'] = atr.iloc[-1] > atr_5day_avg
        
        # 5. Volume > 20-day average
        vol_20day = stock_data['Volume'].rolling(20).mean().iloc[-1]
        reasons['volume_above_avg'] = latest['Volume'] > vol_20day
        
        # All conditions must be True for long eligibility
        eligible = all(reasons.values())
        
        return eligible, reasons
    
    def _check_short_eligibility(self,
                                stock_data: pd.DataFrame,
                                index_data: Optional[pd.DataFrame]) -> Tuple[bool, Dict]:
        """Check if short trades are eligible (inverse of long)"""
        reasons = {}
        
        # Get latest values
        latest = stock_data.iloc[-1]
        
        # 1. Price < EMA200
        reasons['price_below_ema200'] = latest['Close'] < latest['EMA200']
        
        # 2. EMA50 < EMA200
        reasons['ema50_below_ema200'] = latest['EMA50'] < latest['EMA200']
        
        # 3. Index aligned
        if index_data is not None:
            index_data = self._ensure_indicators(index_data)
            index_latest = index_data.iloc[-1]
            reasons['index_below_ema200'] = index_latest['Close'] < index_latest['EMA200']
        else:
            reasons['index_below_ema200'] = True
        
        # 4. ATR expanding (same check)
        atr = self._calculate_atr(stock_data)
        atr_5day_avg = atr.rolling(5).mean().iloc[-1]
        reasons['atr_contracting'] = atr.iloc[-1] > atr_5day_avg
        
        # 5. Volume > 20-day average
        vol_20day = stock_data['Volume'].rolling(20).mean().iloc[-1]
        reasons['volume_above_avg'] = latest['Volume'] > vol_20day
        
        # All conditions must be True for short eligibility
        eligible = all(reasons.values())
        
        return eligible, reasons
    
    def _calculate_regime_score(self, 
                                stock_data: pd.DataFrame,
                                long_reasons: Dict,
                                short_reasons: Dict) -> int:
        """
        Calculate 0-100 regime score
        Higher = stronger bullish regime
        Lower = stronger bearish regime
        50 = neutral
        """
        # Start at neutral
        score = 50
        
        # Count bullish factors (each worth +10)
        bullish_count = sum(1 for v in long_reasons.values() if v)
        
        # Count bearish factors
        bearish_count = sum(1 for v in short_reasons.values() if v)
        
        # Adjust score
        score += (bullish_count * 10)
        score -= (bearish_count * 10)
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs if not already present"""
        data = data.copy()
        
        if 'EMA20' not in data.columns:
            data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        if 'EMA50' not in data.columns:
            data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        if 'EMA200' not in data.columns:
            data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
        
        return data
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def get_market_bias(self, regime_result: RegimeResult) -> str:
        """
        Get human-readable market bias
        
        Args:
            regime_result: Result from analyze_regime
        
        Returns:
            String description of market bias
        """
        if regime_result.eligible_for_long:
            strength = "Strong" if regime_result.score >= 80 else \
                      "Moderate" if regime_result.score >= 60 else "Weak"
            return f"{strength} Bullish Bias (Score: {regime_result.score})"
        elif regime_result.eligible_for_short:
            strength = "Strong" if regime_result.score <= 20 else \
                      "Moderate" if regime_result.score <= 40 else "Weak"
            return f"{strength} Bearish Bias (Score: {regime_result.score})"
        else:
            return f"Neutral/Ranging Market (Score: {regime_result.score}) - NO TRADE"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Market Regime Filter - v1.0")
    print("=" * 70)
    print("\nThis module filters trades based on market regime.")
    print("\nUsage:")
    print("  filter = MarketRegimeFilter()")
    print("  result = filter.analyze_regime(stock_data, index_data)")
    print("  if result.eligible_for_long:")
    print("      # Proceed with long trade analysis")
    print("\nKey Features:")
    print("  ✓ Strict EMA alignment checks")
    print("  ✓ ATR expansion validation")
    print("  ✓ Volume confirmation")
    print("  ✓ Index correlation verification")
    print("  ✓ 0-100 regime scoring")
