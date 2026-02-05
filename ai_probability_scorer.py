"""
AI PROBABILITY SCORER
5-component weighted probability calculation system

Components (with weights):
1. Market Score (25%) - Overall market conditions
2. Trend Score (25%) - Trend quality and strength
3. Momentum Score (20%) - Price momentum indicators
4. Volume Score (20%) - Volume characteristics
5. Risk Score (10%) - Trade risk/reward metrics

Final Probability = weighted sum
Engine 1 threshold: ≥70%
Engine 2 threshold: ≥65%

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ProbabilityComponents:
    """Individual probability component scores"""
    market_score: int  # 0-100
    trend_score: int  # 0-100
    momentum_score: int  # 0-100
    volume_score: int  # 0-100
    risk_score: int  # 0-100
    final_probability: float  # 0-100
    component_details: Dict[str, Dict]  # Breakdown of each component


class AIProbabilityScorer:
    """
    AI-powered probability scoring system
    
    Formula:
    AI_PROBABILITY = 
      (0.25 × market_score) +
      (0.25 × trend_score) +
      (0.20 × momentum_score) +
      (0.20 × volume_score) +
      (0.10 × risk_score)
    
    Thresholds:
    - Engine 1 (Micro-Profit): ≥ 70%
    - Engine 2 (Big-Runner): ≥ 65%
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize probability scorer
        
        Args:
            weights: Optional custom weights (must sum to 1.0)
        """
        # Default weights
        self.weights = weights or {
            'market': 0.25,
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.20,
            'risk': 0.10
        }
        
        # Validate weights sum to 1.0
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        # Probability thresholds
        self.engine1_threshold = 70.0
        self.engine2_threshold = 65.0
    
    def calculate_probability(self,
                             stock_data: pd.DataFrame,
                             index_data: Optional[pd.DataFrame],
                             entry_price: float,
                             stop_loss: float,
                             target_price: float,
                             risk_pct: float,
                             setup_type: str) -> ProbabilityComponents:
        """
        Calculate complete probability score
        
        Args:
            stock_data: Stock OHLCV with indicators
            index_data: Index data for market alignment
            entry_price: Proposed entry price
            stop_loss: Stop loss price
            target_price: Target price
            risk_pct: Risk as % of capital
            setup_type: Type of setup (for volume scoring)
        
        Returns:
            ProbabilityComponents with all scores
        """
        # Calculate each component
        market_score, market_details = self.calculate_market_score(
            stock_data, index_data
        )
        
        trend_score, trend_details = self.calculate_trend_score(stock_data)
        
        momentum_score, momentum_details = self.calculate_momentum_score(stock_data)
        
        volume_score, volume_details = self.calculate_volume_score(
            stock_data, setup_type
        )
        
        risk_score, risk_details = self.calculate_risk_score(
            entry_price, stop_loss, target_price, risk_pct
        )
        
        # Calculate weighted final probability
        final_probability = (
            (self.weights['market'] * market_score) +
            (self.weights['trend'] * trend_score) +
            (self.weights['momentum'] * momentum_score) +
            (self.weights['volume'] * volume_score) +
            (self.weights['risk'] * risk_score)
        )
        
        # Compile details
        component_details = {
            'market': market_details,
            'trend': trend_details,
            'momentum': momentum_details,
            'volume': volume_details,
            'risk': risk_details
        }
        
        return ProbabilityComponents(
            market_score=market_score,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            risk_score=risk_score,
            final_probability=round(final_probability, 1),
            component_details=component_details
        )
    
    def calculate_market_score(self,
                              stock_data: pd.DataFrame,
                              index_data: Optional[pd.DataFrame]) -> Tuple[int, Dict]:
        """
        Calculate market score (max 100)
        
        Factors:
        - Price > EMA200: +30
        - EMA50 > EMA200: +20
        - Index aligned: +20
        - ATR expanding: +15
        - Market breadth positive: +15
        """
        score = 0
        details = {}
        
        latest = stock_data.iloc[-1]
        
        # 1. Price > EMA200 (+30)
        if latest['Close'] > latest['EMA200']:
            score += 30
            details['price_above_ema200'] = True
        else:
            details['price_above_ema200'] = False
        
        # 2. EMA50 > EMA200 (+20)
        if latest['EMA50'] > latest['EMA200']:
            score += 20
            details['ema50_above_ema200'] = True
        else:
            details['ema50_above_ema200'] = False
        
        # 3. Index aligned (+20)
        if index_data is not None and len(index_data) > 0:
            index_latest = index_data.iloc[-1]
            if index_latest['Close'] > index_latest['EMA200']:
                score += 20
                details['index_aligned'] = True
            else:
                details['index_aligned'] = False
        else:
            score += 20  # Assume aligned if no data
            details['index_aligned'] = True
        
        # 4. ATR expanding (+15)
        if 'ATR' in stock_data.columns:
            atr = stock_data['ATR'].iloc[-1]
            atr_5day = stock_data['ATR'].rolling(5).mean().iloc[-1]
            if atr > atr_5day:
                score += 15
                details['atr_expanding'] = True
            else:
                details['atr_expanding'] = False
        else:
            details['atr_expanding'] = None
        
        # 5. Market breadth (+15) - simplified using RSI as proxy
        if 'RSI' in stock_data.columns:
            rsi = stock_data['RSI'].iloc[-1]
            if 50 < rsi < 80:  # Positive but not overbought
                score += 15
                details['breadth_positive'] = True
            else:
                details['breadth_positive'] = False
        else:
            details['breadth_positive'] = None
        
        return min(100, score), details
    
    def calculate_trend_score(self, stock_data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Calculate trend score (max 100)
        
        Factors:
        - Higher highs + higher lows: +30
        - EMA20 > EMA50 > EMA200: +25
        - Breakout above resistance: +25
        - Retest held: +20
        """
        score = 0
        details = {}
        
        latest = stock_data.iloc[-1]
        
        # 1. Higher highs + higher lows (+30)
        recent_20 = stock_data.tail(20)
        if len(recent_20) >= 10:
            first_half_high = recent_20.iloc[:10]['High'].max()
            second_half_high = recent_20.iloc[10:]['High'].max()
            first_half_low = recent_20.iloc[:10]['Low'].min()
            second_half_low = recent_20.iloc[10:]['Low'].min()
            
            hh = second_half_high > first_half_high
            hl = second_half_low > first_half_low
            
            if hh and hl:
                score += 30
                details['higher_highs_lows'] = True
            else:
                details['higher_highs_lows'] = False
        
        # 2. EMA20 > EMA50 > EMA200 (+25)
        ema_aligned = (latest['EMA20'] > latest['EMA50'] > latest['EMA200'])
        if ema_aligned:
            score += 25
            details['ema_alignment'] = True
        else:
            details['ema_alignment'] = False
        
        # 3. Breakout above resistance (+25)
        resistance = stock_data['High'].tail(30).iloc[:-1].max()
        breakout = latest['Close'] > resistance
        if breakout:
            score += 25
            details['breakout'] = True
        else:
            details['breakout'] = False
        
        # 4. Retest held (+20) - check if price retested and held
        if breakout:
            # Check if any recent candle came back near resistance
            recent_5 = stock_data.tail(5)
            for idx, row in recent_5.iterrows():
                if abs(row['Low'] - resistance) / resistance < 0.02:
                    if row['Close'] > resistance:
                        score += 20
                        details['retest_held'] = True
                        break
            else:
                details['retest_held'] = False
        else:
            details['retest_held'] = False
        
        return min(100, score), details
    
    def calculate_momentum_score(self, stock_data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Calculate momentum score (max 100)
        
        Factors:
        - RSI 55-70: +30 (scaled)
        - MACD histogram expanding: +25
        - Strong close: +25
        - ROC positive: +20
        """
        score = 0
        details = {}
        
        latest = stock_data.iloc[-1]
        
        # 1. RSI 55-70 (+30, scaled)
        if 'RSI' in stock_data.columns:
            rsi = latest['RSI']
            if 55 <= rsi <= 70:
                score += 30
                details['rsi_optimal'] = True
            elif 45 <= rsi < 55:
                score += 15  # Partial credit
                details['rsi_optimal'] = False
            elif 70 < rsi <= 80:
                score += 20  # Overbought but still ok
                details['rsi_optimal'] = False
            else:
                details['rsi_optimal'] = False
            details['rsi_value'] = round(rsi, 1)
        
        # 2. MACD histogram expanding (+25)
        if 'MACD_Hist' in stock_data.columns:
            hist_current = latest['MACD_Hist']
            hist_prev = stock_data['MACD_Hist'].iloc[-2]
            expanding = abs(hist_current) > abs(hist_prev) and hist_current > 0
            if expanding:
                score += 25
                details['macd_expanding'] = True
            else:
                details['macd_expanding'] = False
        
        # 3. Strong close (+25)
        candle_range = latest['High'] - latest['Low']
        if candle_range > 0:
            close_position = (latest['Close'] - latest['Low']) / candle_range
            if close_position > 0.7:
                score += 25
                details['strong_close'] = True
            else:
                details['strong_close'] = False
            details['close_position'] = round(close_position, 2)
        else:
            details['strong_close'] = False
        
        # 4. ROC positive (+20)
        roc_period = 10
        if len(stock_data) >= roc_period + 1:
            roc = ((latest['Close'] - stock_data['Close'].iloc[-(roc_period+1)]) / 
                   stock_data['Close'].iloc[-(roc_period+1)]) * 100
            if roc > 0:
                score += 20
                details['roc_positive'] = True
            else:
                details['roc_positive'] = False
            details['roc_value'] = round(roc, 2)
        
        return min(100, score), details
    
    def calculate_volume_score(self,
                              stock_data: pd.DataFrame,
                              setup_type: str) -> Tuple[int, Dict]:
        """
        Calculate volume score (max 100)
        
        Factors:
        - Breakout volume > 2× avg: +35
        - Rising volume trend: +25
        - Delivery volume rising: +20 (placeholder)
        - Pullback volume contraction: +20
        """
        score = 0
        details = {}
        
        latest = stock_data.iloc[-1]
        avg_volume = stock_data['Volume'].rolling(20).mean().iloc[-1]
        
        # 1. Breakout volume > 2× avg (+35)
        if 'breakout' in setup_type.lower():
            if latest['Volume'] > avg_volume * 2.0:
                score += 35
                details['breakout_volume'] = True
            else:
                details['breakout_volume'] = False
        else:
            # For non-breakouts, give partial credit for volume > avg
            if latest['Volume'] > avg_volume * 1.5:
                score += 20
                details['breakout_volume'] = 'partial'
        
        details['volume_vs_avg'] = round(latest['Volume'] / avg_volume, 2)
        
        # 2. Rising volume trend (+25)
        vol_5day = stock_data['Volume'].rolling(5).mean().iloc[-1]
        vol_20day = stock_data['Volume'].rolling(20).mean().iloc[-1]
        rising_trend = vol_5day > vol_20day
        if rising_trend:
            score += 25
            details['rising_volume_trend'] = True
        else:
            details['rising_volume_trend'] = False
        
        # 3. Delivery volume rising (+20)
        # Placeholder - would need NSE delivery data
        score += 20  # Assume OK for now
        details['delivery_volume'] = 'assumed_ok'
        
        # 4. Pullback volume contraction (+20)
        if 'pullback' in setup_type.lower():
            recent_3_vol = stock_data['Volume'].tail(3).mean()
            if recent_3_vol < avg_volume * 0.8:
                score += 20
                details['pullback_contraction'] = True
            else:
                details['pullback_contraction'] = False
        else:
            details['pullback_contraction'] = 'n/a'
        
        return min(100, score), details
    
    def calculate_risk_score(self,
                            entry: float,
                            stop: float,
                            target: float,
                            risk_pct: float) -> Tuple[int, Dict]:
        """
        Calculate risk score (max 100)
        
        Factors:
        - Stop below structure: +40 (assumed if provided)
        - Risk < 1% capital: +30
        - RR ≥ 1:1.5 (Engine 1) or ≥ 1:3 (Engine 2): +20
        """
        score = 0
        details = {}
        
        # 1. Stop below structure (+40)
        # Assume stop is well-placed if provided
        score += 40
        details['stop_below_structure'] = True
        
        # 2. Risk < 1% capital (+30)
        if risk_pct < 1.0:
            score += 30
            details['risk_acceptable'] = True
        elif risk_pct < 1.5:
            score += 15  # Partial credit
            details['risk_acceptable'] = 'marginal'
        else:
            details['risk_acceptable'] = False
        
        details['risk_pct'] = round(risk_pct, 3)
        
        # 3. Risk:reward ratio (+20)
        risk = entry - stop
        reward = target - entry
        
        if risk > 0:
            rr_ratio = reward / risk
            details['rr_ratio'] = round(rr_ratio, 2)
            
            if rr_ratio >= 3.0:
                score += 20
                details['rr_acceptable'] = True
            elif rr_ratio >= 1.5:
                score += 15  # Partial credit
                details['rr_acceptable'] = 'marginal'
            else:
                details['rr_acceptable'] = False
        else:
            details['rr_ratio'] = 0
            details['rr_acceptable'] = False
        
        return min(100, score), details
    
    def meets_threshold(self, probability: float, engine_type: str) -> bool:
        """
        Check if probability meets engine threshold
        
        Args:
            probability: Calculated probability
            engine_type: 'MICRO' or 'BIG_RUNNER'
        
        Returns:
            True if meets threshold
        """
        if engine_type.upper() in ['MICRO', 'ENGINE1', '1']:
            return probability >= self.engine1_threshold
        elif engine_type.upper() in ['BIG_RUNNER', 'ENGINE2', '2']:
            return probability >= self.engine2_threshold
        else:
            return False
    
    def get_probability_report(self, components: ProbabilityComponents) -> str:
        """Generate human-readable probability report"""
        report = f"\n{'='*70}\n"
        report += f"AI PROBABILITY SCORE: {components.final_probability:.1f}%\n"
        report += f"{'='*70}\n\n"
        
        report += "Component Scores:\n"
        report += f"  Market Score:    {components.market_score}/100 "
        report += f"(weight: {self.weights['market']*100:.0f}%)\n"
        report += f"  Trend Score:     {components.trend_score}/100 "
        report += f"(weight: {self.weights['trend']*100:.0f}%)\n"
        report += f"  Momentum Score:  {components.momentum_score}/100 "
        report += f"(weight: {self.weights['momentum']*100:.0f}%)\n"
        report += f"  Volume Score:    {components.volume_score}/100 "
        report += f"(weight: {self.weights['volume']*100:.0f}%)\n"
        report += f"  Risk Score:      {components.risk_score}/100 "
        report += f"(weight: {self.weights['risk']*100:.0f}%)\n\n"
        
        report += f"Engine Thresholds:\n"
        report += f"  Engine 1 (Micro-Profit): ≥{self.engine1_threshold}% "
        if components.final_probability >= self.engine1_threshold:
            report += "✅ PASS\n"
        else:
            report += "❌ FAIL\n"
        
        report += f"  Engine 2 (Big-Runner):   ≥{self.engine2_threshold}% "
        if components.final_probability >= self.engine2_threshold:
            report += "✅ PASS\n"
        else:
            report += "❌ FAIL\n"
        
        report += f"\n{'='*70}\n"
        
        return report


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("AI Probability Scorer - v1.0")
    print("=" * 70)
    print("\n5-Component Weighted Probability System")
    print("\nComponents:")
    print("  1. Market Score (25%)")
    print("  2. Trend Score (25%)")
    print("  3. Momentum Score (20%)")
    print("  4. Volume Score (20%)")
    print("  5. Risk Score (10%)")
    print("\nThresholds:")
    print("  Engine 1: ≥70%")
    print("  Engine 2: ≥65%")
