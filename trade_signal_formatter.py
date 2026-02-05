"""
TRADE SIGNAL FORMATTER
Structured output for trade signals

Generates signals in the required format:
ENGINE TYPE: MICRO-PROFIT | BIG-RUNNER
TRADE TYPE: LONG | SHORT
SETUP TYPE: ...
AI PROBABILITY: XX%
ENTRY: ₹X
STOPLOSS: ₹Y
...etc

Author: AI Trading System
Version: 1.0
"""

from dataclasses import dataclass
from typing import Dict, Optional
import json


@dataclass
class TradeSignal:
    """Complete trade signal"""
    # Core info
    engine_type: str  # "MICRO-PROFIT" | "BIG-RUNNER"
    trade_type: str  # "LONG" | "SHORT"
    setup_type: str  # "Pullback" | "Breakout" | "VWAP" | "Retest" etc
    ai_probability: float
    
    # Prices
    entry: float
    stoploss: float
    target_1: float
    
    # Exit strategy
    runner_mode: str  # "ENABLED" | "DISABLED"
    trailing_method: str  # "EMA20" | "STRUCTURE" | "WEEKLY" | "N/A"
    
    # Risk
    risk_per_trade: float  # As %
    expected_hold: str  # "Intraday" | "Swing" | "Position"
    
    # Context
    sector: str
    index_alignment: str  # "YES" | "NO"
    trend_strength: str  # "Weak" | "Medium" | "Strong"
    
    # Optional additional info
    symbol: Optional[str] = None
    current_price: Optional[float] = None
    probability_components: Optional[Dict] = None


class TradeSignalFormatter:
    """
    Formats trade signals for output
    
    Multiple output formats:
    - Text (human-readable)
    - JSON (machine-readable)
    - Telegram (for alerts)
    """
    
    @staticmethod
    def format_signal(signal: TradeSignal) -> str:
        """
        Format as text output
        
        Args:
            signal: TradeSignal to format
        
        Returns:
            Formatted text string
        """
        output = []
        output.append("=" * 70)
        output.append(f"ENGINE TYPE: {signal.engine_type}")
        output.append(f"TRADE TYPE: {signal.trade_type}")
        output.append(f"SETUP TYPE: {signal.setup_type}")
        output.append(f"AI PROBABILITY: {signal.ai_probability:.1f}%")
        output.append("")
        
        output.append(f"ENTRY: ₹{signal.entry:.2f}")
        output.append(f"STOPLOSS: ₹{signal.stoploss:.2f}")
        output.append(f"TARGET 1: ₹{signal.target_1:.2f}")
        output.append(f"RUNNER MODE: {signal.runner_mode}")
        output.append(f"TRAILING METHOD: {signal.trailing_method}")
        output.append(f"RISK PER TRADE: {signal.risk_per_trade:.2f}%")
        output.append(f"EXPECTED HOLD: {signal.expected_hold}")
        output.append(f"SECTOR: {signal.sector}")
        output.append(f"INDEX ALIGNMENT: {signal.index_alignment}")
        output.append(f"TREND STRENGTH: {signal.trend_strength}")
        
        if signal.symbol:
            output.append(f"\nSYMBOL: {signal.symbol}")
        
        if signal.current_price:
            output.append(f"CURRENT PRICE: ₹{signal.current_price:.2f}")
        
        output.append("=" * 70)
        
        return "\n".join(output)
    
    @staticmethod
    def format_json(signal: TradeSignal) -> str:
        """
        Format as JSON
        
        Args:
            signal: TradeSignal to format
        
        Returns:
            JSON string
        """
        data = {
            'engine_type': signal.engine_type,
            'trade_type': signal.trade_type,
            'setup_type': signal.setup_type,
            'ai_probability': round(signal.ai_probability, 1),
            'entry': round(signal.entry, 2),
            'stoploss': round(signal.stoploss, 2),
            'target_1': round(signal.target_1, 2),
            'runner_mode': signal.runner_mode,
            'trailing_method': signal.trailing_method,
            'risk_per_trade': round(signal.risk_per_trade, 2),
            'expected_hold': signal.expected_hold,
            'sector': signal.sector,
            'index_alignment': signal.index_alignment,
            'trend_strength': signal.trend_strength
        }
        
        if signal.symbol:
            data['symbol'] = signal.symbol
        
        if signal.current_price:
            data['current_price'] = round(signal.current_price, 2)
        
        if signal.probability_components:
            data['probability_components'] = signal.probability_components
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def format_telegram(signal: TradeSignal) -> str:
        """
        Format for Telegram alert (concise)
        
        Args:
            signal: TradeSignal to format
        
        Returns:
            Telegram-formatted string
        """
        lines = []
        lines.append(f"🎯 {signal.engine_type} SIGNAL")
        lines.append(f"📊 {signal.symbol or 'STOCK'} | {signal.trade_type}")
        lines.append(f"🔍 Setup: {signal.setup_type}")
        lines.append(f"🤖 AI Probability: {signal.ai_probability:.1f}%")
        lines.append("")
        lines.append(f"📈 Entry: ₹{signal.entry:.2f}")
        lines.append(f"⛔ Stop: ₹{signal.stoploss:.2f}")
        lines.append(f"🎯 Target: ₹{signal.target_1:.2f}")
        lines.append(f"⚠️ Risk: {signal.risk_per_trade:.2f}%")
        lines.append("")
        lines.append(f"📅 Hold: {signal.expected_hold}")
        lines.append(f"💼 Sector: {signal.sector}")
        lines.append(f"📊 Trend: {signal.trend_strength}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_compact(signal: TradeSignal) -> str:
        """
        Ultra-compact single-line format
        
        Args:
            signal: TradeSignal to format
        
        Returns:
            Compact string
        """
        return (f"{signal.symbol or 'STOCK'} | {signal.engine_type} | "
                f"{signal.trade_type} @ ₹{signal.entry:.0f} | "
                f"SL: ₹{signal.stoploss:.0f} | TGT: ₹{signal.target_1:.0f} | "
                f"Prob: {signal.ai_probability:.0f}% | {signal.setup_type}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Trade Signal Formatter - v1.0")
    print("=" * 70)
    
    # Example signal
    signal = TradeSignal(
        engine_type="MICRO-PROFIT",
        trade_type="LONG",
        setup_type="VWAP Reclaim",
        ai_probability=74.2,
        entry=2450.0,
        stoploss=2420.0,
        target_1=2478.0,
        runner_mode="DISABLED",
        trailing_method="N/A",
        risk_per_trade=0.38,
        expected_hold="Swing",
        sector="BANKING",
        index_alignment="YES",
        trend_strength="Strong",
        symbol="RELIANCE.NS",
        current_price=2445.0
    )
    
    # Format as text
    print("\n📄 TEXT FORMAT:")
    print(TradeSignalFormatter.format_signal(signal))
    
    # Format as JSON
    print("\n📋 JSON FORMAT:")
    print(TradeSignalFormatter.format_json(signal))
    
    # Format for Telegram
    print("\n✈️ TELEGRAM FORMAT:")
    print(TradeSignalFormatter.format_telegram(signal))
    
    # Compact format
    print("\n📌 COMPACT FORMAT:")
    print(TradeSignalFormatter.format_compact(signal))
