"""
AI TRADING SYSTEM - MAIN ORCHESTRATOR
Institutional-grade two-engine trading system

Integrates:
- Market regime filter
- Stock eligibility validator
- Engine 1  (Micro-profit)
- Engine 2 (Big-runner)
- AI probability scorer
- Portfolio risk governor
- Trade database
- Signal formatter

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import all system components
from market_regime_filter import MarketRegimeFilter, RegimeResult
from stock_eligibility_validator import StockEligibilityValidator, EligibilityResult
from data_fetcher import IndianStockDataFetcher
from engine_one_micro_profit import EngineOneMicroProfit, TradeSetup, SetupType as Engine1SetupType
from engine_two_big_runner import EngineTwoBigRunner, BigRunnerSetup, SetupType as Engine2SetupType
from ai_probability_scorer import AIProbabilityScorer, ProbabilityComponents
from portfolio_risk_governor import PortfolioRiskGovernor, RiskCheckResult
from trade_database import TradeDatabase, TradeRecord
from trade_signal_formatter import TradeSignalFormatter, TradeSignal


@dataclass
class TwoEngineConfig:
    """Configuration for trading system"""
    capital: float = 10_00_000  # Default 10 lakh
    enable_engine1: bool = True
    enable_engine2: bool = True
    db_path: str = "trades.db"
    min_turnover_cr: float = 20.0
    auto_log_trades: bool = True


class TradingSystemOrchestrator:
    """
    Main orchestrator for dual-engine trading system
    
    Workflow:
    1. Fetch stock data + index data
    2. Check market regime (LONG/SHORT eligible?)
    3. Check stock eligibility (liquidity, spread, etc.)
    4. Scan for setups (Engine 1 + Engine 2)
    5. Calculate AI probability for each setup
    6. Check portfolio risk limits
    7. Generate trade signals for approved trades
    """
    
    def __init__(self, config: Optional[TwoEngineConfig] = None):
        """
        Initialize trading system
        
        Args:
            config: System configuration
        """
        self.config = config or TwoEngineConfig()
        
        # Initialize all components
        self.data_fetcher = IndianStockDataFetcher()
        self.regime_filter = MarketRegimeFilter()
        self.eligibility_validator = StockEligibilityValidator(
            min_turnover_cr=self.config.min_turnover_cr
        )
        self.engine1 = EngineOneMicroProfit()
        self.engine2 = EngineTwoBigRunner()
        self.probability_scorer = AIProbabilityScorer()
        self.risk_governor = PortfolioRiskGovernor(total_capital=self.config.capital)
        self.trade_db = TradeDatabase(db_path=self.config.db_path)
        self.signal_formatter = TradeSignalFormatter()
        
        print(f"✅ Trading System Initialized")
        print(f"   Capital: ₹{self.config.capital:,.0f}")
        print(f"   Engine 1: {'Enabled' if self.config.enable_engine1 else 'Disabled'}")
        print(f"   Engine 2: {'Enabled' if self.config.enable_engine2 else 'Disabled'}")
    
    def analyze_stock(self,
                     symbol: str,
                     sector: Optional[str] = None,
                     index: str = "NIFTY") -> Dict:
        """
        Complete analysis of a stock
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')
            sector: Stock sector (e.g., 'ENERGY', 'BANKING')
            index: Index to use ('NIFTY' or 'BANKNIFTY')
        
        Returns:
            Dictionary with all analysis results
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING: {symbol}")
        print(f"{'='*70}\n")
        
        # Step 1: Fetch data
        print("📥 Step 1: Fetching data...")
        stock_data = self.data_fetcher.fetch_stock_data(symbol)
        index_data = self.data_fetcher.fetch_index_data(index)
        vix = self.data_fetcher.get_current_vix()
        
        if stock_data is None:
            return {'error': 'Failed to fetch stock data', 'symbol': symbol}
        
        self.risk_governor.update_vix(vix)
        print(f"✅ Data fetched. VIX: {vix:.1f}")
        
        # Step 2: Market regime check
        print("\n🔍 Step 2: Checking market regime...")
        regime_result = self.regime_filter.analyze_regime(stock_data, index_data)
        print(f"   Direction: {regime_result.direction.value}")
        print(f"   Long Eligible: {'✅' if regime_result.eligible_for_long else '❌'}")
        print(f"   Short Eligible: {'✅' if regime_result.eligible_for_short else '❌'}")
        
        if not (regime_result.eligible_for_long or regime_result.eligible_for_short):
            return {
                'symbol': symbol,
                'status': 'NO_TRADE',
                'reason': 'Market regime not favorable',
                'regime': regime_result.direction.value
            }
        
        # Step 3: Stock eligibility
        print("\n📋 Step 3: Validating stock eligibility...")
        eligibility_result = self.eligibility_validator.validate_stock(
            symbol=symbol,
            data=stock_data,
            setup_type="breakout",  # Will check later
            index_data=index_data,
            sector=sector
        )
        print(f"   Eligible: {'✅' if eligibility_result.is_eligible else '❌'}")
        if not eligibility_result.is_eligible:
            print(f"   Warnings: {', '.join(eligibility_result.warnings)}")
        
        if not eligibility_result.is_eligible:
            return {
                'symbol': symbol,
                'status': 'NOT_ELIGIBLE',
                'reason': eligibility_result.warnings,
                'regime': regime_result.direction.value
            }
        
        # Step 4: Scan for setups
        print("\n🎯 Step 4: Scanning for trade setups...")
        all_signals = []
        
        if self.config.enable_engine1 and regime_result.eligible_for_long:
            print("   Scanning Engine 1 (Micro-Profit)...")
            engine1_setups = self.engine1.scan_for_setups(stock_data)
            print(f"   Found {len(engine1_setups)} Engine 1 setup(s)")
            
            for setup in engine1_setups:
                signal = self._process_setup(
                    setup=setup,
                    engine_type="MICRO",
                    symbol=symbol,
                    sector=sector or "UNKNOWN",
                    stock_data=stock_data,
                    index_data=index_data,
                    regime_result=regime_result
                )
                if signal:
                    all_signals.append(signal)
        
        if self.config.enable_engine2 and regime_result.eligible_for_long:
            print("   Scanning Engine 2 (Big-Runner)...")
            engine2_setups = self.engine2.scan_for_setups(stock_data)
            print(f"   Found {len(engine2_setups)} Engine 2 setup(s)")
            
            for setup in engine2_setups:
                signal = self._process_setup(
                    setup=setup,
                    engine_type="BIG_RUNNER",
                    symbol=symbol,
                    sector=sector or "UNKNOWN",
                    stock_data=stock_data,
                    index_data=index_data,
                    regime_result=regime_result
                )
                if signal:
                    all_signals.append(signal)
        
        # Step 5: Return results
        print(f"\n✅ Analysis complete. Generated {len(all_signals)} signal(s)")
        
        return {
            'symbol': symbol,
            'status': 'SUCCESS',
            'regime': regime_result.direction.value,
            'regime_score': regime_result.score,
            'eligible': eligibility_result.is_eligible,
            'signals': all_signals,
            'vix': vix
        }
    
    def _process_setup(self,
                      setup,
                      engine_type: str,
                      symbol: str,
                      sector: str,
                      stock_data: pd.DataFrame,
                      index_data: Optional[pd.DataFrame],
                      regime_result: RegimeResult) -> Optional[Dict]:
        """
        Process a single setup through probability scoring and risk checks
        
        Returns:
            Signal dict if approved, None if rejected
        """
        # Extract setup details
        if engine_type == "MICRO":
            direction = setup.direction
            entry = setup.entry_price
            stop = setup.stop_loss
            target = setup.targets['1.0R']
            setup_name = setup.setup_type.value
            expected_hold = "Swing"
            runner_mode = "DISABLED"
            trailing_method = "N/A"
        else:  # BIG_RUNNER
            direction = setup.direction
            entry = setup.entry_price
            stop = setup.stop_loss
            target = setup.partial_exit_target
            setup_name = setup.setup_type.value
            expected_hold = "Position"
            runner_mode = "ENABLED"
            trailing_method = setup.trailing_method.value
        
        # Calculate AI probability
        probability_result = self.probability_scorer.calculate_probability(
            stock_data=stock_data,
            index_data=index_data,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            risk_pct=0.5,  # Will be adjusted by risk governor
            setup_type=setup_name
        )
        
        prob = probability_result.final_probability
        print(f"\n      Setup: {setup_name}")
        print(f"      AI Probability: {prob:.1f}%")
        
        # Check probability threshold
        meets_threshold = self.probability_scorer.meets_threshold(prob, engine_type)
        if not meets_threshold:
            threshold = 70 if engine_type == "MICRO" else 65
            print(f"      ❌ Rejected: Probability {prob:.1f}% < {threshold}%")
            return None
        
        print(f"      ✅ Probability threshold met")
        
        # Check risk limits
        risk_check = self.risk_governor.can_open_new_trade(
            symbol=symbol,
            sector=sector,
            direction=direction,
            probability=prob,
            engine_type=engine_type
        )
        
        if not risk_check.allowed:
            print(f"      ❌ Risk check failed: {risk_check.reason}")
            return None
        
        print(f"      ✅ Risk check passed")
        
        # Calculate position size
        shares, risk_pct = self.risk_governor.calculate_position_size(
            probability=prob,
            entry_price=entry,
            stop_loss=stop
        )
        
        print(f"      Position: {shares} shares @ {risk_pct:.2f}% risk")
        
        # Determine trend strength
        trend_score = probability_result.trend_score
        if trend_score >= 75:
            trend_strength = "Strong"
        elif trend_score >= 50:
            trend_strength = "Medium"
        else:
            trend_strength = "Weak"
        
        # Create trade signal
        signal = TradeSignal(
            engine_type="MICRO-PROFIT" if engine_type == "MICRO" else "BIG-RUNNER",
            trade_type=direction,
            setup_type=setup_name,
            ai_probability=prob,
            entry=entry,
            stoploss=stop,
            target_1=target,
            runner_mode=runner_mode,
            trailing_method=trailing_method,
            risk_per_trade=risk_pct,
            expected_hold=expected_hold,
            sector=sector,
            index_alignment="YES" if regime_result.index_aligned else "NO",
            trend_strength=trend_strength,
            symbol=symbol,
            current_price=stock_data['Close'].iloc[-1],
            probability_components={
                'market': probability_result.market_score,
                'trend': probability_result.trend_score,
                'momentum': probability_result.momentum_score,
                'volume': probability_result.volume_score,
                'risk': probability_result.risk_score
            }
        )
        
        # Format signal
        signal_text = self.signal_formatter.format_signal(signal)
        signal_json = self.signal_formatter.format_json(signal)
        
        return {
            'signal': signal,
            'text': signal_text,
            'json': signal_json,
            'probability_details': probability_result,
            'position_size': shares,
            'risk_pct': risk_pct
        }
    
    def scan_multiple_stocks(self,
                            symbols: List[str],
                            sectors: Optional[Dict[str, str]] = None) -> Dict:
        """
        Scan multiple stocks for trade setups
        
        Args:
            symbols: List of stock symbols
            sectors: Optional dict mapping symbol to sector
        
        Returns:
            Dictionary with all signals
        """
        sectors = sectors or {}
        all_results = {}
        all_signals = []
        
        print(f"\n{'='*70}")
        print(f"SCANNING {len(symbols)} STOCKS")
        print(f"{'='*70}")
        
        for symbol in symbols:
            sector = sectors.get(symbol, "UNKNOWN")
            result = self.analyze_stock(symbol, sector=sector)
            all_results[symbol] = result
            
            if result.get('status') == 'SUCCESS' and result.get('signals'):
                all_signals.extend(result['signals'])
        
        print(f"\n{'='*70}")
        print(f"SCAN COMPLETE")
        print(f"{'='*70}")
        print(f"Total signals generated: {len(all_signals)}")
        
        return {
            'total_stocks_scanned': len(symbols),
            'total_signals': len(all_signals),
            'signals': all_signals,
            'detailed_results': all_results
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("AI Trading System Orchestrator - v1.0")
    print("=" * 70)
    print("\nInstitutional-Grade Two-Engine Trading System")
    print("\nEngines:")
    print("  1. Micro-Profit (≥70% probability, 0.8R-1.3R targets)")
    print("  2. Big-Runner (≥65% probability, 40-80%+ returns)")
    print("\nFeatures:")
    print("  ✓ Market regime filtering")
    print("  ✓ Stock eligibility validation")
    print("  ✓ AI probability scoring (5 components)")
    print("  ✓ Portfolio-level risk governance")
    print("  ✓ Dynamic position sizing")
    print("\n" + "=" * 70)
    
    # Example: Analyze a stock
    print("\n\nEXAMPLE: Analyzing RELIANCE...")
    
    config = TwoEngineConfig(capital=10_00_000)
    system = TradingSystemOrchestrator(config)
    
    result = system.analyze_stock("RELIANCE", sector="ENERGY")
    
    if result.get('signals'):
        print("\n\n📊 GENERATED SIGNALS:")
        for i, signal_data in enumerate(result['signals'], 1):
            print(f"\nSignal {i}:")
            print(signal_data['text'])
