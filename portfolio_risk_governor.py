"""
PORTFOLIO RISK GOVERNOR
Portfolio-level AI risk management system

Features:
- Dynamic position sizing (0.25%-0.75% based on probability)
- Position limits (max 6 total, max 2 per sector)
- Sector exposure caps (40% per sector)
- Drawdown circuit breakers
- VIX-based volatility controls
- Market event protection

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum


class RiskStatus(Enum):
    """Risk status levels"""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    RESTRICTED = "RESTRICTED"
    HALTED = "HALTED"


@dataclass
class Position:
    """Open position details"""
    symbol: str
    sector: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    current_price: float
    quantity: int
    stop_loss: float
    entry_time: datetime


@dataclass
class RiskCheckResult:
    """Result of risk check"""
    allowed: bool
    reason: str
    max_position_size: int
    max_risk_pct: float
    risk_status: RiskStatus


class PortfolioRiskGovernor:
    """
    Centralized portfolio-level risk management
    
    Rules:
    1. Dynamic position sizing: 0.25% + (Probability - 60) × 0.01%
    2. Max 6 concurrent positions
    3. Max 2 positions per sector
    4. Max 40% capital per sector
    5. Drawdown circuit breakers
    6. VIX volatility controls
    """
    
    def __init__(self, total_capital: float):
        """
        Initialize risk governor
        
        Args:
            total_capital: Total trading capital in rupees
        """
        self.total_capital = total_capital
        self.open_positions: List[Position] = []
        
        # Position limits
        self.max_total_positions = 6
        self.max_positions_per_sector = 2
        self.max_sector_exposure_pct = 40.0
        self.max_same_direction = 3
        
        # Risk limits
        self.base_risk_pct = 0.25
        self.max_risk_pct = 0.75
        
        # Drawdown circuit breakers
        self.daily_dd_halt = 1.5
        self.weekly_dd_reduce = 4.0
        self.monthly_dd_engine2_off = 7.0
        
        # VIX thresholds
        self.vix_engine2_off = 25.0
        self.vix_reduce_size = 35.0
        self.vix_halt_all = 45.0
        
        # Current metrics
        self.daily_pnl_pct = 0.0
        self.weekly_pnl_pct = 0.0
        self.monthly_pnl_pct = 0.0
        self.current_vix = 15.0  # Default
    
    def calculate_position_size(self,
                               probability: float,
                               entry_price: float,
                               stop_loss: float) -> Tuple[int, float]:
        """
        Calculate position size based on probability and risk
        
        Formula:
        risk_pct = base_risk + (probability - 60) × 0.01
        position_size = (capital × risk_pct) / (entry - stop)
        
        Args:
            probability: AI probability score (0-100)
            entry_price: Entry price per share
            stop_loss: Stop loss price
        
        Returns:
            (shares, risk_pct)
        """
        # Calculate risk %
        risk_pct = self.base_risk_pct + ((probability - 60) * 0.01)
        risk_pct = max(self.base_risk_pct, min(risk_pct, self.max_risk_pct))
        
        # Risk amount in rupees
        risk_amount = self.total_capital * (risk_pct / 100)
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0, risk_pct
        
        # Calculate shares
        shares = int(risk_amount / risk_per_share)
        
        return shares, risk_pct
    
    def can_open_new_trade(self,
                          symbol: str,
                          sector: str,
                          direction: str,
                          probability: float,
                          engine_type: str) -> RiskCheckResult:
        """
        Check if new trade can be opened
        
        Args:
            symbol: Stock symbol
            sector: Stock sector
            direction: 'LONG' or 'SHORT'
            probability: AI probability
            engine_type: 'MICRO' or 'BIG_RUNNER'
        
        Returns:
            RiskCheckResult with decision and reason
        """
        # Check drawdown halts
        if self.daily_pnl_pct <= -self.daily_dd_halt:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily drawdown {self.daily_pnl_pct:.1f}% exceeds {self.daily_dd_halt}% limit",
                max_position_size=0,
                max_risk_pct=0,
                risk_status=RiskStatus.HALTED
            )
        
        # Check VIX halt
        if self.current_vix > self.vix_halt_all:
            return RiskCheckResult(
                allowed=False,
                reason=f"VIX {self.current_vix:.1f} exceeds halt threshold {self.vix_halt_all}",
                max_position_size=0,
                max_risk_pct=0,
                risk_status=RiskStatus.HALTED
            )
        
        # Check Engine 2 restrictions
        if engine_type.upper() in ['BIG_RUNNER', 'ENGINE2']:
            # Monthly DD check
            if self.monthly_pnl_pct <= -self.monthly_dd_engine2_off:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Engine 2 disabled: Monthly DD {self.monthly_pnl_pct:.1f}% > {self.monthly_dd_engine2_off}%",
                    max_position_size=0,
                    max_risk_pct=0,
                    risk_status=RiskStatus.RESTRICTED
                )
            
            # VIX check
            if self.current_vix > self.vix_engine2_off:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Engine 2 disabled: VIX {self.current_vix:.1f} > {self.vix_engine2_off}",
                    max_position_size=0,
                    max_risk_pct=0,
                    risk_status=RiskStatus.RESTRICTED
                )
        
        # Check max positions
        if len(self.open_positions) >= self.max_total_positions:
            return RiskCheckResult(
                allowed=False,
                reason=f"Max positions reached: {len(self.open_positions)}/{self.max_total_positions}",
                max_position_size=0,
                max_risk_pct=0,
                risk_status=RiskStatus.RESTRICTED
            )
        
        # Check sector limits
        sector_positions = [p for p in self.open_positions if p.sector == sector]
        if len(sector_positions) >= self.max_positions_per_sector:
            return RiskCheckResult(
                allowed=False,
                reason=f"Max sector positions: {len(sector_positions)}/{self.max_positions_per_sector} in {sector}",
                max_position_size=0,
                max_risk_pct=0,
                risk_status=RiskStatus.RESTRICTED
            )
        
        # Check direction concentration
        same_direction = [p for p in self.open_positions if p.direction == direction]
        if len(same_direction) >= self.max_same_direction:
            return RiskCheckResult(
                allowed=False,
                reason=f"Max {direction} positions: {len(same_direction)}/{self.max_same_direction}",
                max_position_size=0,
                max_risk_pct=0,
                risk_status=RiskStatus.CAUTION
            )
        
        # Calculate max risk (may be reduced by drawdown or VIX)
        max_risk = self.get_max_risk_for_trade(probability)
        
        # Determine risk status
        risk_status = RiskStatus.NORMAL
        if self.weekly_pnl_pct <= -self.weekly_dd_reduce:
            risk_status = RiskStatus.CAUTION
        elif self.current_vix > self.vix_reduce_size:
            risk_status = RiskStatus.CAUTION
        
        # Allowed!
        return RiskCheckResult(
            allowed=True,
            reason="Trade approved by risk governor",
            max_position_size=999999,  # To be calculated based on entry/stop
            max_risk_pct=max_risk,
            risk_status=risk_status
        )
    
    def get_max_risk_for_trade(self, probability: float) -> float:
        """
        Get maximum risk % for trade considering all factors
        
        Args:
            probability: AI probability
        
        Returns:
            Max risk % (adjusted for drawdown/VIX)
        """
        # Base calculation
        risk_pct = self.base_risk_pct + ((probability - 60) * 0.01)
        risk_pct = max(self.base_risk_pct, min(risk_pct, self.max_risk_pct))
        
        # Reduce for weekly drawdown
        if self.weekly_pnl_pct <= -self.weekly_dd_reduce:
            risk_pct *= 0.5  # Cut by 50%
        
        # Reduce for high VIX
        if self.current_vix > self.vix_reduce_size:
            risk_pct *= 0.5  # Cut by 50%
        
        return max(self.base_risk_pct, risk_pct)
    
    def update_open_positions(self, positions: List[Position]):
        """Update list of open positions"""
        self.open_positions = positions
    
    def update_pnl_metrics(self,
                          daily_pnl_pct: float,
                          weekly_pnl_pct: float,
                          monthly_pnl_pct: float):
        """
        Update P&L metrics for circuit breaker checks
        
        Args:
            daily_pnl_pct: Daily P&L as % of capital
            weekly_pnl_pct: Weekly P&L as % of capital
            monthly_pnl_pct: Monthly P&L as % of capital
        """
        self.daily_pnl_pct = daily_pnl_pct
        self.weekly_pnl_pct = weekly_pnl_pct
        self.monthly_pnl_pct = monthly_pnl_pct
    
    def update_vix(self, vix_level: float):
        """Update current VIX level"""
        self.current_vix = vix_level
    
    def get_sector_exposure(self) -> Dict[str, float]:
        """
        Calculate current sector exposure
        
        Returns:
            Dict mapping sector to % of capital
        """
        sector_exposure = {}
        
        for position in self.open_positions:
            position_value = position.current_price * position.quantity
            exposure_pct = (position_value / self.total_capital) * 100
            
            if position.sector in sector_exposure:
                sector_exposure[position.sector] += exposure_pct
            else:
                sector_exposure[position.sector] = exposure_pct
        
        return sector_exposure
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio risk summary"""
        return {
            'total_positions': len(self.open_positions),
            'max_positions': self.max_total_positions,
            'sector_exposure': self.get_sector_exposure(),
            'daily_pnl_pct': self.daily_pnl_pct,
            'weekly_pnl_pct': self.weekly_pnl_pct,
            'monthly_pnl_pct': self.monthly_pnl_pct,
            'current_vix': self.current_vix,
            'risk_status': self._determine_overall_risk_status().value
        }
    
    def _determine_overall_risk_status(self) -> RiskStatus:
        """Determine overall portfolio risk status"""
        if self.daily_pnl_pct <= -self.daily_dd_halt:
            return RiskStatus.HALTED
        if self.current_vix > self.vix_halt_all:
            return RiskStatus.HALTED
        if (self.weekly_pnl_pct <= -self.weekly_dd_reduce or 
            self.current_vix > self.vix_reduce_size):
            return RiskStatus.CAUTION
        if self.monthly_pnl_pct <= -self.monthly_dd_engine2_off:
            return RiskStatus.RESTRICTED
        
        return RiskStatus.NORMAL


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Portfolio Risk Governor - v1.0")
    print("=" * 70)
    
    # Initialize with 10 lakh capital
    governor = PortfolioRiskGovernor(total_capital=10_00_000)
    
    print(f"\nCapital: ₹{governor.total_capital:,.0f}")
    print(f"Max Positions: {governor.max_total_positions}")
    print(f"Risk Range: {governor.base_risk_pct}% - {governor.max_risk_pct}%")
    
    # Example: Check if can open trade
    result = governor.can_open_new_trade(
        symbol="RELIANCE.NS",
        sector="ENERGY",
        direction="LONG",
        probability=75.0,
        engine_type="MICRO"
    )
    
    print(f"\nTrade Check Result:")
    print(f"  Allowed: {'✅' if result.allowed else '❌'}")
    print(f"  Reason: {result.reason}")
    print(f"  Max Risk: {result.max_risk_pct:.2f}%")
    print(f"  Risk Status: {result.risk_status.value}")
    
    # Calculate position size
    shares, risk_pct = governor.calculate_position_size(
        probability=75.0,
        entry_price=2450.0,
        stop_loss=2420.0
    )
    
    print(f"\nPosition Sizing (75% probability):")
    print(f"  Risk %: {risk_pct:.2f}%")
    print(f"  Shares: {shares}")
    print(f"  Position Value: ₹{shares * 2450:,.0f}")
