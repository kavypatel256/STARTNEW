"""
DATA FETCHER & PREPROCESSOR
Unified data fetching for Indian NSE/BSE stocks

Handles:
- Stock OHLCV data fetching (yfinance)
- Index data (NIFTY/BANKNIFTY)
- Technical indicator calculation
- Data validation and cleaning
- India VIX fetching

Author: AI Trading System
Version: 1.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class IndianStockDataFetcher:
    """
    Fetches and preprocesses data for Indian stocks and indices
    
    Features:
    - Automatic NSE/BSE suffix handling
    - Multi-timeframe support
    - Pre-calculated technical indicators
    - Index data fetching
    - VIX data integration
    """
    
    def __init__(self):
        """Initialize data fetcher"""
        self.nifty_symbol = "^NSEI"
        self.banknifty_symbol = "^NSEBANK"
        self.india_vix_symbol = "^INDIAVIX"
    
    def fetch_stock_data(self,
                        symbol: str,
                        period: str = "6mo",
                        interval: str = "1d",
                        auto_add_suffix: bool = True,
                        max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with technical indicators
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', etc.)
            interval: Data interval ('1d', '1wk', '1h', '15m', etc.)
            auto_add_suffix: Automatically add .NS if no suffix
            max_retries: Maximum number of retry attempts
        
        Returns:
            DataFrame with OHLCV + indicators or None if fetch fails
        """
        import time
        
        # Add .NS suffix if not present
        if auto_add_suffix and '.' not in symbol:
            symbol = f"{symbol}.NS"
        
        # Try fetching with retries
        for attempt in range(max_retries):
            try:
                # Add delay between retries to avoid rate limiting
                if attempt > 0:
                    delay = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                    time.sleep(delay)
                
                # Fetch data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    # Add technical indicators
                    data = self._add_indicators(data)
                    return data
                
                # If empty, try .BO as fallback (only on first attempt)
                if attempt == 0 and symbol.endswith('.NS'):
                    time.sleep(1)
                    alt_symbol = symbol.replace('.NS', '.BO')
                    ticker = yf.Ticker(alt_symbol)
                    data = ticker.history(period=period, interval=interval)
                    
                    if not data.empty:
                        data = self._add_indicators(data)
                        return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching {symbol} after {max_retries} attempts: {e}")
                # Continue to next retry
                continue
        
        return None
    
    def fetch_index_data(self,
                        index: str = "NIFTY",
                        period: str = "6mo",
                        interval: str = "1d",
                        max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch index data
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            period: Data period
            interval: Data interval
            max_retries: Maximum retry attempts
        
        Returns:
            DataFrame with index OHLCV + indicators
        """
        import time
        
        symbol = self.nifty_symbol if index.upper() == "NIFTY" else self.banknifty_symbol
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    # Add indicators
                    data = self._add_indicators(data)
                    return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching {index} after {max_retries} attempts: {e}")
                continue
        
        return None
    
    def fetch_india_vix(self, period: str = "1mo", max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch India VIX data
        
        Args:
            period: Data period
            max_retries: Maximum retry attempts
        
        Returns:
            DataFrame with VIX data
        """
        import time
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** attempt)
                
                ticker = yf.Ticker(self.india_vix_symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    return data
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching India VIX after {max_retries} attempts: {e}")
                continue
        
        return None
    
    def get_current_vix(self) -> float:
        """
        Get current India VIX level
        
        Returns:
            Current VIX value or 0 if unavailable
        """
        data = self.fetch_india_vix(period="5d")
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return 0.0
    
    def fetch_multiple_stocks(self,
                             symbols: List[str],
                             period: str = "6mo",
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            data = self.fetch_stock_data(symbol, period, interval)
            if data is not None:
                results[symbol] = data
        
        return results
    
    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data
        
        Indicators added:
        - EMA 20, 50, 200
        - RSI 14
        - MACD (12, 26, 9)
        - ATR 14
        - VWAP (for intraday)
        - Bollinger Bands (20, 2)
        """
        data = data.copy()
        
        # EMAs
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'], period=14)
        
        # MACD
        macd_data = self._calculate_macd(data['Close'])
        data['MACD'] = macd_data['macd']
        data['MACD_Signal'] = macd_data['signal']
        data['MACD_Hist'] = macd_data['histogram']
        
        # ATR
        data['ATR'] = self._calculate_atr(data)
        
        # VWAP (if intraday data available)
        if 'Volume' in data.columns:
            data['VWAP'] = self._calculate_vwap(data)
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(data['Close'])
        data['BB_Upper'] = bb_data['upper']
        data['BB_Middle'] = bb_data['middle']
        data['BB_Lower'] = bb_data['lower']
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, 
                       prices: pd.Series,
                       fast: int = 12,
                       slow: int = 26,
                       signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
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
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        return vwap
    
    def _calculate_bollinger_bands(self,
                                   prices: pd.Series,
                                   period: int = 20,
                                   std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality
        
        Args:
            data: DataFrame to validate
        
        Returns:
            (is_valid, list of issues)
        """
        issues = []
        
        # Check for required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in data.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")
        
        # Check for sufficient data
        if len(data) < 200:
            issues.append(f"Insufficient data: {len(data)} rows (need 200+ for EMA200)")
        
        # Check for NaN values in recent data
        recent_nans = data.tail(50).isna().sum().sum()
        if recent_nans > 0:
            issues.append(f"NaN values in recent data: {recent_nans}")
        
        # Check for zero volume
        if 'Volume' in data.columns:
            zero_vol_count = (data['Volume'] == 0).sum()
            if zero_vol_count > len(data) * 0.1:  # >10% zero volume
                issues.append(f"High zero-volume days: {zero_vol_count}")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Indian Stock Data Fetcher - v1.0")
    print("=" * 70)
    
    # Initialize fetcher
    fetcher = IndianStockDataFetcher()
    
    # Example: Fetch RELIANCE data
    print("\nFetching RELIANCE data...")
    reliance_data = fetcher.fetch_stock_data("RELIANCE", period="6mo")
    
    if reliance_data is not None:
        print(f"✅ Fetched {len(reliance_data)} rows")
        print(f"Columns: {list(reliance_data.columns)}")
        print(f"\nLatest data:")
        print(reliance_data.tail(3))
    
    # Fetch NIFTY
    print("\n\nFetching NIFTY index...")
    nifty_data = fetcher.fetch_index_data("NIFTY")
    if nifty_data is not None:
        print(f"✅ Fetched {len(nifty_data)} rows")
    
    # Get current VIX
    print("\n\nCurrent India VIX:")
    vix = fetcher.get_current_vix()
    print(f"VIX: {vix:.2f}")
    
    print("\n" + "=" * 70)
