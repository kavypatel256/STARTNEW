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
import time
warnings.filterwarnings('ignore')

# Configure yfinance session with retry logic and headers
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def get_yf_session():
    """Create a session with retry logic and proper headers"""
    session = requests.Session()
    
    # Retry strategy: 3 retries with exponential backoff
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Add headers to avoid being blocked
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    return session

# Set global yfinance session
yf.set_tz_cache_location(None)


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
                        auto_add_suffix: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with technical indicators
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE' or 'RELIANCE.NS')
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', etc.)
            interval: Data interval ('1d', '1wk', '1h', '15m', etc.)
            auto_add_suffix: Automatically add .NS if no suffix
        
        Returns:
            DataFrame with OHLCV + indicators or None if fetch fails
        """
        # Add .NS suffix if not present
        if auto_add_suffix and '.' not in symbol:
            symbol = f"{symbol}.NS"
        
        try:
            # Create session with retry logic
            session = get_yf_session()
            
            # Fetch data with custom session
            ticker = yf.Ticker(symbol, session=session)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"Warning: No data fetched for {symbol}")
                # Try with .BO suffix as fallback
                if symbol.endswith('.NS'):
                    time.sleep(1)  # Small delay before retry
                    alt_symbol = symbol.replace('.NS', '.BO')
                    print(f"Trying alternate symbol: {alt_symbol}")
                    ticker = yf.Ticker(alt_symbol, session=session)
                    data = ticker.history(period=period, interval=interval)
                    
                if data.empty:
                    return None
            
            # Add technical indicators
            data = self._add_indicators(data)
            
            return data
        
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def fetch_index_data(self,
                        index: str = "NIFTY",
                        period: str = "6mo",
                        interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch index data
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            period: Data period
            interval: Data interval
        
        Returns:
            DataFrame with index OHLCV + indicators
        """
        symbol = self.nifty_symbol if index.upper() == "NIFTY" else self.banknifty_symbol
        
        try:
            session = get_yf_session()
            ticker = yf.Ticker(symbol, session=session)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Add indicators
            data = self._add_indicators(data)
            
            return data
        
        except Exception as e:
            print(f"Error fetching {index}: {e}")
            return None
    
    def fetch_india_vix(self, period: str = "1mo") -> Optional[pd.DataFrame]:
        """
        Fetch India VIX data
        
        Args:
            period: Data period
        
        Returns:
            DataFrame with VIX data
        """
        try:
            session = get_yf_session()
            ticker = yf.Ticker(self.india_vix_symbol, session=session)
            data = ticker.history(period=period)
            return data
        
        except Exception as e:
            print(f"Error fetching India VIX: {e}")
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
