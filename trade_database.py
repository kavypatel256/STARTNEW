"""
TRADE DATABASE
SQLite database for trade logging and ML training

Schema stores:
- Trade entry/exit details
- Probability component scores
- Outcome (win/loss)
- Additional features for ML

Used by ML self-optimizer for online learning

Author: AI Trading System
Version: 1.0
"""

import sqlite3
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import datetime
import os


@dataclass
class TradeRecord:
    """Single trade record"""
    # Identification
    timestamp: str
    symbol: str
    engine_type: str  # 'MICRO' or 'BIG_RUNNER'
    setup_type: str
    direction: str  # 'LONG' or 'SHORT'
    
    # Prices
    entry_price: float
    stop_loss: float
    target_price: float
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # Probability components
    market_score: int = 0
    trend_score: int = 0
    momentum_score: int = 0
    volume_score: int = 0
    risk_score: int = 0
    final_probability: float = 0.0
    
    # Additional features
    sector: Optional[str] = None
    atr_multiple: Optional[float] = None
    index_strength: Optional[float] = None
    
    # Outcome
    win: Optional[bool] = None
    profit_loss_pct: Optional[float] = None
    r_multiple: Optional[float] = None


class TradeDatabase:
    """
    SQLite database for trade history
    
    Features:
    - Trade logging
    - Query trades for ML training
    - Win rate calculation
    - Performance analytics
    """
    
    def __init__(self, db_path: str = "trades.db"):
        """
        Initialize database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if not exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                engine_type TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                target_price REAL NOT NULL,
                exit_price REAL,
                exit_reason TEXT,
                market_score INTEGER,
                trend_score INTEGER,
                momentum_score INTEGER,
                volume_score INTEGER,
                risk_score INTEGER,
                final_probability REAL,
                sector TEXT,
                atr_multiple REAL,
                index_strength REAL,
                win BOOLEAN,
                profit_loss_pct REAL,
                r_multiple REAL
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON trades(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_engine_type 
            ON trades(engine_type)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_trade(self, trade: TradeRecord) -> int:
        """
        Log a trade to database
        
        Args:
            trade: TradeRecord to log
        
        Returns:
            ID of inserted trade
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert dataclass to dict
        trade_dict = asdict(trade)
        
        # Prepare SQL
        columns = ', '.join(trade_dict.keys())
        placeholders = ', '.join(['?' for _ in trade_dict])
        sql = f'INSERT INTO trades ({columns}) VALUES ({placeholders})'
        
        # Execute
        cursor.execute(sql, list(trade_dict.values()))
        trade_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return trade_id
    
    def update_trade_exit(self,
                         trade_id: int,
                         exit_price: float,
                         exit_reason: str,
                         win: bool,
                         profit_loss_pct: float,
                         r_multiple: float):
        """
        Update trade with exit information
        
        Args:
            trade_id: ID of trade to update
            exit_price: Exit price
            exit_reason: Reason for exit
            win: True if winning trade
            profit_loss_pct: P/L as percentage
            r_multiple: R-multiple achieved
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades
            SET exit_price = ?,
                exit_reason = ?,
                win = ?,
                profit_loss_pct = ?,
                r_multiple = ?
            WHERE id = ?
        ''', (exit_price, exit_reason, win, profit_loss_pct, r_multiple, trade_id))
        
        conn.commit()
        conn.close()
    
    def get_recent_trades(self, count: int = 200) -> List[TradeRecord]:
        """
        Get recent trades
        
        Args:
            count: Number of trades to fetch
        
        Returns:
            List of TradeRecords
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (count,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to TradeRecords
        trades = []
        for row in rows:
            trade = TradeRecord(
                timestamp=row[1],
                symbol=row[2],
                engine_type=row[3],
                setup_type=row[4],
                direction=row[5],
                entry_price=row[6],
                stop_loss=row[7],
                target_price=row[8],
                exit_price=row[9],
                exit_reason=row[10],
                market_score=row[11],
                trend_score=row[12],
                momentum_score=row[13],
                volume_score=row[14],
                risk_score=row[15],
                final_probability=row[16],
                sector=row[17],
                atr_multiple=row[18],
                index_strength=row[19],
                win=row[20],
                profit_loss_pct=row[21],
                r_multiple=row[22]
            )
            trades.append(trade)
        
        return trades
    
    def get_trades_dataframe(self, count: int = 200) -> pd.DataFrame:
        """
        Get trades as pandas DataFrame for ML training
        
        Args:
            count: Number of trades to fetch
        
        Returns:
            DataFrame with trade data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(count,))
        conn.close()
        
        return df
    
    def get_trade_count(self) -> int:
        """Get total number of trades in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM trades')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_win_rate(self, filters: Optional[Dict] = None) -> float:
        """
        Calculate win rate
        
        Args:
            filters: Optional filters (e.g., {'engine_type': 'MICRO'})
        
        Returns:
            Win rate as percentage (0-100)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query
        where_clause = "WHERE win IS NOT NULL"
        params = []
        
        if filters:
            for key, value in filters.items():
                where_clause += f" AND {key} = ?"
                params.append(value)
        
        query = f'''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins
            FROM trades
            {where_clause}
        '''
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        conn.close()
        
        total = result[0]
        wins = result[1]
        
        if total == 0:
            return 0.0
        
        return (wins / total) * 100
    
    def get_performance_stats(self) -> Dict:
        """Get overall performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as winning_trades,
                AVG(profit_loss_pct) as avg_profit_loss,
                AVG(r_multiple) as avg_r_multiple,
                MAX(profit_loss_pct) as max_win,
                MIN(profit_loss_pct) as max_loss
            FROM trades
            WHERE win IS NOT NULL
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        total = result[0] or 0
        wins = result[1] or 0
        
        return {
            'total_trades': total,
            'winning_trades': wins,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'avg_profit_loss_pct': result[2] or 0,
            'avg_r_multiple': result[3] or 0,
            'max_win_pct': result[4] or 0,
            'max_loss_pct': result[5] or 0
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Trade Database - v1.0")
    print("=" * 70)
    
    # Initialize database
    db = TradeDatabase("example_trades.db")
    
    print(f"\nTotal trades logged: {db.get_trade_count()}")
    
    # Example: Log a trade
    trade = TradeRecord(
        timestamp=datetime.now().isoformat(),
        symbol="RELIANCE.NS",
        engine_type="MICRO",
        setup_type="EMA Pullback",
        direction="LONG",
        entry_price=2450.0,
        stop_loss=2420.0,
        target_price=2478.0,
        market_score=75,
        trend_score=80,
        momentum_score=70,
        volume_score=65,
        risk_score=90,
        final_probability=75.5,
        sector="ENERGY",
        atr_multiple=1.2,
        index_strength=72.0
    )
    
    trade_id = db.log_trade(trade)
    print(f"\n✅ Logged trade ID: {trade_id}")
    
    # Get stats
    stats = db.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    
    # Clean up example database
    if os.path.exists("example_trades.db"):
        os.remove("example_trades.db")
