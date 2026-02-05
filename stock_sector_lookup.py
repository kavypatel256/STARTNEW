"""
STOCK SECTOR LOOKUP MODULE
Shared mapping of Indian stocks to their sectors

Used by:
- interactive_analyzer.py
- telegram_bot.py

Author: AI Trading System
Version: 1.0
"""

# NSE Stock to Sector Mapping (Popular stocks)
STOCK_SECTORS = {
    # Banking
    'HDFCBANK': 'BANKING', 'ICICIBANK': 'BANKING', 'SBIN': 'BANKING', 
    'AXISBANK': 'BANKING', 'KOTAKBANK': 'BANKING', 'INDUSINDBK': 'BANKING',
    'BANDHANBNK': 'BANKING', 'FEDERALBNK': 'BANKING', 'IDFCFIRSTB': 'BANKING',
    
    # IT
    'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT',
    'TECHM': 'IT', 'LTIM': 'IT', 'COFORGE': 'IT', 'MPHASIS': 'IT',
    
    # Auto
    'TATAMOTORS': 'AUTO', 'M&M': 'AUTO', 'MARUTI': 'AUTO', 'BAJAJ-AUTO': 'AUTO',
    'EICHERMOT': 'AUTO', 'HEROMOTOCORP': 'AUTO', 'ASHOKLEY': 'AUTO',
    
    # Pharma
    'SUNPHARMA': 'PHARMA', 'DRREDDY': 'PHARMA', 'CIPLA': 'PHARMA', 
    'DIVISLAB': 'PHARMA', 'BIOCON': 'PHARMA', 'AUROPHARMA': 'PHARMA',
    
    # FMCG
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 
    'BRITANNIA': 'FMCG', 'DABUR': 'FMCG', 'MARICO': 'FMCG',
    
    # Energy / Oil & Gas
    'RELIANCE': 'ENERGY', 'ONGC': 'ENERGY', 'BPCL': 'ENERGY', 
    'IOC': 'ENERGY', 'GAIL': 'ENERGY', 'COALINDIA': 'ENERGY',
    
    # Metals
    'TATASTEEL': 'METALS', 'HINDALCO': 'METALS', 'JSWSTEEL': 'METALS',
    'SAIL': 'METALS', 'VEDL': 'METALS', 'NATIONALUM': 'METALS',
    
    # Infra / Construction
    'LT': 'INFRA', 'ADANIPORTS': 'INFRA', 'ADANIENT': 'INFRA',
    
    # Finance (NBFCs)
    'BAJFINANCE': 'FINANCE', 'BAJAJFINSV': 'FINANCE', 'CHOLAFIN': 'FINANCE',
    'SBILIFE': 'FINANCE', 'HDFCLIFE': 'FINANCE', 'ICICIPRULI': 'FINANCE',
    
    # Realty
    'DLF': 'REALTY', 'GODREJPROP': 'REALTY', 'OBEROIRLTY': 'REALTY',
    
    # Cement
    'ULTRACEMCO': 'CEMENT', 'GRASIM': 'CEMENT', 'SHREECEM': 'CEMENT',
    
    # Telecom
    'BHARTIARTL': 'TELECOM', 'INDUSINDBK': 'TELECOM',
    
    # Paints
    'ASIANPAINT': 'PAINTS', 'BERGER': 'PAINTS',
}


def detect_sector(symbol: str) -> str:
    """
    Auto-detect sector from stock symbol
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
    
    Returns:
        Sector name or 'UNKNOWN'
    
    Examples:
        >>> detect_sector('RELIANCE')
        'ENERGY'
        >>> detect_sector('TCS')
        'IT'
        >>> detect_sector('RANDOMSTOCK')
        'UNKNOWN'
    """
    # Remove common suffixes
    clean_symbol = symbol.upper().replace('.NS', '').replace('.BO', '')
    
    # Direct lookup
    if clean_symbol in STOCK_SECTORS:
        return STOCK_SECTORS[clean_symbol]
    
    # Fuzzy matching for common variations
    for stock, sector in STOCK_SECTORS.items():
        if clean_symbol in stock or stock in clean_symbol:
            return sector
    
    return 'UNKNOWN'


def get_all_sectors():
    """Get list of all unique sectors"""
    return sorted(set(STOCK_SECTORS.values()))


def get_stocks_by_sector(sector: str):
    """Get all stocks in a specific sector"""
    return [stock for stock, sect in STOCK_SECTORS.items() if sect == sector]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Stock Sector Lookup - v1.0")
    print("=" * 70)
    
    # Test some stocks
    test_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'TATAMOTORS', 'UNKNOWN']
    
    print("\n📊 Sector Detection Test:")
    for stock in test_stocks:
        sector = detect_sector(stock)
        print(f"   {stock:15} → {sector}")
    
    print(f"\n📋 Total Stocks in Database: {len(STOCK_SECTORS)}")
    print(f"📂 Total Sectors: {len(get_all_sectors())}")
    
    print("\n🏢 All Sectors:")
    for sector in get_all_sectors():
        stocks = get_stocks_by_sector(sector)
        print(f"   {sector:12} ({len(stocks)} stocks)")
