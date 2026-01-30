"""
Utility functions for expiry date selection
"""
from datetime import datetime, time
from data.instrument_master import get_available_expiries

def is_market_open() -> bool:
    """
    Check if market is currently open.
    NSE market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    """
    now = datetime.now()
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    
    # Check time
    market_open = time(9, 15)  # 9:15 AM IST
    market_close = time(15, 30)  # 3:30 PM IST
    
    current_time = now.time()
    return market_open <= current_time <= market_close

def get_trading_expiry(underlying: str) -> str:
    """
    Get appropriate expiry date for trading.
    - If market is open and today is expiry: use today
    - If market is closed and today is expiry: use next expiry
    - Otherwise: use nearest expiry
    """
    expiries = get_available_expiries(underlying)
    if not expiries:
        return ""
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if today is an expiry day
    if today in expiries:
        if is_market_open():
            # Market is open on expiry day - use today
            return today
        else:
            # Market closed on expiry day - use next expiry
            for expiry in sorted(expiries):
                if expiry > today:
                    return expiry
            return expiries[-1]  # Fallback to last expiry
    else:
        # Today is not expiry day - get nearest future expiry
        for expiry in sorted(expiries):
            if expiry >= today:
                return expiry
        return expiries[-1]  # Fallback to last expiry

def get_expiry_for_backtesting(underlying: str, reference_date: str = None) -> str:
    """
    Get expiry for backtesting (can use historical expiries).
    
    Args:
        underlying: Underlying symbol
        reference_date: Reference date in 'YYYY-MM-DD' format (default: today)
    
    Returns:
        Expiry date for backtesting
    """
    expiries = get_available_expiries(underlying)
    if not expiries:
        return ""
    
    if reference_date:
        ref_date = reference_date
    else:
        ref_date = datetime.now().strftime('%Y-%m-%d')
    
    # Find the nearest expiry on or after reference date
    for expiry in sorted(expiries):
        if expiry >= ref_date:
            return expiry
    
    return expiries[-1]