"""
Fixed version of instrument master
"""
from typing import List, Dict, Any
import gzip
import json
from datetime import datetime
from pathlib import Path

def load_instruments() -> List[Dict[str, Any]]:
    """Load instruments from compressed JSON file."""
    instrument_file = Path("data/instruments.json.gz")
    if not instrument_file.exists():
        return []
    
    try:
        with gzip.open(instrument_file, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading instruments: {e}")
        return []

def get_option_keys(underlying: str, expiry: str, max_keys: int = 200) -> List[str]:
    """
    Get option instrument keys for given underlying and expiry.
    
    Args:
        underlying: Underlying symbol (e.g., 'NIFTY')
        expiry: Expiry date in 'YYYY-MM-DD' format
        max_keys: Maximum number of keys to return
    
    Returns:
        List of instrument keys
    """
    instruments = load_instruments()
    
    if not instruments:
        print("Warning: No instruments loaded")
        return []
    
    # Convert expiry string to timestamp
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        # Start and end of day
        expiry_start = int(expiry_dt.replace(hour=0, minute=0, second=0).timestamp() * 1000)
        expiry_end = int(expiry_dt.replace(hour=23, minute=59, second=59).timestamp() * 1000)
    except ValueError as e:
        print(f"Invalid expiry format: {expiry}. Expected YYYY-MM-DD")
        return []
    
    option_keys = []
    
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        
        # Check if it's an option
        instrument_type = instrument.get('instrument_type')
        if instrument_type not in ['CE', 'PE']:
            continue
        
        # Check underlying - try multiple possible fields
        name = instrument.get('name', '')
        underlying_symbol = instrument.get('underlying_symbol', '')
        asset_symbol = instrument.get('asset_symbol', '')
        
        # Accept if any of these match
        is_correct_underlying = (
            name == underlying or 
            underlying_symbol == underlying or 
            asset_symbol == underlying
        )
        
        if not is_correct_underlying:
            continue
        
        # Check expiry (within day range to handle timestamp precision)
        instrument_expiry = instrument.get('expiry')
        if not instrument_expiry:
            continue
            
        if expiry_start <= instrument_expiry <= expiry_end:
            instrument_key = instrument.get('instrument_key')
            if instrument_key:
                option_keys.append(instrument_key)
    
    # Sort by strike price (extract from trading symbol or use strike_price field)
    def get_strike_price(key_info: Dict) -> float:
        # Try to get from strike_price field
        strike = key_info.get('strike_price')
        if strike:
            return float(strike)
        
        # Try to extract from trading_symbol
        trading_symbol = key_info.get('trading_symbol', '')
        import re
        match = re.search(r'(\d+)', trading_symbol)
        if match:
            return float(match.group(1))
        
        return 0.0
    
    # Get full instrument info for sorting
    option_instruments = []
    for key in option_keys:
        for inst in instruments:
            if inst.get('instrument_key') == key:
                option_instruments.append(inst)
                break
    
    # Sort by strike price
    option_instruments.sort(key=get_strike_price)
    
    # Return instrument keys
    return [inst['instrument_key'] for inst in option_instruments[:max_keys]]

def get_nearest_expiry(underlying: str) -> str:
    """Get nearest expiry date for an underlying."""
    instruments = load_instruments()
    
    if not instruments:
        return ""
    
    # Find all expiry dates for this underlying
    expiry_dates = set()
    
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        
        # Check if it's the right underlying
        name = instrument.get('name', '')
        if name != underlying:
            continue
        
        # Check if it's an option
        if instrument.get('instrument_type') not in ['CE', 'PE']:
            continue
        
        expiry_ts = instrument.get('expiry')
        if expiry_ts:
            try:
                expiry_date = datetime.fromtimestamp(expiry_ts / 1000)
                expiry_dates.add(expiry_date.strftime('%Y-%m-%d'))
            except:
                pass
    
    if not expiry_dates:
        return ""
    
    # Return the nearest future expiry
    today = datetime.now().strftime('%Y-%m-%d')
    future_expiries = [d for d in sorted(expiry_dates) if d >= today]
    
    return future_expiries[0] if future_expiries else sorted(expiry_dates)[-1]