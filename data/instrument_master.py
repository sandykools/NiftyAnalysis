"""
Enhanced Instrument Master with robust key generation.
Handles multiple underlying formats and expiry matching.
"""

from typing import List, Dict, Any, Optional
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import warnings

# ==============================
# INSTRUMENT LOADING
# ==============================

def load_instruments() -> List[Dict[str, Any]]:
    """
    Load instruments from compressed JSON file.
    
    Returns:
        List of instrument dictionaries
    """
    instrument_file = Path("data/instruments.json.gz")
    
    if not instrument_file.exists():
        warnings.warn(f"Instrument file not found: {instrument_file}")
        return []
    
    try:
        with gzip.open(instrument_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            print(f"✓ Loaded {len(data)} instruments")
            return data
        else:
            warnings.warn(f"Unexpected data format: {type(data)}")
            return []
            
    except Exception as e:
        warnings.warn(f"Error loading instruments: {e}")
        return []

def save_instruments(instruments: List[Dict[str, Any]]) -> bool:
    """Save instruments to compressed JSON file."""
    try:
        instrument_file = Path("data/instruments.json.gz")
        instrument_file.parent.mkdir(parents=True, exist_ok=True)
        
        with gzip.open(instrument_file, 'wt', encoding='utf-8') as f:
            json.dump(instruments, f)
            
        print(f"✓ Saved {len(instruments)} instruments")
        return True
    except Exception as e:
        warnings.warn(f"Error saving instruments: {e}")
        return False

# ==============================
# OPTION KEY GENERATION
# ==============================

def get_option_keys(
    underlying: str, 
    expiry: str, 
    max_keys: int = 200,
    instrument_type: Optional[str] = None
) -> List[str]:
    """
    Get option instrument keys for given underlying and expiry.
    Fixed: Matches by DATE only, not exact timestamp.
    
    Args:
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')
        expiry: Expiry date in 'YYYY-MM-DD' format
        max_keys: Maximum number of keys to return
        instrument_type: Filter by 'CE', 'PE', or None for both
    
    Returns:
        List of instrument keys
    """
    instruments = load_instruments()
    
    if not instruments:
        print("⚠️ No instruments loaded")
        return []
    
    # Parse target expiry date
    try:
        target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"❌ Invalid expiry format: {expiry}. Expected YYYY-MM-DD")
        return []
    
    option_keys = []
    
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        
        # Check if it's an option
        inst_type = instrument.get('instrument_type')
        if inst_type not in ['CE', 'PE']:
            continue
        
        # Filter by instrument type if specified
        if instrument_type and inst_type != instrument_type:
            continue
        
        # EXACT match for name
        name = instrument.get('name', '')
        if name != underlying:
            continue
        
        # Check expiry - convert timestamp to date
        instrument_expiry_ts = instrument.get('expiry')
        if not instrument_expiry_ts:
            continue
        
        # Convert timestamp to date (ignoring time component)
        try:
            instrument_expiry_date = datetime.fromtimestamp(instrument_expiry_ts / 1000).date()
        except:
            continue
        
        # Match by date only (not exact timestamp)
        if instrument_expiry_date != target_date:
            continue
        
        instrument_key = instrument.get('instrument_key')
        if instrument_key:
            option_keys.append({
                'key': instrument_key,
                'strike_price': float(instrument.get('strike_price', 0)),
                'instrument_type': inst_type,
                'instrument': instrument
            })
    
    if not option_keys:
        print(f"⚠️ No {underlying} options found for expiry {expiry}")
        return []
    
    # Sort by strike price
    option_keys.sort(key=lambda x: x['strike_price'])
    
    print(f"✅ Found {len(option_keys)} {underlying} options for {expiry}")
    
    # Return just the keys
    return [item['key'] for item in option_keys[:max_keys]]

def get_option_keys_around_price(
    underlying: str,
    expiry: str,
    spot_price: float,
    num_strikes: int = 10,
    max_keys: int = 100
) -> List[str]:
    """
    Get option keys around current spot price.
    Fixed: Matches by DATE only, not exact timestamp.
    
    Args:
        underlying: Underlying symbol
        expiry: Expiry date in 'YYYY-MM-DD' format
        spot_price: Current spot price
        num_strikes: Number of strikes to get on each side
        max_keys: Maximum total keys to return
    
    Returns:
        List of instrument keys sorted by proximity to spot
    """
    instruments = load_instruments()
    
    if not instruments:
        return []
    
    # Parse target expiry date
    try:
        target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"❌ Invalid expiry format: {expiry}. Expected YYYY-MM-DD")
        return []
    
    all_options = []
    
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        
        # Check if it's an option
        if instrument.get('instrument_type') not in ['CE', 'PE']:
            continue
        
        # Check underlying
        name = instrument.get('name', '')
        if name != underlying:
            continue
        
        # Check expiry - convert timestamp to date
        instrument_expiry_ts = instrument.get('expiry')
        if not instrument_expiry_ts:
            continue
        
        # Convert timestamp to date (ignoring time component)
        try:
            instrument_expiry_date = datetime.fromtimestamp(instrument_expiry_ts / 1000).date()
        except:
            continue
        
        # Match by date only (not exact timestamp)
        if instrument_expiry_date != target_date:
            continue
        
        # Calculate distance from spot
        strike_price = float(instrument.get('strike_price', 0))
        distance = abs(strike_price - spot_price)
        
        instrument_key = instrument.get('instrument_key')
        if instrument_key:
            all_options.append({
                'key': instrument_key,
                'strike_price': strike_price,
                'distance': distance,
                'instrument_type': instrument.get('instrument_type'),
                'instrument': instrument
            })
    
    if not all_options:
        print(f"⚠️ No {underlying} options found for expiry {expiry} around spot {spot_price}")
        return []
    
    # Sort by distance from spot
    all_options.sort(key=lambda x: x['distance'])
    
    # Get closest strikes
    closest_options = all_options[:num_strikes * 2]  # CE and PE for each strike
    
    # Sort by strike price for consistency
    closest_options.sort(key=lambda x: x['strike_price'])
    
    print(f"✅ Found {len(closest_options[:max_keys])} {underlying} options around spot {spot_price} for {expiry}")
    
    return [item['key'] for item in closest_options[:max_keys]]


def get_available_expiries(underlying: str) -> List[str]:
    """
    Get all available expiry dates for an underlying.
    Fixed: Uses date-only matching.
    
    Args:
        underlying: Underlying symbol
    
    Returns:
        List of expiry dates in 'YYYY-MM-DD' format
    """
    instruments = load_instruments()
    
    if not instruments:
        return []
    
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
                # Convert timestamp to date only
                expiry_date = datetime.fromtimestamp(expiry_ts / 1000).date()
                expiry_dates.add(expiry_date.strftime('%Y-%m-%d'))
            except:
                pass
    
    return sorted(expiry_dates)

def get_nearest_expiry(underlying: str, min_days: int = 0) -> str:
    """
    Get nearest expiry date for an underlying.
    Fixed: Uses date-only comparison.
    
    Args:
        underlying: Underlying symbol
        min_days: Minimum days until expiry
    
    Returns:
        Expiry date in 'YYYY-MM-DD' format, or empty string if none found
    """
    expiry_dates = get_available_expiries(underlying)
    
    if not expiry_dates:
        return ""
    
    # Get today's date (without time)
    today = datetime.now().date()
    
    # Add min_days
    from datetime import timedelta
    target_date = today + timedelta(days=min_days)
    target_str = target_date.strftime('%Y-%m-%d')
    
    # Find first expiry on or after target date
    for expiry in expiry_dates:
        if expiry >= target_str:
            return expiry
    
    # If no future expiry, return the last one
    return expiry_dates[-1]

def get_weekly_expiry(underlying: str) -> str:
    """
    Get nearest Thursday expiry (standard weekly expiry).
    
    Args:
        underlying: Underlying symbol
    
    Returns:
        Expiry date in 'YYYY-MM-DD' format
    """
    # Find the nearest Thursday
    today = datetime.now()
    
    # Thursday is weekday 3 (Monday=0)
    days_until_thursday = (3 - today.weekday()) % 7
    if days_until_thursday == 0:
        days_until_thursday = 7  # If today is Thursday, get next Thursday
    
    next_thursday = today + timedelta(days=days_until_thursday)
    
    # Check if this Thursday is available
    thursday_str = next_thursday.strftime('%Y-%m-%d')
    available_expiries = get_available_expiries(underlying)
    
    if thursday_str in available_expiries:
        return thursday_str
    
    # If not, get the nearest available expiry
    return get_nearest_expiry(underlying)

# ==============================
# INSTRUMENT LOOKUP
# ==============================

def get_instrument_by_key(instrument_key: str) -> Optional[Dict[str, Any]]:
    """Get instrument details by instrument key."""
    instruments = load_instruments()
    
    for instrument in instruments:
        if isinstance(instrument, dict) and instrument.get('instrument_key') == instrument_key:
            return instrument
    
    return None

def get_instruments_by_type(
    instrument_type: str,
    underlying: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get instruments by type.
    
    Args:
        instrument_type: 'EQ', 'CE', 'PE', 'FUT', etc.
        underlying: Optional underlying symbol filter
    
    Returns:
        List of matching instruments
    """
    instruments = load_instruments()
    
    filtered = []
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        
        if instrument.get('instrument_type') != instrument_type:
            continue
        
        if underlying:
            name = instrument.get('name', '')
            if name != underlying:
                continue
        
        filtered.append(instrument)
    
    return filtered

# ==============================
# BATCH OPERATIONS
# ==============================

def get_batch_option_keys(
    underlying: str,
    expiries: List[str],
    max_per_expiry: int = 50
) -> Dict[str, List[str]]:
    """
    Get option keys for multiple expiries.
    
    Args:
        underlying: Underlying symbol
        expiries: List of expiry dates
        max_per_expiry: Maximum keys per expiry
    
    Returns:
        Dictionary mapping expiry to list of keys
    """
    result = {}
    
    for expiry in expiries:
        keys = get_option_keys(underlying, expiry, max_keys=max_per_expiry)
        if keys:
            result[expiry] = keys
    
    return result

def get_instrument_keys_for_strikes(
    underlying: str,
    expiry: str,
    strikes: List[float],
    option_types: List[str] = ['CE', 'PE']
) -> List[str]:
    """
    Get instrument keys for specific strikes and option types.
    
    Args:
        underlying: Underlying symbol
        expiry: Expiry date
        strikes: List of strike prices
        option_types: List of option types
    
    Returns:
        List of instrument keys
    """
    instruments = load_instruments()
    
    keys = []
    
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        
        inst_type = instrument.get('instrument_type')
        if inst_type not in option_types:
            continue
        
        # Check underlying
        name = instrument.get('name', '')
        if name != underlying:
            continue
        
        # Check expiry
        instrument_expiry = instrument.get('expiry')
        if not instrument_expiry:
            continue
        
        # Convert expiry string to timestamp
        try:
            expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
            expiry_ts = int(expiry_dt.timestamp() * 1000)
        except:
            continue
        
        if instrument_expiry != expiry_ts:
            continue
        
        # Check strike
        strike_price = float(instrument.get('strike_price', 0))
        if strike_price not in strikes:
            continue
        
        instrument_key = instrument.get('instrument_key')
        if instrument_key:
            keys.append(instrument_key)
    
    return keys

# ==============================
# UTILITIES
# ==============================

def print_instrument_summary():
    """Print summary of loaded instruments."""
    instruments = load_instruments()
    
    if not instruments:
        print("No instruments loaded")
        return
    
    print(f"Total instruments: {len(instruments)}")
    
    # Count by type
    type_counts = {}
    underlying_counts = {}
    
    for instrument in instruments:
        if not isinstance(instrument, dict):
            continue
        
        inst_type = instrument.get('instrument_type', 'UNKNOWN')
        type_counts[inst_type] = type_counts.get(inst_type, 0) + 1
        
        name = instrument.get('name', 'UNKNOWN')
        underlying_counts[name] = underlying_counts.get(name, 0) + 1
    
    print("\nBy instrument type:")
    for inst_type, count in sorted(type_counts.items()):
        print(f"  {inst_type}: {count}")
    
    print("\nTop underlying symbols:")
    for name, count in sorted(underlying_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name}: {count}")

# ==============================
# TEST FUNCTION
# ==============================

def test_instrument_master():
    """Test the instrument master functions."""
    print("🧪 Testing Instrument Master...")
    
    # Load instruments
    instruments = load_instruments()
    print(f"✓ Loaded {len(instruments)} instruments")
    
    # Get available expiries for NIFTY
    nifty_expiries = get_available_expiries("NIFTY")
    print(f"✓ NIFTY expiries: {nifty_expiries}")
    
    if nifty_expiries:
        # Get nearest expiry
        nearest = get_nearest_expiry("NIFTY")
        print(f"✓ Nearest NIFTY expiry: {nearest}")
        
        # Get option keys for nearest expiry
        if nearest:
            keys = get_option_keys("NIFTY", nearest, max_keys=10)
            print(f"✓ Got {len(keys)} option keys for {nearest}")
            
            if keys:
                print("  Sample keys:")
                for key in keys[:3]:
                    print(f"    - {key}")
                
                # Get instrument details
                instrument = get_instrument_by_key(keys[0])
                if instrument:
                    print(f"  First instrument: {instrument.get('trading_symbol', 'N/A')}")
    
    # Test BANKNIFTY
    banknifty_expiries = get_available_expiries("BANKNIFTY")
    print(f"✓ BANKNIFTY expiries: {banknifty_expiries}")
    
    print("\n✅ Instrument master test complete")
# ==============================
# EXPIRY MANAGEMENT
# ==============================

def get_next_available_expiry(underlying: str) -> str:
    """
    Get the next available expiry date (excluding today if market closed).
    
    Args:
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')
    
    Returns:
        Next available expiry date in 'YYYY-MM-DD' format
    """
    from datetime import datetime
    
    # Get all available expiries
    expiries = get_available_expiries(underlying)
    if not expiries:
        return ""
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if market is open (for testing, you can hardcode this)
    # For now, let's assume if today is expiry day, we should use next expiry
    market_open = False  # You can set this based on actual market hours
    
    if market_open and today in expiries:
        # Market is open and today is an expiry - use it
        return today
    else:
        # Market closed or today not available - get next expiry
        for expiry in sorted(expiries):
            if expiry > today:
                return expiry
        
        # If no future expiry, return the last one
        return expiries[-1]
# Run test if executed directly
if __name__ == "__main__":
    test_instrument_master()