"""
FIXED version - matches by date, not exact timestamp
"""
from typing import List, Dict, Any, Optional
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path

def load_instruments() -> List[Dict[str, Any]]:
    """Load instruments from compressed JSON file."""
    instrument_file = Path("data/instruments.json.gz")
    
    if not instrument_file.exists():
        print(f"❌ Instrument file not found: {instrument_file}")
        return []
    
    try:
        with gzip.open(instrument_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        else:
            print(f"⚠️ Unexpected data format: {type(data)}")
            return []
            
    except Exception as e:
        print(f"❌ Error loading instruments: {e}")
        return []

def get_option_keys_fixed(
    underlying: str, 
    expiry: str, 
    max_keys: int = 200,
    debug: bool = False
) -> List[str]:
    """
    Get option instrument keys for given underlying and expiry.
    Matches by DATE only, not exact timestamp.
    
    Args:
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')
        expiry: Expiry date in 'YYYY-MM-DD' format
        max_keys: Maximum number of keys to return
        debug: Enable debug logging
    
    Returns:
        List of instrument keys
    """
    instruments = load_instruments()
    
    if not instruments:
        if debug: print("⚠️ No instruments loaded")
        return []
    
    # Parse target expiry date
    try:
        target_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        if debug: print(f"Looking for options expiring on: {target_date}")
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
        
        # Match by date only
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
        if debug: print(f"⚠️ No {underlying} options found for expiry {expiry}")
        return []
    
    # Sort by strike price
    option_keys.sort(key=lambda x: x['strike_price'])
    
    if debug: print(f"✅ Found {len(option_keys)} {underlying} options for {expiry}")
    
    # Return just the keys
    return [item['key'] for item in option_keys[:max_keys]]

def get_available_expiries_fixed(underlying: str) -> List[str]:
    """
    Get all available expiry dates for an underlying.
    Fixed version that works with date-only matching.
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
                expiry_date = datetime.fromtimestamp(expiry_ts / 1000).date()
                expiry_dates.add(expiry_date.strftime('%Y-%m-%d'))
            except:
                pass
    
    return sorted(expiry_dates)

def test_fixed_functions():
    """Test the fixed functions"""
    print("🧪 Testing FIXED functions...")
    
    # Test get_available_expiries_fixed
    print("\n🔍 Available expiries (fixed):")
    expiries = get_available_expiries_fixed("NIFTY")
    print(f"NIFTY expiries: {expiries[:5]}...")
    
    # Test get_option_keys_fixed
    if expiries:
        test_expiry = expiries[0] if len(expiries) > 0 else "2026-02-03"
        print(f"\n🔍 Testing option keys for {test_expiry} (fixed):")
        keys = get_option_keys_fixed("NIFTY", test_expiry, max_keys=10, debug=True)
        print(f"Found {len(keys)} keys")
        
        if keys:
            print("Sample keys:")
            for key in keys[:3]:
                print(f"  - {key}")
        else:
            print("❌ Still no keys found!")

if __name__ == "__main__":
    test_fixed_functions()