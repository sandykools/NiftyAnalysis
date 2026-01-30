"""
Quick test of fixed instrument master
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

print("🔧 Testing fixed instrument master...")

# Test the new functions
from data.instrument_master import (
    load_instruments, 
    get_available_expiries,
    get_option_keys,
    test_instrument_master
)

# Run the built-in test
test_instrument_master()

# Additional test
print("\n🔍 Additional test:")
instruments = load_instruments()
print(f"Total instruments: {len(instruments)}")

# Check what's in the first NIFTY instrument
for inst in instruments:
    if isinstance(inst, dict) and inst.get('name') == 'NIFTY':
        print("\nFirst NIFTY instrument found:")
        print(f"  Name: {inst.get('name')}")
        print(f"  Type: {inst.get('instrument_type')}")
        print(f"  Trading Symbol: {inst.get('trading_symbol')}")
        print(f"  Strike: {inst.get('strike_price')}")
        print(f"  Expiry timestamp: {inst.get('expiry')}")
        print(f"  Instrument Key: {inst.get('instrument_key')}")
        break

# Get available expiries
expiries = get_available_expiries("NIFTY")
print(f"\n📅 NIFTY expiries: {expiries}")

# Try to get keys for each expiry
for expiry in expiries[:2]:  # Try first 2 expiries
    print(f"\n🔑 Testing expiry: {expiry}")
    keys = get_option_keys("NIFTY", expiry, max_keys=5)
    print(f"  Found {len(keys)} keys")
    if keys:
        for key in keys[:3]:
            print(f"    - {key}")