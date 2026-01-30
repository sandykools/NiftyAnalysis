"""
Test all expiries to find one that works
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.instrument_master import get_available_expiries, get_option_keys

print("🔍 Testing all NIFTY expiries...")
expiries = get_available_expiries("NIFTY")

working_expiries = []
for expiry in expiries:
    keys = get_option_keys("NIFTY", expiry, max_keys=5, debug=False)
    if keys:
        working_expiries.append((expiry, len(keys)))
        print(f"✅ {expiry}: {len(keys)} keys found")
    else:
        print(f"❌ {expiry}: No keys found")

if working_expiries:
    print(f"\n🎯 Working expiries: {len(working_expiries)}/{len(expiries)}")
    for expiry, count in working_expiries:
        print(f"  - {expiry}: {count} keys")
    
    # Get sample keys from first working expiry
    first_expiry = working_expiries[0][0]
    keys = get_option_keys("NIFTY", first_expiry, max_keys=5, debug=True)
    print(f"\n🔑 Sample keys for {first_expiry}:")
    for key in keys[:3]:
        print(f"  - {key}")
else:
    print("\n❌ No working expiries found!")
    
    # Try with debug on for first expiry
    if expiries:
        print(f"\n🐛 Debugging first expiry: {expiries[0]}")
        get_option_keys("NIFTY", expiries[0], max_keys=5, debug=True)