"""
Test all fixed functions
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

print("🧪 Testing ALL fixed functions...")
print("=" * 60)

from data.instrument_master import (
    load_instruments,
    get_option_keys,
    get_available_expiries,
    get_nearest_expiry,
    get_option_keys_around_price
)

# Test 1: Load instruments
print("\n🔍 Test 1: Load Instruments")
instruments = load_instruments()
print(f"✓ Loaded {len(instruments)} instruments")

# Test 2: Available expiries
print("\n🔍 Test 2: Available Expiries")
expiries = get_available_expiries("NIFTY")
print(f"✓ NIFTY expiries: {len(expiries)} dates")
print(f"  First 5: {expiries[:5]}")

# Test 3: Nearest expiry
print("\n🔍 Test 3: Nearest Expiry")
nearest = get_nearest_expiry("NIFTY")
print(f"✓ Nearest NIFTY expiry: {nearest}")

# Test 4: Get option keys
print("\n🔍 Test 4: Get Option Keys")
if expiries:
    test_expiry = expiries[0]
    keys = get_option_keys("NIFTY", test_expiry, max_keys=10)
    print(f"✓ Options for {test_expiry}: {len(keys)} keys")
    
    if keys:
        print("  Sample keys:")
        for key in keys[:3]:
            print(f"    - {key}")
        
        # Test option keys around price
        print("\n🔍 Test 5: Option Keys Around Price")
        spot_price = 27000  # Example spot price
        around_keys = get_option_keys_around_price(
            "NIFTY", test_expiry, spot_price, 
            num_strikes=5, max_keys=20
        )
        print(f"✓ Options around {spot_price}: {len(around_keys)} keys")
        
        if around_keys:
            print("  Sample keys:")
            for key in around_keys[:3]:
                print(f"    - {key}")
    else:
        print("❌ No option keys found!")
else:
    print("❌ No expiries found!")

# Test 6: Test multiple expiries
print("\n🔍 Test 6: Multiple Expiry Test")
test_results = {}
for expiry in expiries[:3]:  # Test first 3 expiries
    keys = get_option_keys("NIFTY", expiry, max_keys=5)
    test_results[expiry] = len(keys)
    status = "✅" if keys else "❌"
    print(f"  {status} {expiry}: {len(keys)} keys")

print("\n" + "=" * 60)
working = sum(1 for count in test_results.values() if count > 0)
print(f"Summary: {working}/{len(test_results)} expiries working")
print("💡 If all tests pass, your app should now work!")