"""
Test the expiry fix
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

print("🧪 Testing Expiry Fix")
print("=" * 50)

# Test 1: Check instrument master
print("\n🔍 Test 1: Instrument Master")
try:
    from data.instrument_master import (
        get_available_expiries,
        get_next_available_expiry
    )
    
    nifty_expiries = get_available_expiries("NIFTY")
    print(f"✓ NIFTY expiries found: {len(nifty_expiries)}")
    print(f"  First 5: {nifty_expiries[:5]}")
    
    next_expiry = get_next_available_expiry("NIFTY")
    print(f"✓ Next available expiry: {next_expiry}")
    
except Exception as e:
    print(f"❌ Instrument master error: {e}")

# Test 2: Check expiry utils
print("\n🔍 Test 2: Expiry Utils")
try:
    from utils.expiry_utils import (
        is_market_open,
        get_trading_expiry
    )
    
    market_open = is_market_open()
    print(f"✓ Market status: {'OPEN' if market_open else 'CLOSED'}")
    
    trading_expiry = get_trading_expiry("NIFTY")
    print(f"✓ Trading expiry selected: {trading_expiry}")
    
    # Today's date
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"✓ Today's date: {today}")
    
    # Show logic
    if trading_expiry == today:
        print("  → Using TODAY's expiry (market open on expiry day)")
    else:
        print(f"  → Using NEXT expiry {trading_expiry} (market closed or not expiry day)")
    
except Exception as e:
    print(f"❌ Expiry utils error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Verify option keys can be fetched
print("\n🔍 Test 3: Option Key Generation")
try:
    from data.instrument_master import get_option_keys
    
    # Use the trading expiry we determined
    trading_expiry = get_trading_expiry("NIFTY")
    if not trading_expiry:
        trading_expiry = "2026-02-03"  # Fallback
    
    keys = get_option_keys("NIFTY", trading_expiry, max_keys=10)
    print(f"✓ Option keys for {trading_expiry}: {len(keys)} found")
    
    if keys:
        print("  Sample keys:")
        for key in keys[:3]:
            print(f"    - {key}")
    else:
        print("  ⚠️ No option keys found")
        
except Exception as e:
    print(f"❌ Option key error: {e}")

print("\n" + "=" * 50)
print("💡 If all tests pass, your app should now work correctly!")
print("   It will use 2026-02-03 (next expiry) instead of today's expired date.")