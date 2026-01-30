"""
Test expiry date logic
"""
from datetime import datetime
from expiry_utils import get_trading_expiry, is_market_open

print("🧪 Testing Expiry Logic")
print("=" * 50)

# Test for NIFTY
print("\n🔍 Testing NIFTY expiry selection:")
nifty_expiry = get_trading_expiry("NIFTY")
print(f"Selected expiry: {nifty_expiry}")

# Show all available expiries
from data.instrument_master import get_available_expiries
nifty_expiries = get_available_expiries("NIFTY")
print(f"All NIFTY expiries: {nifty_expiries[:5]}...")  # Show first 5

# Test for BANKNIFTY
print("\n🔍 Testing BANKNIFTY expiry selection:")
banknifty_expiry = get_trading_expiry("BANKNIFTY")
print(f"Selected expiry: {banknifty_expiry}")

# Show market status
market_open = is_market_open()
print(f"\n📊 Market Status: {'OPEN 🟢' if market_open else 'CLOSED 🔴'}")

# Today's date
today = datetime.now().strftime('%Y-%m-%d')
print(f"Today's date: {today}")

# Check if today is an expiry day
print(f"\n📅 Today is expiry day: {'YES ⚠️' if today in nifty_expiries else 'NO ✅'}")

print("\n" + "=" * 50)
print("💡 Since today (2026-01-27) is expiry day and market is closed,")
print("   the system should select the NEXT expiry (2026-02-03).")