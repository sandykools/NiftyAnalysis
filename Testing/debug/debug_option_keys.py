"""
Debug why get_option_keys returns 0 keys
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

print("🔍 Debugging get_option_keys...")

# Test directly
from data.instrument_master import load_instruments, get_option_keys

# Load instruments
instruments = load_instruments()
print(f"Total instruments: {len(instruments)}")

# Find NIFTY options for 2026-02-03
print("\n🔍 Looking for NIFTY options with expiry 2026-02-03...")

# First, let's check what instruments we have
nifty_options = []
for inst in instruments:
    if isinstance(inst, dict):
        name = inst.get('name', '')
        inst_type = inst.get('instrument_type', '')
        expiry_ts = inst.get('expiry')
        
        if name == 'NIFTY' and inst_type in ['CE', 'PE'] and expiry_ts:
            # Convert timestamp to date
            from datetime import datetime
            expiry_date = datetime.fromtimestamp(expiry_ts / 1000)
            expiry_str = expiry_date.strftime('%Y-%m-%d')
            
            if expiry_str == '2026-02-03':
                nifty_options.append(inst)

print(f"Found {len(nifty_options)} NIFTY options for 2026-02-03")

if nifty_options:
    print("\n📋 Sample options:")
    for i, opt in enumerate(nifty_options[:5]):
        print(f"{i+1}. {opt.get('trading_symbol', 'N/A')}")
        print(f"   Key: {opt.get('instrument_key', 'N/A')}")
        print(f"   Strike: {opt.get('strike_price', 'N/A')}")
        print(f"   Type: {opt.get('instrument_type', 'N/A')}")
        print()

# Now test the get_option_keys function
print("\n🧪 Testing get_option_keys function...")
keys = get_option_keys("NIFTY", "2026-02-03", max_keys=20)
print(f"get_option_keys returned: {len(keys)} keys")

if keys:
    print("Keys returned:")
    for key in keys[:5]:
        print(f"  - {key}")
else:
    print("❌ No keys returned!")
    
    # Let's debug the function
    print("\n🐛 Debugging get_option_keys logic...")
    
    # Manually check what the function does
    from datetime import datetime
    
    # Convert expiry string to timestamp
    expiry_dt = datetime.strptime("2026-02-03", "%Y-%m-%d")
    expiry_ts = int(expiry_dt.timestamp() * 1000)
    print(f"Expiry timestamp for 2026-02-03: {expiry_ts}")
    
    # Check how many instruments match
    matches = []
    for inst in instruments:
        if isinstance(inst, dict):
            name = inst.get('name', '')
            inst_type = inst.get('instrument_type', '')
            inst_expiry = inst.get('expiry')
            
            if name == 'NIFTY' and inst_type in ['CE', 'PE']:
                if inst_expiry == expiry_ts:
                    matches.append(inst)
    
    print(f"Direct matching found {len(matches)} instruments")
    
    if matches:
        print("\nMatching instruments:")
        for i, match in enumerate(matches[:3]):
            print(f"{i+1}. {match.get('trading_symbol', 'N/A')}")
            print(f"   Expiry timestamp: {match.get('expiry')}")
            print(f"   Key: {match.get('instrument_key')}")