"""
Debug timestamp mismatch
"""
import sys
from pathlib import Path
import json
import gzip
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

print("🔍 Debugging timestamp mismatch...")

# Load the data directly
with gzip.open("data/instruments.json.gz", 'rt', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total instruments: {len(data)}")

# Find NIFTY options for 2026-02-03
nifty_options = []
for inst in data:
    if isinstance(inst, dict):
        name = inst.get('name', '')
        inst_type = inst.get('instrument_type', '')
        expiry_ts = inst.get('expiry')
        
        if name == 'NIFTY' and inst_type in ['CE', 'PE'] and expiry_ts:
            # Convert timestamp to date
            expiry_date = datetime.fromtimestamp(expiry_ts / 1000)
            expiry_str = expiry_date.strftime('%Y-%m-%d %H:%M:%S')
            
            if expiry_date.strftime('%Y-%m-%d') == '2026-02-03':
                nifty_options.append((inst, expiry_ts, expiry_str))

print(f"\nFound {len(nifty_options)} NIFTY options for date 2026-02-03")

if nifty_options:
    print("\n📊 Sample options with timestamps:")
    for i, (opt, ts, dt_str) in enumerate(nifty_options[:5]):
        print(f"{i+1}. {opt.get('trading_symbol', 'N/A')}")
        print(f"   Timestamp: {ts}")
        print(f"   DateTime: {dt_str}")
        print(f"   Key: {opt.get('instrument_key', 'N/A')}")
        print()

    # Check if all timestamps are the same
    unique_timestamps = set(ts for _, ts, _ in nifty_options)
    print(f"Unique timestamps found: {len(unique_timestamps)}")
    for ts in unique_timestamps:
        dt = datetime.fromtimestamp(ts / 1000)
        print(f"  - {ts} = {dt.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate what timestamp our code is looking for
print("\n🧮 What our code calculates:")
expiry_dt_midnight = datetime.strptime("2026-02-03", "%Y-%m-%d")
expiry_ts_midnight = int(expiry_dt_midnight.timestamp() * 1000)
print(f"Midnight timestamp: {expiry_ts_midnight}")
print(f"Midnight datetime: {expiry_dt_midnight.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate typical market close time (3:30 PM)
expiry_dt_1530 = datetime.strptime("2026-02-03 15:30:00", "%Y-%m-%d %H:%M:%S")
expiry_ts_1530 = int(expiry_dt_1530.timestamp() * 1000)
print(f"\n3:30 PM timestamp: {expiry_ts_1530}")
print(f"3:30 PM datetime: {expiry_dt_1530.strftime('%Y-%m-%d %H:%M:%S')}")