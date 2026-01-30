"""
Fix the instrument filter to be more specific
"""
import json
import gzip
from datetime import datetime

# Load the data to see what's happening
with gzip.open("data/instruments.json.gz", 'rt', encoding='utf-8') as f:
    data = json.load(f)

print("🔍 Debugging instrument filter...")

# Find all instruments with "NIFTY" in name
nifty_like = []
for inst in data:
    if isinstance(inst, dict):
        name = inst.get('name', '')
        if 'NIFTY' in name:
            nifty_like.append(inst)

print(f"Found {len(nifty_like)} instruments with 'NIFTY' in name")

# Group by exact name
from collections import Counter
name_counter = Counter([inst.get('name', '') for inst in nifty_like])

print("\n📊 Breakdown by name:")
for name, count in name_counter.most_common():
    print(f"  {name}: {count}")

# Check what we're getting for expiry 2026-01-27
print("\n🔑 Checking options for 2026-01-27 expiry:")

# Convert expiry to timestamp
expiry_dt = datetime.strptime("2026-01-27", "%Y-%m-%d")
expiry_ts = int(expiry_dt.timestamp() * 1000)

options_for_date = []
for inst in nifty_like:
    if inst.get('expiry') == expiry_ts and inst.get('instrument_type') in ['CE', 'PE']:
        options_for_date.append(inst)

print(f"Found {len(options_for_date)} options for 2026-01-27")

# Group by name
date_name_counter = Counter([inst.get('name', '') for inst in options_for_date])
print("\nOptions by name for 2026-01-27:")
for name, count in date_name_counter.most_common():
    print(f"  {name}: {count}")
    
    # Show sample
    sample = next((inst for inst in options_for_date if inst.get('name') == name), None)
    if sample:
        print(f"    Sample: {sample.get('trading_symbol')} | Strike: {sample.get('strike_price')}")