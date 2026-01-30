"""
Check what NIFTY option expiries actually exist in the data
"""
import json
import gzip
from datetime import datetime
from collections import Counter

print("🔍 Checking actual NIFTY option expiries...")

# Load the data
with gzip.open("data/instruments.json.gz", 'rt', encoding='utf-8') as f:
    data = json.load(f)

# Find all NIFTY options
nifty_options = []
for inst in data:
    if isinstance(inst, dict):
        name = inst.get('name', '')
        inst_type = inst.get('instrument_type', '')
        
        if name == 'NIFTY' and inst_type in ['CE', 'PE']:
            nifty_options.append(inst)

print(f"Found {len(nifty_options)} NIFTY options")

# Group by expiry
expiry_counter = Counter()
expiry_details = {}

for option in nifty_options:
    expiry_ts = option.get('expiry')
    if expiry_ts:
        try:
            expiry_date = datetime.fromtimestamp(expiry_ts / 1000)
            date_str = expiry_date.strftime('%Y-%m-%d')
            expiry_counter[date_str] += 1
            
            # Store sample for each expiry
            if date_str not in expiry_details:
                expiry_details[date_str] = {
                    'sample': option.get('trading_symbol', ''),
                    'strike': option.get('strike_price', 0),
                    'type': option.get('instrument_type', '')
                }
        except:
            pass

print("\n📅 NIFTY Option Expiries (with counts):")
for date_str, count in sorted(expiry_counter.items()):
    details = expiry_details.get(date_str, {})
    print(f"  {date_str}: {count} options")
    print(f"    Sample: {details.get('sample', 'N/A')}")

# Check what strikes are available for a specific expiry
print("\n🎯 Checking strikes for 2026-06-30 expiry:")
strikes = []
for option in nifty_options:
    expiry_ts = option.get('expiry')
    if expiry_ts:
        try:
            expiry_date = datetime.fromtimestamp(expiry_ts / 1000)
            if expiry_date.strftime('%Y-%m-%d') == '2026-06-30':
                strikes.append({
                    'strike': option.get('strike_price', 0),
                    'type': option.get('instrument_type', ''),
                    'symbol': option.get('trading_symbol', ''),
                    'key': option.get('instrument_key', '')
                })
        except:
            pass

if strikes:
    strikes.sort(key=lambda x: x['strike'])
    print(f"Found {len(strikes)} options for 2026-06-30")
    print("Strike range:", strikes[0]['strike'], "to", strikes[-1]['strike'])
    
    # Show sample
    print("\nSample options:")
    for i in range(min(5, len(strikes))):
        s = strikes[i]
        print(f"  {s['type']} {s['strike']}: {s['symbol']}")
        print(f"    Key: {s['key']}")
else:
    print("No options found for 2026-06-30")

# Check if we have any near-term expiries (within 30 days from a reference date)
print("\n⏰ Checking for near-term expiries:")
reference_date = datetime(2026, 1, 1)  # Using 2026 as reference since your data is for 2026
print(f"Reference date: {reference_date.strftime('%Y-%m-%d')}")

near_term = []
for date_str in expiry_counter.keys():
    expiry_date = datetime.strptime(date_str, '%Y-%m-%d')
    days_diff = (expiry_date - reference_date).days
    
    if 0 <= days_diff <= 30:  # Within next 30 days
        near_term.append((date_str, days_diff, expiry_counter[date_str]))

if near_term:
    print("Near-term expiries (within 30 days of reference):")
    for date_str, days_diff, count in sorted(near_term, key=lambda x: x[1]):
        print(f"  {date_str} (+{days_diff} days): {count} options")
else:
    print("No near-term expiries found")