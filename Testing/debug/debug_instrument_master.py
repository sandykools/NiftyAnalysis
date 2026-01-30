"""
Debug instrument master to see why option keys aren't generated
"""
import streamlit as st
import gzip
import json
from datetime import datetime
import pandas as pd

st.set_page_config(layout="wide")
st.title("🐛 Debug Instrument Master")

# Load instrument data
instrument_path = "data/instruments.json.gz"
with gzip.open(instrument_path, 'rt', encoding='utf-8') as f:
    data = json.load(f)

st.write(f"Total instruments: {len(data)}")

# Filter NIFTY options
nifty_options = []
for instrument in data:
    if isinstance(instrument, dict):
        name = instrument.get('name', '')
        instrument_type = instrument.get('instrument_type', '')
        
        if 'NIFTY' in name and instrument_type in ['CE', 'PE']:
            nifty_options.append(instrument)

st.write(f"NIFTY options found: {len(nifty_options)}")

if nifty_options:
    # Show first 5
    st.write("### 📋 Sample NIFTY Options:")
    for i, option in enumerate(nifty_options[:5]):
        st.write(f"**Option {i+1}:**")
        
        # Convert timestamp to date
        expiry_timestamp = option.get('expiry')
        if expiry_timestamp:
            try:
                expiry_date = datetime.fromtimestamp(expiry_timestamp / 1000)
                option['expiry_date'] = expiry_date.strftime('%Y-%m-%d')
            except:
                option['expiry_date'] = 'Invalid timestamp'
        
        # Show key fields
        display_info = {
            'name': option.get('name'),
            'instrument_type': option.get('instrument_type'),
            'strike_price': option.get('strike_price'),
            'expiry_timestamp': option.get('expiry'),
            'expiry_date': option.get('expiry_date', 'N/A'),
            'instrument_key': option.get('instrument_key'),
            'trading_symbol': option.get('trading_symbol')
        }
        st.json(display_info)

# Check what dates are available
st.write("### 📅 Available Expiry Dates:")
expiry_dates = set()
for option in nifty_options[:100]:  # Check first 100
    expiry = option.get('expiry')
    if expiry:
        try:
            date_str = datetime.fromtimestamp(expiry / 1000).strftime('%Y-%m-%d')
            expiry_dates.add(date_str)
        except:
            pass

st.write(f"Found {len(expiry_dates)} unique expiry dates:")
for date in sorted(expiry_dates)[:10]:  # Show first 10
    st.write(f"- {date}")

# Check current instrument_master.py logic
st.write("### 🔧 Current get_option_keys logic:")
st.code("""
def get_option_keys(underlying: str, expiry: str, max_keys: int = 200) -> List[str]:
    # Load instruments
    instruments = load_instruments()
    
    # Filter for the underlying
    underlying_instruments = [
        inst for inst in instruments 
        if inst.get('underlying_symbol') == underlying
    ]
    
    # Convert expiry string to timestamp for comparison
    try:
        expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
        expiry_ts = int(expiry_dt.timestamp() * 1000)
    except:
        return []
    
    # Filter by expiry and option type
    option_instruments = [
        inst for inst in underlying_instruments
        if inst.get('expiry') == expiry_ts 
        and inst.get('instrument_type') in ['CE', 'PE']
    ]
    
    # Sort by strike price and limit
    option_instruments.sort(key=lambda x: x.get('strike_price', 0))
    return [inst['instrument_key'] for inst in option_instruments[:max_keys]]
""")

# Test the actual function
st.write("### 🧪 Test get_option_keys function:")
try:
    from data.instrument_master import get_option_keys
    
    # Try with a date from the data
    if expiry_dates:
        test_date = list(expiry_dates)[0]
        st.write(f"Testing with date: {test_date}")
        
        keys = get_option_keys("NIFTY", test_date, max_keys=10)
        st.write(f"Result: {len(keys)} keys")
        if keys:
            for key in keys[:5]:
                st.write(f"- {key}")
        else:
            st.error("No keys returned!")
            
            # Let's debug why
            st.write("**Debugging...**")
            from data.instrument_master import load_instruments
            instruments = load_instruments()
            
            # Check what underlying_symbol values exist
            underlying_symbols = set()
            for inst in instruments[:1000]:
                if isinstance(inst, dict):
                    sym = inst.get('underlying_symbol')
                    if sym:
                        underlying_symbols.add(sym)
            
            st.write(f"Found underlying symbols: {list(underlying_symbols)}")
            
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.error(traceback.format_exc())