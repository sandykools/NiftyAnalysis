"""
Test with existing session (run after main app)
"""
import streamlit as st
from core.session import UpstoxSession, initialize_session
from data.upstox_client import UpstoxClient

st.set_page_config(layout="wide")
st.title("🧪 Test with Existing Session")

# Initialize session (won't show login if already logged in)
initialize_session()

# Check if we have a token
access_token = UpstoxSession.get_access_token()

if not access_token:
    st.error("❌ No active session found. Please:")
    st.write("1. Run `streamlit run app.py` first")
    st.write("2. Log in through the main app")
    st.write("3. Then run this test")
    st.stop()

st.success(f"✅ Using existing session")

# Initialize client
client = UpstoxClient(access_token)

# List available methods
st.write("### 📋 Available Methods in UpstoxClient:")
methods = [m for m in dir(client) if not m.startswith('_')]
fetch_methods = [m for m in methods if 'fetch' in m.lower()]

for method in sorted(fetch_methods):
    st.write(f"- `{method}`")

# Test index quote
st.write("### 📈 Test Index Quote:")
try:
    index_data = client.fetch_index_quote("NSE_INDEX|Nifty 50")
    if index_data:
        st.metric("Nifty 50", f"{float(index_data.get('ltp', 0)):,.0f}")
        st.json(index_data)
    else:
        st.warning("No index data")
except Exception as e:
    st.error(f"Error: {e}")

# Test what instrument keys we can get
st.write("### 🔧 Test Instrument Discovery:")
try:
    from data.instrument_master import get_option_keys
    
    # Find nearest expiry
    import pandas as pd
    from datetime import datetime
    
    # Load instruments to find expiries
    import gzip
    import json
    
    with gzip.open("data/instruments.json.gz", 'rt', encoding='utf-8') as f:
        instruments = json.load(f)
    
    # Find NIFTY option expiries
    nifty_expiries = set()
    for inst in instruments:
        if isinstance(inst, dict):
            if inst.get('name') == 'NIFTY' and inst.get('instrument_type') in ['CE', 'PE']:
                expiry_ts = inst.get('expiry')
                if expiry_ts:
                    expiry_date = datetime.fromtimestamp(expiry_ts / 1000)
                    nifty_expiries.add(expiry_date.strftime('%Y-%m-%d'))
    
    if nifty_expiries:
        st.write(f"Found {len(nifty_expiries)} NIFTY expiry dates")
        
        # Try each expiry
        for expiry in sorted(nifty_expiries)[:3]:  # Try first 3
            st.write(f"**Trying expiry: {expiry}**")
            keys = get_option_keys("NIFTY", expiry, max_keys=5)
            
            if keys:
                st.success(f"✓ Found {len(keys)} keys")
                for key in keys[:3]:
                    st.write(f"  - {key}")
                
                # Try to fetch option chain
                st.write("**Testing option chain...**")
                try:
                    option_data = client.fetch_option_chain_with_analytics(keys, 25000)
                    if option_data:
                        st.success("✓ Option chain fetched")
                        if 'analytics' in option_data:
                            st.json(option_data['analytics'])
                    else:
                        st.warning("No option data returned")
                except Exception as e:
                    st.error(f"Option chain error: {e}")
            else:
                st.warning(f"No keys for {expiry}")
    else:
        st.error("No NIFTY expiries found!")
        
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.error(traceback.format_exc())