"""
Download fresh instrument data from Upstox
"""
import streamlit as st
from core.session import UpstoxSession, initialize_session
from data.upstox_client import UpstoxClient
import json
import gzip
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📥 Download Fresh Instrument Data")

# Initialize session
initialize_session()

# Get token
access_token = UpstoxSession.get_access_token()
if not access_token:
    st.error("Please log in first through the main app")
    st.stop()

client = UpstoxClient(access_token)

st.write("### 🔄 Fetching instrument master...")

try:
    # Try to fetch instruments (method name might vary)
    if hasattr(client, 'fetch_instrument_master'):
        instruments = client.fetch_instrument_master()
    elif hasattr(client, 'get_instrument_master'):
        instruments = client.get_instrument_master()
    else:
        st.error("No instrument master method found in UpstoxClient")
        st.write("Available methods:", [m for m in dir(client) if 'instrument' in m.lower()])
        st.stop()
    
    if instruments:
        st.success(f"✅ Fetched {len(instruments)} instruments")
        
        # Save to file
        output_path = "data/instruments.json.gz"
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(instruments, f)
        
        st.success(f"✅ Saved to: {output_path}")
        
        # Show sample
        with st.expander("📋 Sample Data"):
            st.json(instruments[0] if instruments else {})
        
        # Count NIFTY options
        nifty_options = [i for i in instruments if isinstance(i, dict) 
                        and i.get('name') == 'NIFTY' 
                        and i.get('instrument_type') in ['CE', 'PE']]
        
        st.write(f"**NIFTY options found:** {len(nifty_options)}")
        
        # Show expiries
        if nifty_options:
            expiries = set()
            for opt in nifty_options[:10]:
                expiry_ts = opt.get('expiry')
                if expiry_ts:
                    try:
                        expiry_date = datetime.fromtimestamp(expiry_ts / 1000)
                        expiries.add(expiry_date.strftime('%Y-%m-%d'))
                    except:
                        pass
            
            st.write(f"**NIFTY expiry dates:** {sorted(expiries)}")
            
    else:
        st.error("No instruments returned")
        
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.error(traceback.format_exc())