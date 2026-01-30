"""
Check instrument master file
"""
import streamlit as st
import gzip
import json
from pathlib import Path
import pandas as pd

st.set_page_config(layout="wide")
st.title("🔧 Instrument Master Check")

instrument_path = Path("data/instruments.json.gz")
st.write(f"**Path:** {instrument_path.absolute()}")
st.write(f"**Exists:** {instrument_path.exists()}")
st.write(f"**Size:** {instrument_path.stat().st_size if instrument_path.exists() else 0} bytes")

if instrument_path.exists():
    try:
        # Read the file
        with gzip.open(instrument_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        st.success(f"✅ File loaded successfully")
        st.write(f"**Data type:** {type(data)}")
        
        if isinstance(data, dict):
            st.write(f"**Keys:** {list(data.keys())}")
            
            # Check NIFTY options
            if 'NIFTY' in data:
                st.write("### 📊 NIFTY Instruments")
                nifty_data = data['NIFTY']
                
                if isinstance(nifty_data, dict):
                    st.write(f"**NIFTY keys:** {list(nifty_data.keys())}")
                    
                    # Check expiries
                    if 'expiries' in nifty_data:
                        st.write(f"**Expiries:** {nifty_data['expiries'][:5]}...")
                    
                    # Check instruments
                    if 'instruments' in nifty_data:
                        instruments = nifty_data['instruments']
                        st.write(f"**Total instruments:** {len(instruments)}")
                        
                        # Show first few
                        st.write("**Sample instruments:**")
                        for i, (key, details) in enumerate(list(instruments.items())[:5]):
                            st.write(f"{i+1}. **{key}**: {details}")
                
            elif isinstance(data, list):
                st.write(f"**List length:** {len(data)}")
                st.write("**First item:**")
                st.json(data[0] if data else {})
                
        elif isinstance(data, list):
            st.write(f"**List length:** {len(data)}")
            
            # Find NIFTY instruments
            nifty_instruments = [item for item in data if isinstance(item, dict) and item.get('name', '').startswith('NIFTY')]
            st.write(f"**NIFTY instruments found:** {len(nifty_instruments)}")
            
            if nifty_instruments:
                st.write("**First NIFTY instrument:**")
                st.json(nifty_instruments[0])
        
        # Try to get option keys
        st.write("### 🔍 Test Option Key Generation")
        try:
            from data.instrument_master import get_option_keys
            
            # Try different expiries
            test_expiries = [
                "2026-01-29",
                "2026-02-05", 
                "2026-02-12"
            ]
            
            for expiry in test_expiries:
                try:
                    keys = get_option_keys("NIFTY", expiry, max_keys=5)
                    st.write(f"**Expiry {expiry}:** {len(keys) if keys else 0} keys")
                    if keys:
                        for key in keys[:3]:
                            st.write(f"  - {key}")
                except Exception as e:
                    st.write(f"**Expiry {expiry}:** Error - {e}")
                    
        except ImportError as e:
            st.error(f"❌ Cannot import get_option_keys: {e}")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        import traceback
        st.error(traceback.format_exc())
else:
    st.error("❌ Instrument file not found!")
    st.info("""
    **You need to download instrument data:**
    1. Run the instrument download script
    2. Or check if the file is in a different location
    3. The file should be at: `G:/trading_app/data/instruments.json.gz`
    """)

# Check alternative locations
st.write("### 🔎 Check Alternative Locations")
alternative_paths = [
    Path("instruments.json.gz"),
    Path("../data/instruments.json.gz"),
    Path("../../data/instruments.json.gz"),
    Path("G:/trading_app/data/instruments.json.gz"),
]

for alt_path in alternative_paths:
    st.write(f"- `{alt_path}`: {'✅ Exists' if alt_path.exists() else '❌ Missing'}")