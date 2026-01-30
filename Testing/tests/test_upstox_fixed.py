"""
Fixed Upstox connection test with correct method names
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Fixed Upstox Data Test")

# Initialize from your main app components
from core.session import UpstoxSession, initialize_session
from data.upstox_client import UpstoxClient

# Initialize session
initialize_session()

# Get token
access_token = UpstoxSession.get_access_token()
if not access_token:
    st.error("❌ Not authenticated. Please run the main app first to log in.")
    st.stop()

# Initialize client
try:
    client = UpstoxClient(access_token)
    st.success(f"✅ Authenticated: {st.session_state.get('upstox_profile', {}).get('user_name', 'User')}")
except Exception as e:
    st.error(f"❌ Error initializing client: {e}")
    st.stop()

# Test different endpoints
st.markdown("## 🔍 Data Fetching Tests")

# Test 1: Get Profile (if method exists)
with st.expander("👤 User Profile", expanded=True):
    try:
        # Check what methods client has
        st.write("**Available methods in UpstoxClient:**")
        methods = [m for m in dir(client) if not m.startswith('_')]
        st.write(", ".join(methods))
        
        # Try to get profile if method exists
        if hasattr(client, 'get_user_profile'):
            profile = client.get_user_profile()
            st.json(profile)
            st.success("✅ Profile fetched successfully")
        else:
            st.info("⚠️ No get_user_profile method found")
    except Exception as e:
        st.error(f"❌ Profile error: {e}")

# Test 2: Index Quote
with st.expander("📈 Index Quote (Nifty 50)"):
    try:
        index_data = client.fetch_index_quote("NSE_INDEX|Nifty 50")
        if index_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nifty 50", f"{float(index_data.get('ltp', 0)):,.0f}")
                st.metric("Change", f"{float(index_data.get('percent_change', 0)):+.2f}%")
            with col2:
                st.metric("Open", f"{float(index_data.get('open', 0)):,.0f}")
                st.metric("High/Low", f"{float(index_data.get('high', 0)):,.0f}/{float(index_data.get('low', 0)):,.0f}")
            
            st.write(f"**Timestamp:** {index_data.get('timestamp')}")
            st.success("✅ Index quote fetched successfully")
        else:
            st.warning("⚠️ No index data returned")
    except Exception as e:
        st.error(f"❌ Index quote error: {e}")

# Test 3: FIXED Equity Quotes - Correct symbol format
with st.expander("🏢 Equity Quotes (Fixed Format)"):
    try:
        # CORRECT FORMAT: NSE_EQ|SYMBOL
        symbols = ["NSE_EQ|INFY", "NSE_EQ|RELIANCE", "NSE_EQ|TCS", "NSE_EQ|HDFCBANK"]
        st.write(f"**Fetching symbols:** {symbols}")
        
        equity_data = client.fetch_equity_quotes(symbols)
        if equity_data:
            df = pd.DataFrame(equity_data)
            st.dataframe(df)
            st.success(f"✅ Fetched {len(df)} equity quotes")
        else:
            st.warning("⚠️ No equity data returned")
    except Exception as e:
        st.error(f"❌ Equity quotes error: {e}")

# Test 4: Check Instrument Master
with st.expander("🔧 Instrument Master Check"):
    try:
        from data.instrument_master import get_option_keys, load_instruments
        
        # Check if instrument file exists
        import json
        import gzip
        from pathlib import Path
        
        instrument_path = Path("data/instruments.json.gz")
        st.write(f"**Instrument file:** {instrument_path}")
        st.write(f"**Exists:** {instrument_path.exists()}")
        
        if instrument_path.exists():
            try:
                # Try to load a sample
                with gzip.open(instrument_path, 'rt', encoding='utf-8') as f:
                    sample = f.read(500)
                    st.write("**File sample:**")
                    st.code(sample)
                    
                # Try to get option keys
                expiry = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
                option_keys = get_option_keys(
                    underlying="NIFTY",
                    expiry=expiry,
                    max_keys=10
                )
                
                if option_keys:
                    st.write(f"**Found {len(option_keys)} option keys for {expiry}:**")
                    for key in option_keys[:5]:
                        st.write(f"- {key}")
                else:
                    st.warning("⚠️ No option keys found. The file might be empty or format is wrong.")
                    
            except Exception as e:
                st.error(f"❌ Error reading instrument file: {e}")
        else:
            st.error("❌ Instrument file not found!")
            
    except Exception as e:
        st.error(f"❌ Instrument master error: {e}")

# Test 5: Check what other data we can fetch
with st.expander("📡 Other Available Endpoints"):
    try:
        # List available endpoints based on client methods
        endpoints = [
            ('fetch_option_chain', 'Option Chain'),
            ('fetch_option_chain_with_analytics', 'Option Chain with Analytics'),
            ('fetch_market_depth', 'Market Depth'),
        ]
        
        for method_name, display_name in endpoints:
            if hasattr(client, method_name):
                st.write(f"✅ **{display_name}** - Available")
                try:
                    # Try a simple test for option chain if we have keys
                    if method_name == 'fetch_option_chain' or method_name == 'fetch_option_chain_with_analytics':
                        # Try to get some option keys first
                        from data.instrument_master import get_option_keys
                        expiry = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
                        option_keys = get_option_keys("NIFTY", expiry, 5)
                        
                        if option_keys:
                            index_data = client.fetch_index_quote("NSE_INDEX|Nifty 50")
                            ltp = float(index_data.get('ltp', 22000)) if index_data else 22000
                            
                            if method_name == 'fetch_option_chain':
                                result = client.fetch_option_chain(option_keys)
                            else:
                                result = client.fetch_option_chain_with_analytics(option_keys, ltp)
                            
                            if result:
                                st.success(f"  ↳ {display_name} test successful")
                            else:
                                st.warning(f"  ↳ {display_name} returned no data")
                except Exception as e:
                    st.error(f"  ↳ {display_name} test failed: {str(e)[:100]}")
            else:
                st.write(f"❌ **{display_name}** - Not available")
                
    except Exception as e:
        st.error(f"❌ Endpoints check error: {e}")

st.markdown("---")
st.info("💡 Check which methods are actually available in your UpstoxClient class.")