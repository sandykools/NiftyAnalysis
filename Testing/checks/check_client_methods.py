"""
Check what methods UpstoxClient has
"""
import streamlit as st
from core.session import UpstoxSession, initialize_session

st.set_page_config(layout="wide")
st.title("🔍 UpstoxClient Method Check")

initialize_session()
access_token = UpstoxSession.get_access_token()

if access_token:
    from data.upstox_client import UpstoxClient
    client = UpstoxClient(access_token)
    
    st.write("### 📋 All Methods in UpstoxClient:")
    methods = [m for m in dir(client) if not m.startswith('_')]
    
    # Group methods
    fetch_methods = [m for m in methods if 'fetch' in m.lower()]
    other_methods = [m for m in methods if m not in fetch_methods]
    
    st.write("#### 🔄 Fetch Methods:")
    for method in sorted(fetch_methods):
        st.write(f"- `{method}`")
    
    st.write("#### ⚙️ Other Methods:")
    for method in sorted(other_methods):
        st.write(f"- `{method}`")
    
    st.write(f"**Total methods:** {len(methods)}")
    
    # Check specific methods
    st.write("### 🧪 Method Availability Check:")
    required_methods = [
        'fetch_index_quote',
        'fetch_equity_quotes', 
        'fetch_option_chain',
        'fetch_option_chain_with_analytics',
        'fetch_profile',
        'fetch_holdings',
        'fetch_historical_data'
    ]
    
    for method in required_methods:
        if hasattr(client, method):
            st.success(f"✅ `{method}` - Available")
        else:
            st.error(f"❌ `{method}` - NOT available")
            
else:
    st.error("Not authenticated")