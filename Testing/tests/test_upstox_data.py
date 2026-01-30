"""
Test Upstox connection and data fetching capabilities
"""
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Upstox Data Test")

# Initialize from your main app components
from core.session import UpstoxSession, initialize_session
from data.upstox_client import UpstoxClient

# Initialize session
initialize_session()

# Get token - handle None return
access_token = UpstoxSession.get_access_token()
if not access_token:
    # Try to authenticate
    access_token = UpstoxSession.authenticate()
    
if not access_token:
    st.error("❌ Not authenticated. Please log in first.")
    
    # Show login button
    login_url = UpstoxSession.get_login_url()
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <a href="{login_url}">
            <button style="
                background-color: #00d09c;
                color: white;
                padding: 15px 30px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                width: 60%;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
            📈 Login with Upstox
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Initialize client
try:
    client = UpstoxClient(access_token)
    st.success(f"✅ Authenticated successfully")
    
    # Show user info
    profile = st.session_state.get('upstox_profile', {})
    if profile:
        st.info(f"👤 User: {profile.get('user_name', 'Unknown')} | Email: {profile.get('email', 'N/A')}")
    
except Exception as e:
    st.error(f"❌ Error initializing client: {e}")
    st.stop()

# Test different endpoints
st.markdown("## 🔍 Data Fetching Tests")

# Test 1: Profile
with st.expander("👤 User Profile", expanded=True):
    try:
        profile = client.fetch_profile()
        if profile:
            st.json(profile)
            st.success("✅ Profile fetched successfully")
        else:
            st.warning("⚠️ No profile data returned")
    except Exception as e:
        st.error(f"❌ Profile error: {e}")

# Test 2: Index Quote
with st.expander("📈 Index Quote (Nifty 50)"):
    try:
        index_data = client.fetch_index_quote("NSE_INDEX|Nifty 50")
        if index_data:
            st.write(f"**LTP:** {index_data.get('ltp')}")
            st.write(f"**Change %:** {index_data.get('percent_change')}")
            st.write(f"**Timestamp:** {index_data.get('timestamp')}")
            st.write(f"**Open:** {index_data.get('open')}")
            st.write(f"**High:** {index_data.get('high')}")
            st.write(f"**Low:** {index_data.get('low')}")
            st.write(f"**Close:** {index_data.get('close')}")
            st.success("✅ Index quote fetched successfully")
        else:
            st.warning("⚠️ No index data returned")
    except Exception as e:
        st.error(f"❌ Index quote error: {e}")

# Test 3: Equity Quotes
with st.expander("🏢 Equity Quotes (Sample)"):
    try:
        symbols = ["NSE_EQ|INFY", "NSE_EQ|RELIANCE", "NSE_EQ|TCS", "NSE_EQ|HDFCBANK"]
        equity_data = client.fetch_equity_quotes(symbols)
        if equity_data:
            df = pd.DataFrame(equity_data)
            st.dataframe(df)
            st.success(f"✅ Fetched {len(df)} equity quotes")
        else:
            st.warning("⚠️ No equity data returned")
    except Exception as e:
        st.error(f"❌ Equity quotes error: {e}")

# Test 4: Option Chain
with st.expander("📊 Option Chain"):
    try:
        # Get option keys first
        from data.instrument_master import get_option_keys
        import json
        
        # Find expiry date (nearest Thursday)
        today = datetime.now()
        days_ahead = (3 - today.weekday()) % 7  # Thursday is weekday 3
        expiry_date = today + pd.Timedelta(days=days_ahead)
        
        option_keys = get_option_keys(
            underlying="NIFTY",
            expiry=expiry_date.strftime("%Y-%m-%d"),
            max_keys=30
        )
        
        if option_keys:
            st.write(f"Found {len(option_keys)} option keys (Expiry: {expiry_date.strftime('%Y-%m-%d')})")
            
            # Get current price for ATM calculation
            index_data = client.fetch_index_quote("NSE_INDEX|Nifty 50")
            ltp = float(index_data.get('ltp', 22000)) if index_data else 22000
            
            # Test option chain with limited keys
            option_data = client.fetch_option_chain_with_analytics(option_keys[:10], ltp)
            
            if option_data:
                if 'raw_data' in option_data and not option_data['raw_data'].empty:
                    st.write("**Raw Data (first 10 rows):**")
                    st.dataframe(option_data['raw_data'].head(10))
                
                if 'analytics' in option_data:
                    st.write("**Analytics:**")
                    st.json(option_data['analytics'])
                
                if 'market_insights' in option_data:
                    st.write("**Market Insights:**")
                    for insight in option_data['market_insights'][:3]:
                        st.info(insight)
                
                st.success("✅ Option chain fetched successfully")
            else:
                st.warning("⚠️ No option chain data returned")
        else:
            st.warning("⚠️ No option keys found")
    except Exception as e:
        st.error(f"❌ Option chain error: {e}")
        import traceback
        st.error(traceback.format_exc())

# Test 5: Historical Data
with st.expander("📅 Historical Data"):
    try:
        # Example: Get last 5 days of Nifty data
        from datetime import timedelta
        
        historical = client.fetch_historical_data(
            instrument_key="NSE_INDEX|Nifty 50",
            interval="1d",
            from_date=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            to_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        if historical is not None and not historical.empty:
            st.dataframe(historical)
            st.success(f"✅ Fetched {len(historical)} historical bars")
        else:
            st.warning("⚠️ No historical data returned")
    except Exception as e:
        st.error(f"❌ Historical data error: {e}")

# Test 6: Holdings
with st.expander("💰 Holdings"):
    try:
        holdings = client.fetch_holdings()
        if holdings:
            st.write(f"Found {len(holdings)} holdings")
            st.json(holdings[:5])  # Show first 5
            st.success("✅ Holdings fetched successfully")
        else:
            st.info("No holdings found")
    except Exception as e:
        st.error(f"❌ Holdings error: {e}")

st.markdown("---")
st.info("💡 All tests completed. Check for any errors above.")