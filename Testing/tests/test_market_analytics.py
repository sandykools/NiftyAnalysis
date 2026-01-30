"""
Test live market analytics during market hours
"""
import streamlit as st
import pandas as pd
from datetime import datetime, time
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Live Market Analytics Test")

# Check if market is open
now = datetime.now().time()
market_open = time(9, 15)
market_close = time(15, 30)

if market_open <= now <= market_close:
    st.success("✅ Market is open - running live tests")
    
    from core.session import UpstoxSession, initialize_session
    from data.upstox_client import UpstoxClient
    
    # Initialize
    initialize_session()
    access_token = UpstoxSession.authenticate()
    
    if access_token:
        client = UpstoxClient(access_token)
        
        # Test 1: Live Index Data
        st.markdown("## 🔄 Live Index Data")
        
        col1, col2, col3 = st.columns(3)
        
        try:
            nifty_data = client.fetch_index_quote("NSE_INDEX|Nifty 50")
            if nifty_data:
                with col1:
                    st.metric(
                        "Nifty 50", 
                        f"{float(nifty_data.get('ltp', 0)):,.0f}",
                        f"{float(nifty_data.get('percent_change', 0)):+.2f}%"
                    )
                
                with col2:
                    st.metric("Open", f"{float(nifty_data.get('open', 0)):,.0f}")
                
                with col3:
                    st.metric("Timestamp", nifty_data.get('timestamp', 'N/A'))
                
                st.success("✅ Live index data working")
            else:
                st.warning("⚠️ No live index data")
        except Exception as e:
            st.error(f"❌ Index data error: {e}")
        
        # Test 2: Option Chain Analytics
        st.markdown("## 📊 Live Option Analytics")
        
        try:
            from data.instrument_master import get_option_keys
            
            option_keys = get_option_keys(
                underlying="NIFTY",
                expiry=datetime.now().strftime("%Y-%m-%d"),
                max_keys=100
            )
            
            if option_keys:
                st.info(f"Found {len(option_keys)} option instruments")
                
                # Get current price
                ltp = float(nifty_data.get('ltp', 22000)) if nifty_data else 22000
                
                # Fetch option chain
                option_data = client.fetch_option_chain_with_analytics(
                    option_keys, 
                    ltp
                )
                
                if option_data and 'analytics' in option_data:
                    analytics = option_data['analytics']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        pcr = analytics.get('put_call_ratio', 1.0)
                        st.metric("Put/Call Ratio", f"{pcr:.2f}")
                    
                    with col2:
                        oi_velocity = analytics.get('oi_velocity', 0)
                        st.metric("OI Velocity", f"{oi_velocity:.2f}σ")
                    
                    with col3:
                        net_gamma = analytics.get('gamma_exposure', {}).get('net_gamma', 0)
                        st.metric("Net Gamma", f"{net_gamma:,.0f}")
                    
                    # Display insights if available
                    if 'market_insights' in analytics:
                        st.markdown("#### 💡 Market Insights")
                        for insight in analytics['market_insights'][:3]:
                            st.info(insight)
                    
                    st.success("✅ Option analytics working")
                else:
                    st.warning("⚠️ No option analytics available")
            else:
                st.warning("⚠️ No option keys found")
        except Exception as e:
            st.error(f"❌ Option analytics error: {e}")
        
        # Test 3: Breadth Analysis
        st.markdown("## 🏢 Market Breadth")
        
        try:
            # Load Nifty weights
            import json
            with open("config/nifty_weights.json", "r") as f:
                nifty_weights = json.load(f)
            
            # Fetch equity quotes for top 20 constituents
            top_symbols = list(nifty_weights.keys())[:20]
            equity_data = client.fetch_equity_quotes(top_symbols)
            
            if equity_data:
                from features.breadth import build_constituents_df
                
                constituents_df = build_constituents_df(
                    equity_quotes_df=equity_data,
                    weights={k: v for k, v in nifty_weights.items() if k in top_symbols}
                )
                
                if not constituents_df.empty:
                    # Calculate CCC
                    ccc_value = (
                        constituents_df["weight"] * constituents_df["price_change"]
                    ).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("CCC Value", f"{ccc_value:.4f}")
                    
                    with col2:
                        positive = len(constituents_df[constituents_df["price_change"] > 0])
                        st.metric("Positive Constituents", f"{positive}/{len(constituents_df)}")
                    
                    with col3:
                        avg_change = constituents_df["price_change"].mean()
                        st.metric("Avg Change", f"{avg_change:.2f}%")
                    
                    # Display heatmap
                    st.markdown("#### 📊 Constituent Performance")
                    fig = go.Figure(data=go.Heatmap(
                        z=[constituents_df["price_change"].values],
                        x=constituents_df["symbol"].str.replace("NSE_EQ|", ""),
                        colorscale='RdYlGn',
                        showscale=True
                    ))
                    fig.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Breadth analysis working")
                else:
                    st.warning("⚠️ No constituents data")
            else:
                st.warning("⚠️ No equity data")
        except Exception as e:
            st.error(f"❌ Breadth analysis error: {e}")
    
    else:
        st.error("❌ Not authenticated")
else:
    st.warning("⏸️ Market is closed. Live tests only work during market hours (9:15 AM - 3:30 PM)")

st.markdown("---")
st.info("💡 Run this during market hours to test live data feeds.")