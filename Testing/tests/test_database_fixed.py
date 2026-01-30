"""
Fixed version of database test
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🗄️ Database & Features Test (Fixed)")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    # Test imports
    st.markdown("## 📦 Import Tests")
    
    from storage.repository import (
        fetch_latest_features,
        get_database_stats,
        fetch_active_signals,
        fetch_recent_analytics
    )
    
    from ml.feature_contract import FEATURE_VERSION
    
    st.success("✅ All imports successful")
    
    # Test database stats
    with st.expander("📊 Database Statistics", expanded=True):
        try:
            stats = get_database_stats()
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Features", stats.get('market_features_count', 0))
                with col2:
                    st.metric("Signals", stats.get('signals_count', 0))
                with col3:
                    st.metric("Version", stats.get('feature_version', 'N/A'))
                with col4:
                    st.metric("Size", f"{stats.get('database_size_mb', 0):.1f} MB")
                st.success("✅ Database stats fetched")
            else:
                st.warning("⚠️ No database stats returned")
        except Exception as e:
            st.error(f"❌ Database stats error: {e}")
    
    # Test latest features - FIXED VERSION
    with st.expander("🔬 Latest Features"):
        try:
            features = fetch_latest_features(FEATURE_VERSION)
            if features is not None and not features.empty:
                st.write(f"Found {len(features)} features")
                st.dataframe(features.head())
                
                # Show feature summary
                st.markdown("**Feature Summary:**")
                feature_dict = features.iloc[0].to_dict()
                
                # Display important features
                important_features = {
                    k: v for k, v in feature_dict.items() 
                    if any(keyword in k.lower() for keyword in ['oi', 'gamma', 'velocity', 'divergence', 'ratio'])
                }
                
                st.json(important_features)
                st.success("✅ Features fetched successfully")
            else:
                st.warning("⚠️ No features found")
        except Exception as e:
            st.error(f"❌ Features error: {e}")
    
    # Test active signals
    with st.expander("🎯 Active Signals"):
        try:
            signals = fetch_active_signals(limit=10)
            if signals is not None and not signals.empty:
                st.write(f"Found {len(signals)} active signals")
                st.dataframe(signals[["timestamp", "signal_type", "confidence", "status"]].head())
                
                # Signal distribution
                signal_counts = signals["signal_type"].value_counts()
                st.markdown("**Signal Distribution:**")
                for signal_type, count in signal_counts.items():
                    st.write(f"- {signal_type}: {count}")
                st.success("✅ Signals fetched successfully")
            else:
                st.info("📭 No active signals found")
        except Exception as e:
            st.error(f"❌ Signals error: {e}")
    
    # Test recent analytics
    with st.expander("📈 Recent Analytics"):
        try:
            analytics = fetch_recent_analytics(limit=5)
            if analytics is not None and not analytics.empty:
                st.write(f"Found {len(analytics)} analytics records")
                st.dataframe(analytics.head())
                st.success("✅ Analytics fetched successfully")
            else:
                st.info("No analytics records found")
        except Exception as e:
            st.error(f"❌ Analytics error: {e}")
    
    # Test feature pipeline with proper dummy data
    with st.expander("⚙️ Feature Pipeline Test"):
        try:
            from core.feature_pipeline import build_and_store_features
            
            # Create proper dummy data
            dummy_option_chain = pd.DataFrame({
                'strike': [21000, 21500, 22000, 22500, 23000] * 2,
                'option_type': ['CE'] * 5 + ['PE'] * 5,
                'oi': [1000, 1500, 2000, 1800, 1200] * 2,
                'volume': [100, 150, 200, 180, 120] * 2,
                'ltp': [10, 15, 20, 18, 12] * 2,
                'change_oi': [50, 75, 100, 90, 60] * 2,
                'implied_volatility': [0.15] * 10
            })
            
            # Create dummy series
            price_series = pd.Series([21900, 21950, 22000, 22050, 22100])
            volume_series = pd.Series([1000000, 1200000, 1500000, 1300000, 1400000])
            
            # Create dummy constituents
            constituents_df = pd.DataFrame({
                'symbol': ['INFY', 'RELIANCE', 'TCS', 'HDFC', 'ICICI'],
                'weight': [0.08, 0.10, 0.07, 0.09, 0.06],
                'price_change': [0.5, -0.3, 0.8, 0.2, -0.1]
            })
            
            # Create CCC history
            ccc_history = pd.Series([0.01, 0.02, 0.015, 0.025, 0.03])
            
            # Try to build features
            result = build_and_store_features(
                timestamp=pd.Timestamp.now(),
                option_chain_df=dummy_option_chain,
                spot_price=22000,
                expiry_datetime=datetime.now() + timedelta(days=3),
                ltp=22000,
                vwap=21980,
                price_series=price_series,
                volume_series=volume_series,
                constituents_df=constituents_df,
                ccc_history=ccc_history,
                client=None,  # Can be None for test
                option_keys=[]
            )
            
            if result:
                st.write("**Feature Pipeline Result:**")
                st.json(result.get("features", {}))
                st.success("✅ Feature pipeline works")
            else:
                st.warning("⚠️ Feature pipeline returned no result")
        except ImportError as e:
            st.warning(f"⚠️ Feature pipeline not available: {e}")
        except Exception as e:
            st.error(f"❌ Feature pipeline error: {e}")
            import traceback
            st.error(traceback.format_exc())

except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.write("Make sure you're running from the correct directory")
    import traceback
    st.error(traceback.format_exc())
except Exception as e:
    st.error(f"❌ General error: {e}")
    import traceback
    st.error(traceback.format_exc())

st.markdown("---")
st.info("💡 Run these tests to verify your system components are working properly.")