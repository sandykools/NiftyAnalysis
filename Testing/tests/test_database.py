"""
Test database connection and feature pipeline
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🗄️ Database & Features Test")

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
    
    # Test latest features
    with st.expander("🔬 Latest Features"):
        try:
            features = fetch_latest_features(FEATURE_VERSION)
            if features is not None and not features.empty:
                st.write(f"Found {len(features)} features")
                st.dataframe(features.head())
                
                # Show feature summary
                st.markdown("**Feature Summary:**")
                feature_summary = {
                    "Timestamp": features.iloc[0]["timestamp"] if "timestamp" in features.columns else "N/A",
                    "OI Velocity": features.iloc[0]["oi_velocity"] if "oi_velocity" in features.columns else "N/A",
                    "Net Gamma": features.iloc[0]["net_gamma"] if "net_gamma" in features.columns else "N/A",
                    "Put/Call Ratio": features.iloc[0]["put_call_ratio"] if "put_call_ratio" in features.columns else "N/A",
                }
                st.json(feature_summary)
                st.success("✅ Features fetched successfully")
            else:
                st.warning("⚠️ No features found")
        except Exception as e:
            st.error(f"❌ Features error: {e}")
    
    # Test active signals
    with st.expander("🎯 Active Signals"):
        try:
            signals = fetch_active_signals(limit=10)
            if not signals.empty:
                st.write(f"Found {len(signals)} active signals")
                st.dataframe(signals[["timestamp", "signal_type", "confidence", "status"]])
                
                # Signal distribution
                if not signals.empty:
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
            if not analytics.empty:
                st.write(f"Found {len(analytics)} analytics records")
                st.dataframe(analytics)
                st.success("✅ Analytics fetched successfully")
            else:
                st.info("No analytics records found")
        except Exception as e:
            st.error(f"❌ Analytics error: {e}")
    
    # Test feature pipeline (if available)
    with st.expander("⚙️ Feature Pipeline Test"):
        try:
            from core.feature_pipeline import build_and_store_features, create_feature_summary
            
            # Create dummy data for testing
            dummy_option_chain = pd.DataFrame({
                'strike': [21000, 21500, 22000, 22500, 23000],
                'option_type': ['CE', 'CE', 'CE', 'PE', 'PE'],
                'oi': [1000, 1500, 2000, 1800, 1200],
                'volume': [100, 150, 200, 180, 120],
                'ltp': [10, 15, 20, 18, 12]
            })
            
            result = build_and_store_features(
                timestamp=pd.Timestamp.now(),
                option_chain_df=dummy_option_chain,
                spot_price=22000,
                expiry_datetime=datetime.now() + pd.Timedelta(days=3)
            )
            
            if result:
                st.write("**Feature Pipeline Result:**")
                st.json(result.get("features", {}))
                st.success("✅ Feature pipeline works")
            else:
                st.warning("⚠️ Feature pipeline returned no result")
        except ImportError:
            st.warning("⚠️ Feature pipeline not available")
        except Exception as e:
            st.error(f"❌ Feature pipeline error: {e}")

except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.write("Make sure you're running from the correct directory")
except Exception as e:
    st.error(f"❌ General error: {e}")
    import traceback
    st.error(traceback.format_exc())

st.markdown("---")
st.info("💡 Run these tests to verify your system components are working properly.")