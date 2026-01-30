#!/usr/bin/env python3
"""
Comprehensive test script for Upstox API connectivity and data fetching.
Tests all the functionality that will be used during market hours.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_authentication():
    """Test Upstox authentication."""
    print("=" * 70)
    print("TEST 1: AUTHENTICATION")
    print("=" * 70)
    
    try:
        from core.session import UpstoxSession, initialize_session
        
        # Initialize session
        initialize_session()
        
        # Authenticate
        print("Authenticating with Upstox...")
        access_token = UpstoxSession.authenticate()
        
        if access_token:
            print(f"✅ Authentication successful")
            print(f"   Token: {access_token[:30]}...")
            return access_token
        else:
            print("❌ Authentication failed")
            return None
            
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_upstox_client(access_token):
    """Test UpstoxClient functionality."""
    print("\n" + "=" * 70)
    print("TEST 2: UPSTOX CLIENT")
    print("=" * 70)
    
    try:
        from data.upstox_client import UpstoxClient
        
        # Initialize client
        print("Initializing UpstoxClient...")
        client = UpstoxClient(access_token)
        print("✅ UpstoxClient initialized")
        
        # Test 2.1: Fetch user profile
        print("\n2.1 Testing fetch_profile()...")
        try:
            profile = client.fetch_profile()
            if profile:
                print(f"✅ Profile fetched successfully")
                print(f"   User: {profile.get('user_name', 'N/A')}")
                print(f"   Email: {profile.get('email', 'N/A')}")
            else:
                print("⚠ Profile fetch returned None")
        except Exception as e:
            print(f"❌ Profile fetch error: {e}")
        
        # Test 2.2: Fetch index quote (Nifty 50)
        print("\n2.2 Testing fetch_index_quote()...")
        try:
            index_symbol = "NSE_INDEX|Nifty 50"
            index_data = client.fetch_index_quote(index_symbol)
            
            if index_data:
                print(f"✅ Index quote fetched successfully")
                print(f"   Symbol: {index_symbol}")
                print(f"   LTP: {index_data.get('ltp', 'N/A')}")
                print(f"   Open: {index_data.get('open', 'N/A')}")
                print(f"   Change %: {index_data.get('percent_change', 'N/A')}")
                
                # Check data types
                ltp = index_data.get('ltp')
                if ltp is not None:
                    try:
                        float(ltp)
                        print(f"   LTP type check: ✓ (can convert to float)")
                    except:
                        print(f"   LTP type check: ✗ (cannot convert to float)")
            else:
                print("❌ Index quote fetch returned None")
        except Exception as e:
            print(f"❌ Index quote error: {e}")
        
        # Test 2.3: Fetch equity quotes (for breadth analysis)
        print("\n2.3 Testing fetch_equity_quotes()...")
        try:
            # Test with a few symbols
            test_symbols = ["NSE|RELIANCE", "NSE|TCS", "NSE|HDFCBANK"]
            equity_data = client.fetch_equity_quotes(test_symbols)
            
            if equity_data is not None:
                print(f"✅ Equity quotes fetched")
                print(f"   Received data for {len(equity_data) if hasattr(equity_data, '__len__') else 'unknown'} symbols")
                
                if isinstance(equity_data, pd.DataFrame) and not equity_data.empty:
                    print(f"   DataFrame shape: {equity_data.shape}")
                    print(f"   Columns: {list(equity_data.columns)}")
                    
                    # Check key columns
                    required_columns = ['ltp', 'percent_change']
                    for col in required_columns:
                        if col in equity_data.columns:
                            print(f"   Column '{col}': ✓")
                        else:
                            print(f"   Column '{col}': ✗ (missing)")
                else:
                    print("⚠ Equity data is not a DataFrame or is empty")
            else:
                print("❌ Equity quotes fetch returned None")
        except Exception as e:
            print(f"❌ Equity quotes error: {e}")
        
        # Test 2.4: Fetch option chain
        print("\n2.4 Testing fetch_option_chain_with_analytics()...")
        try:
            # Get option keys first
            from data.instrument_master import get_option_keys
            
            expiry_date = (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d")
            option_keys = get_option_keys(
                underlying="NIFTY",
                expiry=expiry_date,
                max_keys=50
            )
            
            if option_keys:
                print(f"✅ Got {len(option_keys)} option keys")
                print(f"   Sample keys: {option_keys[:3]}")
                
                # Test with a subset
                test_keys = option_keys[:10]
                spot_price = 22000  # Example spot price
                
                option_data = client.fetch_option_chain_with_analytics(
                    test_keys,
                    spot_price
                )
                
                if option_data:
                    print(f"✅ Option chain analytics fetched")
                    
                    # Check structure
                    if 'raw_data' in option_data:
                        raw_data = option_data['raw_data']
                        if isinstance(raw_data, pd.DataFrame):
                            print(f"   Raw data shape: {raw_data.shape}")
                            print(f"   Raw data columns: {list(raw_data.columns)}")
                        
                        # Check for key columns
                        key_columns = ['strike', 'call_oi', 'put_oi', 'call_volume', 'put_volume']
                        missing_cols = []
                        for col in key_columns:
                            if col in raw_data.columns:
                                print(f"   Column '{col}': ✓")
                            else:
                                print(f"   Column '{col}': ✗")
                                missing_cols.append(col)
                        
                        if missing_cols:
                            print(f"   ⚠ Missing columns: {missing_cols}")
                    
                    if 'analytics' in option_data:
                        analytics = option_data['analytics']
                        print(f"   Analytics keys: {list(analytics.keys())}")
                        
                        # Check key analytics
                        important_metrics = ['put_call_ratio', 'max_pain', 'total_oi', 'total_volume']
                        for metric in important_metrics:
                            if metric in analytics:
                                print(f"   Metric '{metric}': ✓ = {analytics[metric]}")
                            else:
                                print(f"   Metric '{metric}': ✗")
                    
                    if 'market_insights' in option_data:
                        insights = option_data['market_insights']
                        print(f"   Market insights: {len(insights)} insights")
                else:
                    print("❌ Option chain fetch returned None")
            else:
                print("❌ No option keys retrieved")
        except Exception as e:
            print(f"❌ Option chain error: {e}")
            import traceback
            traceback.print_exc()
        
        return client
        
    except Exception as e:
        print(f"❌ UpstoxClient initialization error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_feature_pipeline(client):
    """Test feature pipeline functionality."""
    print("\n" + "=" * 70)
    print("TEST 3: FEATURE PIPELINE")
    print("=" * 70)
    
    try:
        from core.feature_pipeline import build_and_store_features, create_feature_summary
        from features.breadth import build_constituents_df
        
        print("Testing feature pipeline...")
        
        # Create mock data for testing
        print("\n3.1 Creating mock data...")
        
        # Mock index data
        spot_price = 22000.0
        open_price = 21950.0
        
        # Mock price series
        price_series = pd.Series([21800, 21900, 21950, 22000, 22050])
        volume_series = pd.Series([1000000, 1200000, 1100000, 1300000, 1400000])
        
        # Mock constituents data
        constituents_df = pd.DataFrame({
            'symbol': ['RELIANCE', 'TCS', 'HDFCBANK'],
            'weight': [0.10, 0.08, 0.12],
            'price_change': [1.5, -0.5, 2.0]
        })
        
        ccc_history = pd.Series([0.5, 0.3, 0.1, -0.2, -0.1])
        
        # Mock option chain data
        option_chain_df = pd.DataFrame({
            'strike': [21500, 22000, 22500],
            'call_oi': [1000, 1500, 1200],
            'put_oi': [800, 1300, 900],
            'call_volume': [100, 150, 120],
            'put_volume': [80, 130, 90],
            'call_ltp': [150.5, 75.2, 30.1],
            'put_ltp': [25.1, 80.5, 155.2]
        })
        
        # Test 3.2: Build features
        print("\n3.2 Testing build_and_store_features()...")
        try:
            feature_result = build_and_store_features(
                timestamp=pd.Timestamp.utcnow(),
                option_chain_df=option_chain_df,
                spot_price=spot_price,
                expiry_datetime=datetime.utcnow() + timedelta(days=3),
                
                ltp=spot_price,
                vwap=open_price,
                price_series=price_series,
                volume_series=volume_series,
                
                constituents_df=constituents_df,
                ccc_history=ccc_history,
                
                client=client,
                option_keys=["NIFTY28JAN22500CE", "NIFTY28JAN22500PE"]
            )
            
            if feature_result:
                print("✅ Features built successfully")
                
                if 'features' in feature_result:
                    features = feature_result['features']
                    print(f"   Number of features: {len(features)}")
                    print(f"   Feature keys: {list(features.keys())}")
                    
                    # Check key features
                    key_features = [
                        'oi_velocity', 'net_gamma', 'put_call_ratio', 
                        'wall_strength', 'trap_probability', 'divergence_score'
                    ]
                    
                    for feature in key_features:
                        if feature in features:
                            value = features[feature]
                            print(f"   Feature '{feature}': ✓ = {value}")
                            
                            # Check data type
                            if isinstance(value, (int, float)):
                                print(f"     Type: {type(value).__name__} ✓")
                            else:
                                print(f"     Type: {type(value).__name__} ⚠ (expected numeric)")
                        else:
                            print(f"   Feature '{feature}': ✗ (missing)")
                else:
                    print("⚠ No 'features' key in result")
            else:
                print("❌ Feature building returned None")
                
        except Exception as e:
            print(f"❌ Feature building error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 3.3: Create feature summary
        print("\n3.3 Testing create_feature_summary()...")
        try:
            if 'features' in locals() and features:
                summary = create_feature_summary(features)
                if summary:
                    print("✅ Feature summary created")
                    print(f"   Summary type: {type(summary)}")
                    if isinstance(summary, dict):
                        print(f"   Summary keys: {list(summary.keys())}")
                else:
                    print("❌ Feature summary returned None")
            else:
                print("⚠ Skipping feature summary test (no features available)")
                
        except Exception as e:
            print(f"❌ Feature summary error: {e}")
        
        print("\n✅ Feature pipeline tests completed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This is OK if feature pipeline is not available yet")
    except Exception as e:
        print(f"❌ Feature pipeline error: {e}")
        import traceback
        traceback.print_exc()

def test_database():
    """Test database connectivity."""
    print("\n" + "=" * 70)
    print("TEST 4: DATABASE")
    print("=" * 70)
    
    try:
        from storage.repository import (
            fetch_latest_features,
            get_database_stats,
            fetch_active_signals
        )
        from ml.feature_contract import FEATURE_VERSION
        
        print("Testing database connectivity...")
        
        # Test 4.1: Get database stats
        print("\n4.1 Testing get_database_stats()...")
        try:
            stats = get_database_stats()
            if stats:
                print("✅ Database stats fetched")
                print(f"   Database size: {stats.get('database_size_mb', 'N/A')} MB")
                print(f"   Feature version: {stats.get('feature_version', 'N/A')}")
                print(f"   Features count: {stats.get('market_features_count', 'N/A')}")
                print(f"   Signals count: {stats.get('signals_count', 'N/A')}")
            else:
                print("❌ Database stats returned None")
        except Exception as e:
            print(f"❌ Database stats error: {e}")
        
        # Test 4.2: Fetch latest features
        print("\n4.2 Testing fetch_latest_features()...")
        try:
            features = fetch_latest_features(FEATURE_VERSION)
            if features is not None:
                if hasattr(features, 'empty'):
                    if not features.empty:
                        print("✅ Latest features fetched")
                        print(f"   Shape: {features.shape}")
                        print(f"   Columns: {list(features.columns)}")
                    else:
                        print("⚠ Latest features DataFrame is empty")
                else:
                    print(f"✅ Latest features: {features}")
            else:
                print("⚠ No features found in database")
        except Exception as e:
            print(f"❌ Fetch features error: {e}")
        
        # Test 4.3: Fetch active signals
        print("\n4.3 Testing fetch_active_signals()...")
        try:
            signals = fetch_active_signals(limit=5)
            if signals is not None:
                if hasattr(signals, 'empty'):
                    if not signals.empty:
                        print("✅ Active signals fetched")
                        print(f"   Number of signals: {len(signals)}")
                        print(f"   Signal types: {signals['signal_type'].unique()}")
                    else:
                        print("⚠ No active signals in database")
                else:
                    print(f"✅ Active signals: {signals}")
            else:
                print("⚠ No signals returned")
        except Exception as e:
            print(f"❌ Fetch signals error: {e}")
        
        print("\n✅ Database tests completed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This is OK if database modules are not available yet")
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("=" * 70)
    print("MARKET HOURS COMPREHENSIVE TEST")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current time in IST: {(datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Test 1: Authentication
    access_token = test_authentication()
    
    if not access_token:
        print("\n❌ Authentication failed. Cannot proceed with other tests.")
        return
    
    # Test 2: Upstox Client
    client = test_upstox_client(access_token)
    
    if not client:
        print("\n❌ UpstoxClient initialization failed. Cannot proceed with feature pipeline test.")
    else:
        # Test 3: Feature Pipeline
        test_feature_pipeline(client)
    
    # Test 4: Database
    test_database()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("""
    ✅ If all tests passed: Your app is ready for market hours!
    
    ⚠ If some tests failed:
      1. Authentication failures: Check credentials in secrets.toml
      2. API call failures: Check network connectivity or API limits
      3. Data type issues: Check the returned data structure matches expectations
      4. Missing columns: Update feature pipeline to handle missing data
      
    🚀 Next steps for market hours:
      1. Start the app: streamlit run app.py
      2. Ensure authentication works
      3. During market hours (9:15 AM - 3:30 PM IST):
         - Watch Live Analytics tab
         - Check if research cycles execute
         - Monitor database growth
         - Look for generated signals
    """)
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    # Check if we're in market hours
    current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)  # Convert to IST
    market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    print(f"Current IST time: {current_time.strftime('%H:%M')}")
    print(f"Market hours: 9:15 AM - 3:30 PM IST")
    
    if market_open <= current_time <= market_close and current_time.weekday() < 5:
        print("✅ Currently in market hours - real data will be available")
    else:
        print("⚠ Currently outside market hours - using mock/test data")
        print("   Some API calls may fail or return cached data")
    
    input("\nPress Enter to start tests...")
    main()