#!/usr/bin/env python3
"""
REAL API TEST - Only tests actual Upstox API calls with real data.
No mock data - tests exactly what happens during market hours.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def get_access_token():
    """Get access token from session or authenticate."""
    try:
        from core.session import UpstoxSession
        
        # Check if already authenticated
        if UpstoxSession.is_authenticated():
            token = UpstoxSession.get_access_token()
            if token:
                print(f"✓ Using existing token: {token[:30]}...")
                return token
        
        # Try to authenticate
        print("Attempting to authenticate...")
        
        # We need streamlit context for authentication
        # For testing, we'll check if token exists in environment
        token = os.getenv('UPSTOX_ACCESS_TOKEN')
        if token:
            print(f"✓ Using token from environment: {token[:30]}...")
            return token
        
        print("⚠ No access token available. You need to:")
        print("   1. Run: streamlit run app.py")
        print("   2. Complete the OAuth login")
        print("   3. Then run this test")
        return None
        
    except Exception as e:
        print(f"❌ Token retrieval error: {e}")
        return None

def test_real_api_calls(access_token):
    """Test REAL API calls with NO mock data."""
    print("\n" + "=" * 70)
    print("REAL API CALLS TEST (No Mock Data)")
    print("=" * 70)
    
    if not access_token:
        print("❌ No access token - cannot test API calls")
        return None
    
    try:
        from data.upstox_client import UpstoxClient
        
        print("Initializing UpstoxClient with real token...")
        client = UpstoxClient(access_token)
        print("✅ UpstoxClient initialized")
        
        test_results = {
            'profile': None,
            'index_quote': None,
            'equity_quotes': None,
            'option_chain': None,
            'errors': []
        }
        
        # TEST 1: Fetch REAL user profile
        print("\n1. Testing fetch_profile() with REAL API call...")
        try:
            profile = client.fetch_profile()
            if profile:
                test_results['profile'] = profile
                print(f"✅ REAL profile fetched:")
                print(f"   User Name: {profile.get('user_name', 'N/A')}")
                print(f"   Email: {profile.get('email', 'N/A')}")
                print(f"   User ID: {profile.get('user_id', 'N/A')}")
                
                # Check data structure
                if isinstance(profile, dict):
                    print(f"   ✓ Profile is dictionary with {len(profile)} keys")
                else:
                    print(f"   ⚠ Profile is {type(profile)}, expected dict")
            else:
                print("❌ Profile fetch returned None/Empty")
                test_results['errors'].append("Profile fetch returned None")
        except Exception as e:
            print(f"❌ Profile fetch error: {e}")
            test_results['errors'].append(f"Profile error: {e}")
        
        # TEST 2: Fetch REAL index quote (Nifty 50)
        print("\n2. Testing fetch_index_quote() with REAL API call...")
        try:
            index_symbol = "NSE_INDEX|Nifty 50"
            index_data = client.fetch_index_quote(index_symbol)
            
            if index_data:
                test_results['index_quote'] = index_data
                print(f"✅ REAL index quote fetched for {index_symbol}")
                
                # Check ALL fields in the response
                print(f"   Response keys: {list(index_data.keys())}")
                
                # Critical fields that should exist
                critical_fields = ['ltp', 'open', 'high', 'low', 'close', 'volume']
                for field in critical_fields:
                    if field in index_data:
                        value = index_data[field]
                        print(f"   {field}: {value} (type: {type(value).__name__})")
                        
                        # Try to convert to float to check data type
                        try:
                            float_val = float(value)
                            print(f"     ✓ Can convert to float: {float_val}")
                        except:
                            print(f"     ⚠ Cannot convert to float")
                    else:
                        print(f"   ⚠ Missing field: {field}")
                
                # Additional checks
                if 'timestamp' in index_data:
                    print(f"   Timestamp: {index_data['timestamp']}")
                
                if 'percent_change' in index_data:
                    print(f"   Percent Change: {index_data['percent_change']}")
            else:
                print("❌ Index quote fetch returned None")
                test_results['errors'].append("Index quote returned None")
        except Exception as e:
            print(f"❌ Index quote error: {e}")
            test_results['errors'].append(f"Index quote error: {e}")
        
        # TEST 3: Fetch REAL equity quotes (for breadth analysis)
        print("\n3. Testing fetch_equity_quotes() with REAL API call...")
        try:
            # Use REAL Nifty 50 constituents
            test_symbols = [
                "NSE|RELIANCE",    # Reliance
                "NSE|TCS",         # TCS
                "NSE|HDFCBANK",    # HDFC Bank
                "NSE|ICICIBANK",   # ICICI Bank
                "NSE|INFY"         # Infosys
            ]
            
            equity_data = client.fetch_equity_quotes(test_symbols)
            
            if equity_data is not None:
                test_results['equity_quotes'] = equity_data
                print("✅ REAL equity quotes fetched")
                
                if isinstance(equity_data, pd.DataFrame):
                    print(f"   DataFrame shape: {equity_data.shape}")
                    print(f"   Columns: {list(equity_data.columns)}")
                    
                    # Check each column's data type
                    for col in equity_data.columns:
                        dtype = equity_data[col].dtype
                        sample = equity_data[col].iloc[0] if len(equity_data) > 0 else None
                        print(f"   Column '{col}': dtype={dtype}, sample={sample}")
                    
                    # Check for required columns
                    required_for_breadth = ['ltp', 'percent_change', 'volume']
                    missing = [col for col in required_for_breadth if col not in equity_data.columns]
                    if missing:
                        print(f"   ⚠ Missing columns for breadth analysis: {missing}")
                    else:
                        print(f"   ✓ All required columns present for breadth analysis")
                        
                        # Check data quality
                        for col in required_for_breadth:
                            null_count = equity_data[col].isnull().sum()
                            if null_count > 0:
                                print(f"   ⚠ Column '{col}' has {null_count} null values")
                            else:
                                print(f"   ✓ Column '{col}' has no null values")
                else:
                    print(f"   ⚠ Equity data is {type(equity_data)}, expected DataFrame")
                    test_results['errors'].append(f"Equity data type mismatch: {type(equity_data)}")
            else:
                print("❌ Equity quotes fetch returned None")
                test_results['errors'].append("Equity quotes returned None")
        except Exception as e:
            print(f"❌ Equity quotes error: {e}")
            test_results['errors'].append(f"Equity quotes error: {e}")
        
        # TEST 4: Fetch REAL option chain data
        print("\n4. Testing fetch_option_chain_with_analytics() with REAL API call...")
        try:
            # First get REAL option keys
            from data.instrument_master import get_option_keys
            
            expiry_date = (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d")
            print(f"   Getting option keys for NIFTY expiry: {expiry_date}")
            
            option_keys = get_option_keys(
                underlying="NIFTY",
                expiry=expiry_date,
                max_keys=20  # Small number for testing
            )
            
            if option_keys and len(option_keys) > 0:
                print(f"   Got {len(option_keys)} REAL option keys")
                print(f"   Sample: {option_keys[:3]}")
                
                # Get current spot price for realistic test
                if test_results.get('index_quote') and 'ltp' in test_results['index_quote']:
                    spot_price = float(test_results['index_quote']['ltp'])
                    print(f"   Using REAL spot price: {spot_price}")
                    
                    # Test with a few REAL option keys
                    test_option_keys = option_keys[:5]  # Small subset
                    
                    print(f"   Fetching option chain for {len(test_option_keys)} instruments...")
                    option_data = client.fetch_option_chain_with_analytics(
                        test_option_keys,
                        spot_price
                    )
                    
                    if option_data:
                        test_results['option_chain'] = option_data
                        print("✅ REAL option chain fetched")
                        
                        # Check structure
                        if 'raw_data' in option_data:
                            raw_data = option_data['raw_data']
                            if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                                print(f"   Raw data shape: {raw_data.shape}")
                                print(f"   Raw data columns: {list(raw_data.columns)}")
                                
                                # Critical columns for feature pipeline
                                critical_columns = [
                                    'strike', 'call_oi', 'put_oi', 
                                    'call_volume', 'put_volume', 'call_ltp', 'put_ltp'
                                ]
                                
                                for col in critical_columns:
                                    if col in raw_data.columns:
                                        # Check data type and quality
                                        col_data = raw_data[col]
                                        null_count = col_data.isnull().sum()
                                        print(f"   Column '{col}': dtype={col_data.dtype}, nulls={null_count}")
                                        
                                        if null_count == len(col_data):
                                            print(f"     ⚠ ALL values are null!")
                                        elif null_count > 0:
                                            print(f"     ⚠ Has {null_count} null values")
                                        else:
                                            print(f"     ✓ No null values")
                                    else:
                                        print(f"   ⚠ Missing critical column: {col}")
                                        test_results['errors'].append(f"Missing column in option chain: {col}")
                            else:
                                print(f"   ⚠ Raw data is not DataFrame or is empty")
                                if isinstance(raw_data, pd.DataFrame):
                                    print(f"     DataFrame is empty: {raw_data.empty}")
                        
                        if 'analytics' in option_data:
                            analytics = option_data['analytics']
                            print(f"   Analytics keys: {list(analytics.keys())}")
                            
                            # Check important metrics
                            important_metrics = ['put_call_ratio', 'max_pain', 'total_oi', 'total_volume']
                            for metric in important_metrics:
                                if metric in analytics:
                                    value = analytics[metric]
                                    print(f"   Metric '{metric}': {value} (type: {type(value).__name__})")
                                else:
                                    print(f"   ⚠ Missing metric: {metric}")
                                    test_results['errors'].append(f"Missing metric in analytics: {metric}")
                        
                        if 'market_insights' in option_data:
                            insights = option_data['market_insights']
                            print(f"   Market insights: {len(insights)} items")
                            if insights:
                                print(f"   Sample insight: {insights[0][:100]}...")
                    else:
                        print("❌ Option chain fetch returned None")
                        test_results['errors'].append("Option chain returned None")
                else:
                    print("❌ Cannot get spot price for option chain test")
                    test_results['errors'].append("No spot price available")
            else:
                print("❌ No option keys retrieved")
                test_results['errors'].append("No option keys available")
        except Exception as e:
            print(f"❌ Option chain error: {e}")
            test_results['errors'].append(f"Option chain error: {e}")
            import traceback
            traceback.print_exc()
        
        return test_results
        
    except Exception as e:
        print(f"❌ UpstoxClient initialization error: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_feature_pipeline_requirements(test_results):
    """Check if REAL data meets feature pipeline requirements."""
    print("\n" + "=" * 70)
    print("FEATURE PIPELINE DATA REQUIREMENTS CHECK")
    print("=" * 70)
    
    requirements_met = True
    missing_data = []
    
    # Check 1: Index quote data
    if test_results.get('index_quote'):
        print("1. Index Quote Data: ✓ Available")
        
        # Check critical fields
        required_fields = ['ltp', 'open']
        for field in required_fields:
            if field in test_results['index_quote']:
                value = test_results['index_quote'][field]
                try:
                    float(value)
                    print(f"   Field '{field}': ✓ Numeric value: {value}")
                except:
                    print(f"   Field '{field}': ✗ Not numeric: {value}")
                    requirements_met = False
            else:
                print(f"   Field '{field}': ✗ Missing")
                missing_data.append(f"index_quote.{field}")
                requirements_met = False
    else:
        print("1. Index Quote Data: ✗ Missing")
        missing_data.append("index_quote")
        requirements_met = False
    
    # Check 2: Equity quotes for breadth analysis
    if test_results.get('equity_quotes') is not None:
        equity_data = test_results['equity_quotes']
        if isinstance(equity_data, pd.DataFrame) and not equity_data.empty:
            print("2. Equity Quotes Data: ✓ Available")
            print(f"   Shape: {equity_data.shape}")
            
            # Check columns needed for breadth analysis
            breadth_columns = ['ltp', 'percent_change']
            for col in breadth_columns:
                if col in equity_data.columns:
                    null_count = equity_data[col].isnull().sum()
                    if null_count == 0:
                        print(f"   Column '{col}': ✓ No nulls")
                    else:
                        print(f"   Column '{col}': ⚠ Has {null_count} null values")
                        if null_count == len(equity_data):
                            print(f"     ✗ ALL values are null!")
                            requirements_met = False
                else:
                    print(f"   Column '{col}': ✗ Missing")
                    missing_data.append(f"equity_quotes.{col}")
                    requirements_met = False
        else:
            print("2. Equity Quotes Data: ✗ Empty or not DataFrame")
            missing_data.append("equity_quotes.valid_data")
            requirements_met = False
    else:
        print("2. Equity Quotes Data: ✗ Missing")
        missing_data.append("equity_quotes")
        requirements_met = False
    
    # Check 3: Option chain data for features
    if test_results.get('option_chain'):
        option_data = test_results['option_chain']
        print("3. Option Chain Data: ✓ Available")
        
        if 'raw_data' in option_data:
            raw_data = option_data['raw_data']
            if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                print(f"   Raw data shape: {raw_data.shape}")
                
                # Check critical columns for feature calculation
                critical_columns = ['strike', 'call_oi', 'put_oi', 'call_volume', 'put_volume']
                missing_critical = []
                
                for col in critical_columns:
                    if col in raw_data.columns:
                        null_count = raw_data[col].isnull().sum()
                        if null_count == 0:
                            print(f"   Column '{col}': ✓ No nulls")
                        else:
                            print(f"   Column '{col}': ⚠ Has {null_count} null values")
                            if null_count == len(raw_data):
                                print(f"     ✗ ALL values are null!")
                                missing_critical.append(col)
                    else:
                        print(f"   Column '{col}': ✗ Missing")
                        missing_critical.append(col)
                
                if missing_critical:
                    print(f"   ✗ Missing critical columns: {missing_critical}")
                    missing_data.extend([f"option_chain.{col}" for col in missing_critical])
                    requirements_met = False
            else:
                print("   ✗ Raw data is empty or not DataFrame")
                missing_data.append("option_chain.raw_data")
                requirements_met = False
        else:
            print("   ✗ No 'raw_data' in option chain")
            missing_data.append("option_chain.raw_data")
            requirements_met = False
    else:
        print("3. Option Chain Data: ✗ Missing")
        missing_data.append("option_chain")
        requirements_met = False
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if requirements_met:
        print("✅ ALL data requirements met for feature pipeline!")
        print("   Your app should work during market hours.")
    else:
        print("⚠ Some data requirements NOT met:")
        for item in missing_data:
            print(f"   - Missing: {item}")
        print("\n   This might cause issues during market hours.")
        print("   Common fixes:")
        print("   1. Wait for market hours (real data only available then)")
        print("   2. Check API credentials/permissions")
        print("   3. Verify data subscriptions with Upstox")
    
    return requirements_met, missing_data

def main():
    """Run the real API test."""
    print("=" * 70)
    print("REAL UPSTOX API TEST (No Mock Data)")
    print("=" * 70)
    print("This test makes REAL API calls to check data availability and structure.")
    print("It tests exactly what your app will do during market hours.")
    print("=" * 70)
    
    # Check current time
    current_time = datetime.utcnow() + timedelta(hours=5, minutes=30)  # IST
    print(f"Current IST time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if market is open
    market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if market_open <= current_time <= market_close and current_time.weekday() < 5:
        print("✅ MARKET IS OPEN - Testing with LIVE data")
    else:
        print("⚠ MARKET IS CLOSED - Some API calls may fail or return cached data")
        print("   For complete test, run during market hours (9:15 AM - 3:30 PM IST, Mon-Fri)")
    
    print("\n" + "=" * 70)
    print("STEP 1: Getting Access Token")
    print("=" * 70)
    
    # Try to get token
    token = get_access_token()
    
    if not token:
        print("\n❌ Cannot proceed without access token.")
        print("\nTo get a token:")
        print("1. Run: streamlit run app.py")
        print("2. Click 'Login with Upstox'")
        print("3. Complete OAuth flow")
        print("4. Then run this test again")
        return
    
    print("\n" + "=" * 70)
    print("STEP 2: Making REAL API Calls")
    print("=" * 70)
    print("This will make actual API calls to Upstox...")
    
    # Run real API tests
    test_results = test_real_api_calls(token)
    
    if not test_results:
        print("\n❌ API tests failed completely")
        return
    
    print("\n" + "=" * 70)
    print("STEP 3: Checking Feature Pipeline Requirements")
    print("=" * 70)
    
    # Check if data meets requirements
    requirements_met, missing_data = check_feature_pipeline_requirements(test_results)
    
    # Error summary
    if test_results.get('errors'):
        print(f"\n⚠ API Errors encountered ({len(test_results['errors'])}):")
        for error in test_results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(test_results['errors']) > 5:
            print(f"  ... and {len(test_results['errors']) - 5} more")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS FOR MARKET HOURS:")
    print("=" * 70)
    
    if requirements_met:
        print("""
        ✅ YOUR APP IS READY FOR MARKET HOURS!
        
        Tomorrow during market hours (9:15 AM - 3:30 PM IST):
        1. Start the app: streamlit run app.py
        2. It should automatically:
           - Fetch live Nifty 50 data
           - Get option chain data
           - Calculate features (OI velocity, gamma, etc.)
           - Generate trading signals
           - Update the dashboard in real-time
        
        Monitor these tabs:
        - 📈 Live Analytics: Real-time charts and metrics
        - 🎯 Signals: Generated trading signals
        - 🔍 Research: Detailed market analysis
        """)
    else:
        print("""
        ⚠ SOME ISSUES DETECTED
        
        Before market hours, check:
        1. Upstox API credentials in secrets.toml
        2. Data subscriptions (Options data might need separate subscription)
        3. Network connectivity to Upstox API
        
        During market hours, if issues persist:
        1. Check browser console for errors
        2. Look at Streamlit logs for API errors
        3. Verify your Upstox account has access to:
           - Live market data
           - Options chain data
           - Real-time quotes
        
        Missing data points:""")
        for item in missing_data:
            print(f"  - {item}")
    
    print("\n" + "=" * 70)
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    # Warn user about API calls
    print("\n⚠ WARNING: This test makes REAL API calls to Upstox.")
    print("   It will use your API quota and fetch real market data.")
    print("   Only run this if you have valid Upstox credentials.")
    print("   Press Ctrl+C to cancel.")
    
    try:
        import time
        for i in range(5, 0, -1):
            print(f"Starting in {i} seconds...", end='\r')
            time.sleep(1)
        print("\n" + "=" * 70)
        
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Test cancelled by user")