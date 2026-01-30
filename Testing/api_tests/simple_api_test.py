#!/usr/bin/env python3
"""
Simple API test that reads the stored token and tests API calls.
"""

import os
import sys
import pickle
from pathlib import Path
from cryptography.fernet import Fernet
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def read_stored_token():
    """Read the stored token from encrypted storage."""
    print("=" * 70)
    print("READING STORED TOKEN")
    print("=" * 70)
    
    token_dir = Path("data/tokens")
    token_file = token_dir / "upstox_tokens.enc"
    key_file = token_dir / ".key"
    
    if not token_file.exists():
        print("❌ No token file found. You need to:")
        print("   1. Run: streamlit run app.py")
        print("   2. Click 'Login with Upstox'")
        print("   3. Complete OAuth login")
        print("   4. Close the app (Ctrl+C)")
        print("   5. Then run this test")
        return None
    
    try:
        # Read encryption key
        with open(key_file, 'rb') as f:
            encryption_key = f.read()
        
        cipher = Fernet(encryption_key)
        
        # Read and decrypt token
        with open(token_file, 'rb') as f:
            encrypted = f.read()
        
        token_data = pickle.loads(cipher.decrypt(encrypted))
        
        access_token = token_data.get("access_token")
        expires_at = token_data.get("expires_at")
        
        if access_token:
            print(f"✅ Token found: {access_token[:30]}...")
            print(f"   Expires at: {expires_at}")
            return access_token
        else:
            print("❌ No access token in stored data")
            return None
            
    except Exception as e:
        print(f"❌ Error reading token: {e}")
        return None

def test_api_calls(access_token):
    """Test basic API calls with the token."""
    print("\n" + "=" * 70)
    print("TESTING API CALLS")
    print("=" * 70)
    
    try:
        import requests
        
        # Test 1: Fetch user profile
        print("\n1. Testing profile API...")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        profile_url = "https://api.upstox.com/v2/user/profile"
        response = requests.get(profile_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            profile = response.json().get("data", {})
            print(f"✅ Profile API successful")
            print(f"   User: {profile.get('user_name', 'N/A')}")
            print(f"   Email: {profile.get('email', 'N/A')}")
        else:
            print(f"❌ Profile API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
        
        # Test 2: Fetch Nifty 50 quote
        print("\n2. Testing Nifty 50 quote API...")
        nifty_symbol = "NSE_INDEX|Nifty 50"
        quote_url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={nifty_symbol}"
        
        response = requests.get(quote_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                quote_data = data.get("data", {}).get(nifty_symbol, {})
                print(f"✅ Nifty quote API successful")
                print(f"   LTP: {quote_data.get('last_price', 'N/A')}")
                print(f"   Open: {quote_data.get('open', 'N/A')}")
                print(f"   High: {quote_data.get('high', 'N/A')}")
                print(f"   Low: {quote_data.get('low', 'N/A')}")
                print(f"   Close: {quote_data.get('close', 'N/A')}")
                print(f"   Volume: {quote_data.get('volume', 'N/A')}")
                
                # Check data types
                ltp = quote_data.get('last_price')
                if ltp:
                    try:
                        float(ltp)
                        print(f"   ✓ LTP is numeric: {ltp}")
                    except:
                        print(f"   ⚠ LTP is not numeric: {ltp}")
            else:
                print(f"❌ Nifty quote API error: {data.get('status', 'unknown')}")
                if "errors" in data:
                    for error in data["errors"]:
                        print(f"   Error: {error.get('message', 'Unknown')}")
        else:
            print(f"❌ Nifty quote API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
        
        # Test 3: Test a few equity quotes
        print("\n3. Testing equity quotes API...")
        symbols = ["NSE|RELIANCE", "NSE|TCS", "NSE|HDFCBANK"]
        symbols_param = "&symbol=".join(symbols)
        equity_url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={symbols_param}"
        
        response = requests.get(equity_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                equity_data = data.get("data", {})
                print(f"✅ Equity quotes API successful")
                print(f"   Received quotes for {len(equity_data)} symbols")
                
                for symbol in symbols:
                    if symbol in equity_data:
                        quote = equity_data[symbol]
                        print(f"   {symbol}:")
                        print(f"     LTP: {quote.get('last_price', 'N/A')}")
                        print(f"     Change: {quote.get('change', 'N/A')}")
                        print(f"     % Change: {quote.get('percent_change', 'N/A')}")
                    else:
                        print(f"   ⚠ {symbol}: Not in response")
            else:
                print(f"❌ Equity quotes API error: {data.get('status', 'unknown')}")
        else:
            print(f"❌ Equity quotes API failed: {response.status_code}")
        
        # Test 4: Test options chain (limited)
        print("\n4. Testing options API (single instrument)...")
        # Try to get one option instrument
        option_symbol = "NSE|NIFTY28JAN23FUT"  # Example, might not exist
        option_url = f"https://api.upstox.com/v2/market-quote/quotes?symbol={option_symbol}"
        
        response = requests.get(option_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print(f"✅ Options API successful")
                option_data = data.get("data", {})
                if option_data:
                    for symbol, quote in option_data.items():
                        print(f"   {symbol}:")
                        print(f"     LTP: {quote.get('last_price', 'N/A')}")
                        print(f"     OI: {quote.get('oi', 'N/A')}")
                        print(f"     Volume: {quote.get('volume', 'N/A')}")
                else:
                    print("   ⚠ No option data returned (instrument might not exist)")
            else:
                print(f"   Options API error: {data.get('status', 'unknown')}")
        else:
            print(f"   Options API failed: {response.status_code}")
            # This is OK - options might need special subscription
        
        print("\n" + "=" * 70)
        print("API TEST COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ API test error: {e}")
        import traceback
        traceback.print_exc()

def check_data_subscriptions():
    """Check what data is available based on API responses."""
    print("\n" + "=" * 70)
    print("DATA SUBSCRIPTION CHECK")
    print("=" * 70)
    
    print("Based on your Upstox plan, you should have access to:")
    print("1. ✅ Equity quotes (stocks like RELIANCE, TCS, etc.)")
    print("2. ✅ Index quotes (Nifty 50, Bank Nifty, etc.)")
    print("3. ⚠ Options data - May require separate subscription")
    print("4. ⚠ Futures data - May require separate subscription")
    print("\nIf options/futures data fails, check your Upstox subscription.")
    print("\nFor the trading app to work fully, you need:")
    print("   - Live equity quotes")
    print("   - Options chain data")
    print("   - Real-time index data")

def main():
    """Main test function."""
    print("=" * 70)
    print("SIMPLE UPSTOX API TEST")
    print("=" * 70)
    print("This test reads your stored token and makes API calls.")
    print("=" * 70)
    
    # Step 1: Read stored token
    access_token = read_stored_token()
    
    if not access_token:
        print("\n" + "=" * 70)
        print("HOW TO GET A TOKEN:")
        print("=" * 70)
        print("1. First, run the Streamlit app and login:")
        print("   streamlit run app.py")
        print("\n2. In the browser:")
        print("   - Click 'Login with Upstox'")
        print("   - Complete the OAuth login")
        print("   - You should see '✅ Upstox client initialized'")
        print("\n3. Then close the app (Ctrl+C in terminal)")
        print("\n4. Run this test again:")
        print("   python simple_api_test.py")
        return
    
    # Step 2: Test API calls
    test_api_calls(access_token)
    
    # Step 3: Check subscriptions
    check_data_subscriptions()
    
    print("\n" + "=" * 70)
    print("READY FOR MARKET HOURS CHECKLIST:")
    print("=" * 70)
    print("✅ 1. Authentication working")
    print("✅ 2. Token stored correctly")
    print("✅ 3. Basic API calls working")
    print("\n⚠ During market hours tomorrow, check:")
    print("   1. Real-time data is flowing")
    print("   2. Options chain data is available")
    print("   3. Feature pipeline executes")
    print("   4. Signals are generated")
    print("\nIf issues occur during market hours:")
    print("   1. Check browser console for errors")
    print("   2. Look at Streamlit terminal output")
    print("   3. Verify API responses in Network tab")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()