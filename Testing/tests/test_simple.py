import requests
import json

def test_with_live_token():
    """Test with token from your UpstoxSession"""
    
    # Import your UpstoxSession to get a fresh token
    try:
        from core.session import UpstoxSession
        access_token = UpstoxSession.authenticate()
        
        if not access_token:
            print("❌ Could not get access token")
            return
        
        print(f"✅ Got token: {access_token[:30]}...")
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        # Test the working format
        symbol = "NSE_INDEX|Nifty 50"
        
        response = requests.get(
            "https://api.upstox.com/v2/market-quote/quotes",
            params={"symbol": symbol},
            headers=headers,
            timeout=10
        )
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                # Response key uses colon, not pipe
                response_key = symbol.replace("|", ":")
                quote = data["data"].get(response_key, {})
                
                print("✅ SUCCESS!")
                print(f"Symbol: {symbol}")
                print(f"LTP: {quote.get('last_price')}")
                print(f"Open: {quote.get('ohlc', {}).get('open')}")
                print(f"High: {quote.get('ohlc', {}).get('high')}")
                print(f"Low: {quote.get('ohlc', {}).get('low')}")
                print(f"Close: {quote.get('ohlc', {}).get('close')}")
                print(f"Change: {quote.get('net_change')}")
            else:
                print(f"❌ API Error: {data}")
        else:
            print(f"❌ HTTP Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_with_live_token()