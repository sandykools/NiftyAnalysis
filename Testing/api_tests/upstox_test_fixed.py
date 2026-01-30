import requests
import json
from datetime import datetime


def test_upstox_api_fixed(access_token: str):
    """Test the actual working endpoints"""
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    # Test endpoints based on actual Upstox documentation
    test_cases = [
        {
            "name": "GET quotes (main endpoint)",
            "method": "GET",
            "url": "https://api.upstox.com/v2/market-quote/quotes",
            "params": {"symbol": "NIFTY 50"}
        },
        {
            "name": "GET quotes with NSE_INDEX",
            "method": "GET",
            "url": "https://api.upstox.com/v2/market-quote/quotes",
            "params": {"symbol": "NSE_INDEX|Nifty 50"}
        },
        {
            "name": "GET quotes - NIFTY BANK",
            "method": "GET",
            "url": "https://api.upstox.com/v2/market-quote/quotes",
            "params": {"symbol": "NIFTY BANK"}
        },
        {
            "name": "GET quotes - BANKNIFTY",
            "method": "GET",
            "url": "https://api.upstox.com/v2/market-quote/quotes",
            "params": {"symbol": "BANKNIFTY"}
        },
        {
            "name": "GET quotes - multiple symbols",
            "method": "GET",
            "url": "https://api.upstox.com/v2/market-quote/quotes",
            "params": {"symbol": "NIFTY 50,NIFTY BANK"}
        }
    ]
    
    print(f"\n{'='*60}")
    print(f"Upstox API Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Token preview: {access_token[:30]}...")
    print('='*60)
    
    for test in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {test['name']}")
        print(f"URL: {test['url']}")
        print(f"Params: {test.get('params')}")
        
        try:
            if test["method"] == "GET":
                response = requests.get(
                    test["url"],
                    params=test.get("params"),
                    headers=headers,
                    timeout=10
                )
            else:
                response = requests.post(
                    test["url"],
                    json=test.get("data"),
                    headers=headers,
                    timeout=10
                )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ SUCCESS!")
                data = response.json()
                
                # Pretty print the response structure
                print("\nResponse Structure:")
                print(f"Status: {data.get('status')}")
                
                if data.get("status") == "success":
                    data_dict = data.get("data", {})
                    print(f"Number of symbols returned: {len(data_dict)}")
                    
                    for key, value in data_dict.items():
                        print(f"\nSymbol: {key}")
                        print(f"  Last Price: {value.get('last_price')}")
                        print(f"  Change: {value.get('change')}")
                        
                        ohlc = value.get("ohlc", {})
                        print(f"  OHLC: {ohlc.get('open')}/{ohlc.get('high')}/{ohlc.get('low')}/{ohlc.get('close')}")
                        
                        # Just show first symbol details
                        break
                else:
                    print(f"Error in response: {data}")
                    
            elif response.status_code == 401:
                print("❌ 401 UNAUTHORIZED - Invalid token")
                print(f"Response: {response.text}")
                print("\n⚠️  TOKEN ISSUE DETECTED!")
                print("Your access token is invalid or expired.")
                print("Please regenerate your token from Upstox Developer Console.")
                
            elif response.status_code == 400:
                print("❌ 400 BAD REQUEST")
                print(f"Response: {response.text}")
                
            else:
                print(f"❌ Error {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Exception: {e}")


if __name__ == "__main__":
    # Replace with your actual token
    ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiJBRjY1NzAiLCJqdGkiOiI2OTc1ZmJlNjg5M2Y0MDY1MjE3YmUwZWQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc2OTMzOTg3OCwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzY5Mzc4NDAwfQ.eEUnBABSTpvvTaHTmn8Ms9bmixB-LKkcGENrTdKIp_A"
    
    if ACCESS_TOKEN.startswith("eyJ0eXAiOiJKV1Qi"):
        test_upstox_api_fixed(ACCESS_TOKEN)
    else:
        print("Please update the ACCESS_TOKEN variable with your actual token")