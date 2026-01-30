import requests
import json

def test_upstox_api():
    # Replace with your actual access token
    ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"
    
    # Test different endpoints
    endpoints = [
        {
            "name": "index-quote (POST)",
            "url": "https://api.upstox.com/v2/market-quote/index-quote",
            "method": "POST",
            "data": {"index_key": "NIFTY 50"}
        },
        {
            "name": "quote (GET with symbol)",
            "url": "https://api.upstox.com/v2/market-quote/quote",
            "method": "GET",
            "params": {"symbol": "NIFTY 50"}
        },
        {
            "name": "quote (GET with NSE_INDEX)",
            "url": "https://api.upstox.com/v2/market-quote/quote",
            "method": "GET",
            "params": {"symbol": "NSE_INDEX|Nifty 50"}
        }
    ]
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    for endpoint in endpoints:
        print(f"\n{'='*50}")
        print(f"Testing: {endpoint['name']}")
        print(f"URL: {endpoint['url']}")
        
        try:
            if endpoint["method"] == "POST":
                response = requests.post(
                    endpoint["url"],
                    json=endpoint.get("data"),
                    headers=headers,
                    timeout=10
                )
            else:
                response = requests.get(
                    endpoint["url"],
                    params=endpoint.get("params"),
                    headers=headers,
                    timeout=10
                )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("Success!")
                data = response.json()
                print(f"Response Keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
                print(f"Response Preview: {json.dumps(data, indent=2)[:500]}...")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_upstox_api()