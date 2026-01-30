# Create: debug_token.py
import os
import pickle
from pathlib import Path
from cryptography.fernet import Fernet
from datetime import datetime

def debug_token():
    print("=" * 70)
    print("DEBUGGING TOKEN STORAGE")
    print("=" * 70)
    
    token_dir = Path("data/tokens")
    token_file = token_dir / "upstox_tokens.enc"
    key_file = token_dir / ".key"
    
    if not token_file.exists():
        print("❌ No token file found at:", token_file)
        print("\nThe app should show a login button.")
        print("If it's not showing, there's an issue in the authentication flow.")
        return
    
    try:
        # Read encryption key
        with open(key_file, 'rb') as f:
            encryption_key = f.read()
        
        cipher = Fernet(encryption_key)
        
        # Read and decrypt token
        with open(token_file, 'rb') as f:
            encrypted = f.read()
        
        token_data = pickle.loads(cipher.decrypt(encrypted))
        
        print("✅ Token file found and decrypted")
        print("\nToken data:")
        for key, value in token_data.items():
            if key == 'access_token':
                print(f"  {key}: {value[:30]}...")
            elif key in ['expires_at', 'created_at']:
                print(f"  {key}: {value}")
                # Check if expired
                try:
                    expires = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    now = datetime.utcnow()
                    if now > expires:
                        print(f"    ⚠ EXPIRED! (Current: {now}, Expired: {expires})")
                    else:
                        print(f"    ✓ Still valid")
                except:
                    print(f"    ⚠ Cannot parse date")
            else:
                print(f"  {key}: {value}")
        
        # Check if token is valid by making a simple API call
        print("\n" + "=" * 70)
        print("TESTING TOKEN VALIDITY")
        print("=" * 70)
        
        import requests
        access_token = token_data.get('access_token')
        
        if access_token:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}"
            }
            
            # Try to fetch profile
            try:
                response = requests.get(
                    "https://api.upstox.com/v2/user/profile",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    print("✅ Token is VALID - API call successful")
                    profile = response.json().get("data", {})
                    print(f"   User: {profile.get('user_name', 'N/A')}")
                else:
                    print(f"❌ Token is INVALID - API error: {response.status_code}")
                    print(f"   Response: {response.text[:200]}")
                    
                    if response.status_code == 401:
                        print("\n⚠ Token is expired or invalid.")
                        print("   The app should show a login button.")
                        print("   If it's not showing, there's a bug in UpstoxSession.authenticate()")
            except Exception as e:
                print(f"❌ API call failed: {e}")
        else:
            print("❌ No access token in stored data")
        
    except Exception as e:
        print(f"❌ Error reading token: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_token()