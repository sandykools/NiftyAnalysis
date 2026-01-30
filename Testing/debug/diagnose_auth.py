import os
import sys
from pathlib import Path

print("=" * 60)
print("AUTHENTICATION DIAGNOSIS")
print("=" * 60)

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check if secrets.toml exists
secrets_path = ".streamlit/secrets.toml"
print(f"\n1. Checking secrets.toml:")
print(f"   Path: {secrets_path}")
print(f"   Exists: {os.path.exists(secrets_path)}")

if os.path.exists(secrets_path):
    try:
        with open(secrets_path, 'r') as f:
            content = f.read()
            print(f"   Content preview: {content[:200]}...")
    except Exception as e:
        print(f"   Error reading: {e}")

# Check token directory
token_dir = "data/tokens"
print(f"\n2. Checking token directory:")
print(f"   Path: {token_dir}")
print(f"   Exists: {os.path.exists(token_dir)}")
if os.path.exists(token_dir):
    files = os.listdir(token_dir)
    print(f"   Files: {files}")

# Check .env file
env_file = ".env"
print(f"\n3. Checking .env file:")
print(f"   Path: {env_file}")
print(f"   Exists: {os.path.exists(env_file)}")
if os.path.exists(env_file):
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            # Hide sensitive info
            safe_content = content.replace('\n', '\\n').replace('UPSTOX_', '[REDACTED]_')
            print(f"   Content: {safe_content[:100]}...")
    except Exception as e:
        print(f"   Error reading: {e}")

# Check session.py logic
print(f"\n4. Testing authentication logic...")
try:
    # Simulate session.py logic
    class MockSession:
        def __init__(self):
            self.upstox_client = None
    
    session = MockSession()
    
    # Check if we should show login
    has_tokens = os.path.exists("data/tokens/upstox_tokens.enc")
    print(f"   Token file exists: {has_tokens}")
    print(f"   Should show login: {not has_tokens}")
    
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)