#!/usr/bin/env python3
"""
Force re-login by clearing all tokens and starting fresh.
"""

import os
import shutil
from pathlib import Path

def force_login():
    print("=" * 70)
    print("FORCING RE-LOGIN")
    print("=" * 70)
    
    # 1. Delete token directory
    token_dir = Path("data/tokens")
    if token_dir.exists():
        shutil.rmtree(token_dir)
        print("✓ Deleted token directory")
    else:
        print("✓ No token directory found")
    
    # 2. Clear .env file
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Remove UPSTOX_ACCESS_TOKEN line
        new_lines = [line for line in lines if not line.startswith("UPSTOX_ACCESS_TOKEN")]
        
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
        print("✓ Cleared token from .env")
    
    # 3. Clear Streamlit cache
    cache_dir = Path("__pycache__")
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("✓ Cleared __pycache__")
    
    # 4. Clear .streamlit cache if exists
    streamlit_cache = Path(".streamlit/__pycache__")
    if streamlit_cache.exists():
        shutil.rmtree(streamlit_cache, ignore_errors=True)
        print("✓ Cleared .streamlit cache")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Run: streamlit run app.py")
    print("2. You SHOULD see a 'Login with Upstox' button")
    print("3. Click it and complete OAuth")
    print("4. If you don't see login button, there's a bug in core/session.py")
    print("\nIf still no login button, check:")
    print("   - core/session.py authenticate() method")
    print("   - That CLIENT_ID, CLIENT_SECRET are in secrets.toml")
    print("   - Browser console for errors")

if __name__ == "__main__":
    force_login()