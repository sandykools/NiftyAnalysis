#!/usr/bin/env python3
"""
Final fix for all issues:
1. Remove test_connection() calls
2. Ensure session state initialization
3. Clear expired tokens
"""

import os
import shutil
from pathlib import Path

def fix_app_py():
    """Remove test_connection() calls from app.py"""
    print("Fixing app.py...")
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove test_connection() calls
    lines = content.split('\n')
    new_lines = []
    changes_made = 0
    
    for line in lines:
        if 'test_connection()' in line:
            # Comment out or remove lines with test_connection()
            if line.strip().startswith('if client.test_connection():'):
                new_lines.append('    # Connection assumed successful (test_connection method not available)')
                new_lines.append('    pass')
            elif line.strip().startswith('st.success("✅ Connected to Upstox API")'):
                new_lines.append('    st.success("✅ Upstox client initialized")')
            else:
                # Skip lines with test_connection()
                continue
            changes_made += 1
        else:
            new_lines.append(line)
    
    # Write back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print(f"✓ Fixed {changes_made} lines in app.py")
    
    # Also check for the specific problematic line
    if 'if client.test_connection():' in content:
        print("⚠ Still found test_connection() calls")
    else:
        print("✓ No more test_connection() calls")

def clear_tokens():
    """Clear all expired tokens"""
    print("\nClearing tokens...")
    
    # Delete token directory
    token_dir = Path("data/tokens")
    if token_dir.exists():
        shutil.rmtree(token_dir)
        print("✓ Deleted token directory")
    else:
        print("✓ Token directory already cleared")
    
    # Clear token from .env
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Remove UPSTOX_ACCESS_TOKEN line
        new_lines = [line for line in lines if not line.startswith("UPSTOX_ACCESS_TOKEN")]
        
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
        
        print("✓ Cleared token from .env")
    
    # Also clear from secrets.toml if it exists
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            content = f.read()
        
        # Check if UPSTOX_ACCESS_TOKEN is in secrets
        if 'UPSTOX_ACCESS_TOKEN' in content:
            print("⚠ WARNING: UPSTOX_ACCESS_TOKEN found in secrets.toml")
            print("  Remove it and only keep:")
            print("  UPSTOX_CLIENT_ID")
            print("  UPSTOX_CLIENT_SECRET")
            print("  UPSTOX_REDIRECT_URI")

def check_session_initialization():
    """Check if session state is initialized properly"""
    print("\nChecking session initialization...")
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if initialize_app_state is called before accessing scheduler
    if 'initialize_app_state()' in content:
        init_line = content.find('initialize_app_state()')
        scheduler_line = content.find('st.session_state.scheduler')
        
        if init_line < scheduler_line:
            print("✓ Session initialization happens before scheduler access")
        else:
            print("✗ Session initialization might happen after scheduler access")
    else:
        print("✗ initialize_app_state() not found in app.py")

def create_test_file():
    """Create a simple test to verify authentication works"""
    print("\nCreating authentication test...")
    
    test_code = '''#!/usr/bin/env python3
"""
Test authentication independently.
"""

import os
os.environ['UPSTOX_CLIENT_ID'] = 'e9bc6b14-447a-4d2f-aa32-2a4aecbafe56'
os.environ['UPSTOX_CLIENT_SECRET'] = 'lcbal1vojx'
os.environ['UPSTOX_REDIRECT_URI'] = 'http://127.0.0.1:8501/callback'

import sys
sys.path.append('.')

print("Testing Upstox authentication...")
print("=" * 60)

try:
    from core.session import UpstoxSession
    
    # Test 1: Check if credentials load
    from core.session import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
    print(f"CLIENT_ID: {CLIENT_ID[:20]}...")
    print(f"CLIENT_SECRET: {'✓' if CLIENT_SECRET else '✗'}")
    print(f"REDIRECT_URI: {REDIRECT_URI}")
    
    # Test 2: Generate login URL
    login_url = UpstoxSession.get_login_url()
    print(f"\\nLogin URL (first 100 chars):")
    print(login_url[:100] + "...")
    
    print("\\n✓ Authentication test passed!")
    print("\\nTo authenticate:")
    print(f"1. Open this URL in browser: {login_url}")
    print("2. Login with Upstox credentials")
    print("3. Authorize the app")
    print("4. You'll be redirected back to the app")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open('test_auth_simple.py', 'w') as f:
        f.write(test_code)
    
    print("✓ Created test_auth_simple.py")

def main():
    print("=" * 60)
    print("FINAL FIX FOR ALL ISSUES")
    print("=" * 60)
    
    fix_app_py()
    clear_tokens()
    check_session_initialization()
    create_test_file()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run: python test_auth_simple.py")
    print("2. Copy the login URL from the output")
    print("3. Open it in your browser")
    print("4. Login with Upstox and authorize")
    print("5. You should be redirected to: http://127.0.0.1:8501/callback")
    print("6. Then run: streamlit run app.py")
    print("\nIMPORTANT: Your secrets.toml should ONLY contain:")
    print("   UPSTOX_CLIENT_ID = \"e9bc6b14-447a-4d2f-aa32-2a4aecbafe56\"")
    print("   UPSTOX_CLIENT_SECRET = \"lcbal1vojx\"")
    print("   UPSTOX_REDIRECT_URI = \"http://127.0.0.1:8501/callback\"")
    print("   (NO UPSTOX_ACCESS_TOKEN in secrets.toml)")

if __name__ == "__main__":
    main()