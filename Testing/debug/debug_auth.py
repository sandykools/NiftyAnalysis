#!/usr/bin/env python3
"""
Debug script to trace authentication flow.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def trace_authentication():
    """Trace the authentication flow step by step."""
    
    print("=" * 70)
    print("AUTHENTICATION DEBUG TRACE")
    print("=" * 70)
    
    # 1. Check if we can import the session module
    print("\n1. Importing core.session...")
    try:
        from core.session import UpstoxSession, CLIENT_ID, CLIENT_SECRET, REDIRECT_URI
        print(f"   ✓ Import successful")
        print(f"   CLIENT_ID: {'✓ SET' if CLIENT_ID else '✗ NOT SET'}")
        print(f"   CLIENT_SECRET: {'✓ SET' if CLIENT_SECRET else '✗ NOT SET'}")
        print(f"   REDIRECT_URI: {REDIRECT_URI}")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        return
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # 2. Test the authenticate method step by step
    print("\n2. Testing UpstoxSession.authenticate() step by step:")
    
    # Mock streamlit for testing
    class MockStreamlit:
        def __init__(self):
            self.secrets = {
                'UPSTOX_CLIENT_ID': CLIENT_ID,
                'UPSTOX_CLIENT_SECRET': CLIENT_SECRET,
                'UPSTOX_REDIRECT_URI': REDIRECT_URI
            }
            self.query_params = {}
            self.session_state = {}
            self._calls = []
        
        def markdown(self, text, unsafe_allow_html=False):
            self._calls.append(('markdown', text[:100]))
            print(f"   [STREAMLIT] markdown() called")
            if "Login" in text:
                print(f"     ↳ Contains 'Login': ✓")
        
        def error(self, text):
            self._calls.append(('error', text))
            print(f"   [STREAMLIT] error(): {text}")
        
        def success(self, text):
            self._calls.append(('success', text))
            print(f"   [STREAMLIT] success(): {text}")
        
        def warning(self, text):
            self._calls.append(('warning', text))
            print(f"   [STREAMLIT] warning(): {text}")
        
        def info(self, text):
            self._calls.append(('info', text))
            print(f"   [STREAMLIT] info(): {text}")
        
        def stop(self):
            self._calls.append(('stop', ''))
            print(f"   [STREAMLIT] stop() called")
            raise SystemExit("Streamlit stop")
        
        def rerun(self):
            self._calls.append(('rerun', ''))
            print(f"   [STREAMLIT] rerun() called")
    
    # Replace streamlit module with mock
    import core.session as session_module
    original_st = sys.modules.get('streamlit')
    mock_st = MockStreamlit()
    sys.modules['streamlit'] = mock_st
    
    try:
        # Re-import to use mocked streamlit
        import importlib
        importlib.reload(session_module)
        from core.session import UpstoxSession
        
        print(f"   Testing UpstoxSession.is_authenticated(): {UpstoxSession.is_authenticated()}")
        
        # Test get_login_url
        try:
            login_url = UpstoxSession.get_login_url()
            print(f"   Login URL generated: ✓")
            print(f"     URL length: {len(login_url)} chars")
            print(f"     Contains 'api.upstox.com': {'✓' if 'api.upstox.com' in login_url else '✗'}")
            print(f"     Contains 'client_id': {'✓' if 'client_id' in login_url else '✗'}")
        except Exception as e:
            print(f"   ✗ Error generating login URL: {e}")
        
        # Try to call authenticate
        print(f"\n   Attempting to call authenticate()...")
        try:
            token = UpstoxSession.authenticate()
            if token:
                print(f"   ✓ authenticate() returned token (length: {len(token)})")
            else:
                print(f"   ⚠ authenticate() returned None")
        except SystemExit as e:
            print(f"   ⚠ authenticate() called st.stop() (expected when not authenticated)")
        except Exception as e:
            print(f"   ✗ Error in authenticate(): {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n   Streamlit calls made:")
        for i, (func, arg) in enumerate(mock_st._calls, 1):
            print(f"     {i}. {func}({str(arg)[:80]}...)")
        
    finally:
        # Restore original streamlit
        if original_st:
            sys.modules['streamlit'] = original_st
        else:
            del sys.modules['streamlit']
    
    # 3. Check if there's an issue with the app.py authentication call
    print("\n3. Checking app.py authentication section...")
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find the authentication section
        import re
        auth_section = re.search(r'# =+.*AUTHENTICATION.*?access_token = UpstoxSession\.authenticate.*?(?=\n#|\Z)', 
                                content, re.DOTALL | re.IGNORECASE)
        if auth_section:
            print(f"   Found authentication section in app.py: ✓")
            section = auth_section.group(0)
            lines = section.split('\n')
            for line in lines[:10]:
                if 'authenticate' in line or 'access_token' in line:
                    print(f"     {line.strip()}")
        else:
            print(f"   ✗ Could not find authentication section in app.py")
            
    except Exception as e:
        print(f"   ✗ Error reading app.py: {e}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED FIX:")
    print("=" * 70)
    print("""
    1. The issue is likely in app.py where UpstoxSession.authenticate() is called.
    2. Let's modify app.py to handle authentication better.
    
    Change this section in app.py (around line 118):
    
    FROM:
    # ==============================
    # AUTHENTICATION
    # ==============================
    
    st.markdown('<div class="sub-header">🔐 Authentication</div>', unsafe_allow_html=True)
    
    access_token = UpstoxSession.authenticate()
    if not access_token:
        st.error("❌ Authentication failed. Please login with Upstox.")
        st.stop()
    
    TO:
    # ==============================
    # AUTHENTICATION
    # ==============================
    
    st.markdown('<div class="sub-header">🔐 Authentication</div>', unsafe_allow_html=True)
    
    try:
        access_token = UpstoxSession.authenticate()
        if not access_token:
            # This means authenticate() showed login button and stopped
            st.stop()
    except SystemExit:
        # authenticate() called st.stop() to show login button
        st.stop()
    except Exception as e:
        st.error(f"❌ Authentication error: {e}")
        st.stop()
    """)
    
    print("\nRun this script and share the output!")

if __name__ == "__main__":
    trace_authentication()