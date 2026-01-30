"""
Check if the redirect URI is correctly configured
"""
import streamlit as st

st.title("🔍 Redirect URI Checker")

# Check secrets
try:
    from core.session import get_config
    config = get_config()
    
    st.write("### Current Configuration:")
    st.json(config)
    
    st.write("### Testing Redirect URI:")
    redirect_uri = config['redirect_uri']
    st.write(f"Redirect URI: `{redirect_uri}`")
    
    # Check if this matches what's in Upstox Developer Portal
    st.info("""
    **Make sure this redirect URI matches exactly what's configured in:**
    1. Go to https://upstox.com/developer/
    2. Find your app
    3. Check the Redirect URI setting
    4. It MUST be: `http://127.0.0.1:8501/callback`
    """)
    
    # Test if we can generate a proper login URL
    from core.session import UpstoxSession
    login_url = UpstoxSession.get_login_url()
    
    st.write("### Generated Login URL:")
    st.code(login_url, language=None)
    
    # Extract redirect URI from login URL
    import urllib.parse
    parsed = urllib.parse.urlparse(login_url)
    params = urllib.parse.parse_qs(parsed.query)
    
    st.write("### Extracted Parameters:")
    st.json(params)
    
    if params.get('redirect_uri', [''])[0] == redirect_uri:
        st.success("✅ Redirect URI matches!")
    else:
        st.error(f"❌ Redirect URI mismatch!")
        st.write(f"Expected: {redirect_uri}")
        st.write(f"Got in URL: {params.get('redirect_uri', [''])[0]}")
        
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.error(traceback.format_exc())