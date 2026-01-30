print("Checking session.py logic...")

# The key issue is likely in the `authenticate()` method
# It should return True (need authentication) when tokens don't exist
# Here's what it should look like:

correct_logic = """
def authenticate():
    \"""Check if user is authenticated, show login if not.\"""
    # Check if tokens exist
    token_path = Path("data/tokens/upstox_tokens.enc")
    
    if not token_path.exists():
        # SHOW LOGIN BUTTON
        st.sidebar.markdown("### 🔐 Authentication Required")
        if st.sidebar.button("Login with Upstox", type="primary", use_container_width=True):
            # Initiate OAuth flow
            auth_url = upstox_auth.get_authorization_url()
            st.session_state.auth_url = auth_url
            st.rerun()
        return False
    
    try:
        # Try to load and validate tokens
        # ... existing token loading logic ...
        return True
    except Exception as e:
        # Token loading failed
        st.error(f"Authentication error: {e}")
        # Show login button again
        if st.button("Re-login with Upstox"):
            clear_tokens()
            st.rerun()
        return False
"""

print("\nMake sure your session.py has this pattern:")
print("- Check if token file exists FIRST")
print("- If NOT exists → show login button immediately")
print("- Don't try to load tokens if file doesn't exist")