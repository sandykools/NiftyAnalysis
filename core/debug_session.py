import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide", page_title="Debug Auth")

st.title("🔍 Authentication Debug")

# Check token file
token_path = Path("data/tokens/upstox_tokens.enc")
st.write(f"Token path: {token_path}")
st.write(f"Token exists: {token_path.exists()}")

# Simulate what authenticate() should do
if not token_path.exists():
    st.success("✅ Perfect! Token file doesn't exist - should show login button")
    
    # Show what the login button should look like
    st.markdown("---")
    st.subheader("This is what should appear:")
    
    login_url = "https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id=e9bc6b14-447a-4d2f-aa32-2a4aecbafe56&redirect_uri=http://127.0.0.1:8501/callback&scope=order placement portfolio"
    
    st.markdown(f"""
    <div style="text-align: center;">
        <a href="{login_url}" target="_blank">
            <button style="
                background-color: #00d09c;
                color: white;
                padding: 15px 30px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                width: 80%;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
            📈 Login with Upstox
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("If you don't see this in the main app, the issue is in how authenticate() is being called.")
else:
    st.warning("Token file exists - try deleting it with: `rm data/tokens/upstox_tokens.enc`")

# Check how authenticate is called in app.py
st.markdown("---")
st.subheader("How authenticate() is called in app.py:")
st.code("""
# This should be in your app.py:
from core.session import UpstoxSession

# Early in the app initialization:
token = UpstoxSession.authenticate()

# If token is None, authenticate() should have shown login button
# and stopped execution with st.stop()
""")