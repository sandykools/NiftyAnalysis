import streamlit as st
from pathlib import Path

# Minimal test app
st.set_page_config(layout="wide")

# Check credentials
CLIENT_ID = "e9bc6b14-447a-4d2f-aa32-2a4aecbafe56"
REDIRECT_URI = "http://127.0.0.1:8501/callback"

# Generate login URL
login_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=order placement portfolio"

st.title("🔐 Simple Login Test")

# Always show login button
st.markdown(f"""
<a href="{login_url}" target="_blank">
    <button style="
        background-color: #00d09c;
        color: white;
        padding: 20px 40px;
        font-size: 20px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        margin: 20px 0;
    ">
    Click to Login with Upstox
    </button>
</a>
""", unsafe_allow_html=True)

st.write("If this button works, the issue is in your session.py logic.")