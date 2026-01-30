"""
Enhanced Upstox Session Manager with persistent token storage,
refresh mechanism, and market state tracking.
Incorporates research insights: funding liquidity monitoring via OI velocity
and market microstructure analysis.
"""

import os
import json
import requests
import streamlit as st
from urllib.parse import urlencode
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from cryptography.fernet import Fernet
import hashlib

# ==============================
# CONFIG (EDIT THESE)
# ==============================

# Upstox API v3 endpoints
UPSTOX_AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
UPSTOX_PROFILE_URL = "https://api.upstox.com/v2/user/profile"

# Get from Streamlit secrets or environment
def get_config():
    """Get configuration from secrets.toml or environment variables."""
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets'):
            return {
                'client_id': st.secrets.get("UPSTOX_CLIENT_ID"),
                'client_secret': st.secrets.get("UPSTOX_CLIENT_SECRET"),
                'redirect_uri': st.secrets.get("UPSTOX_REDIRECT_URI", "http://127.0.0.1:8501/callback")
            }
    except:
        pass
    
    # Fallback to environment variables
    return {
        'client_id': os.getenv("UPSTOX_CLIENT_ID"),
        'client_secret': os.getenv("UPSTOX_CLIENT_SECRET"),
        'redirect_uri': os.getenv("UPSTOX_REDIRECT_URI", "http://127.0.0.1:8501/callback")
    }

config = get_config()
CLIENT_ID = config['client_id']
CLIENT_SECRET = config['client_secret']
REDIRECT_URI = config['redirect_uri']

# ==============================
# SECURE TOKEN STORAGE
# ==============================

class TokenStorage:
    """
    Secure token storage with encryption and persistence.
    Stores tokens locally with automatic refresh capability.
    """
    
    def __init__(self, storage_path: str = "data/tokens"):
        self.storage_dir = Path(storage_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or load encryption key
        self.key_file = self.storage_dir / ".key"
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(self.encryption_key)
        
        self.cipher = Fernet(self.encryption_key)
        self.token_file = self.storage_dir / "upstox_tokens.enc"
        
        # Track funding liquidity state (research concept)
        self.funding_state = {
            "last_oi_velocity": 0.0,
            "liquidity_regime": "NORMAL",  # NORMAL, CONSTRICTED, EXPANSIVE
            "last_update": datetime.utcnow()
        }
    
    def _generate_user_hash(self) -> str:
        """Generate unique user identifier for multi-user support"""
        # In production, use actual user ID. Here we use client ID as proxy
        return hashlib.sha256(f"{CLIENT_ID}_{REDIRECT_URI}".encode()).hexdigest()[:16]
    
    def save_tokens(self, access_token: str, refresh_token: str, expires_in: int):
        """Securely save tokens with expiration"""
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat(),
            "client_id": CLIENT_ID,
            "user_hash": self._generate_user_hash(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Encrypt and save
        encrypted = self.cipher.encrypt(pickle.dumps(token_data))
        with open(self.token_file, 'wb') as f:
            f.write(encrypted)
        
        st.success("✓ Authentication tokens saved securely")
    
    def load_tokens(self) -> Optional[Dict]:
        """Load and decrypt tokens, check expiration"""
        if not self.token_file.exists():
            return None
        
        try:
            with open(self.token_file, 'rb') as f:
                encrypted = f.read()
            
            token_data = pickle.loads(self.cipher.decrypt(encrypted))
            
            # Check expiration
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if datetime.utcnow() > expires_at - timedelta(minutes=5):  # 5 min buffer
                st.warning("⚠️ Access token expired or near expiry")
                return None
            
            # Verify client matches
            if token_data.get("client_id") != CLIENT_ID:
                st.error("Client ID mismatch - reauthentication required")
                return None
            
            return token_data
        except Exception as e:
            st.error(f"Token load error: {e}")
            return None
    
    def update_funding_state(self, oi_velocity: float):
        """
        Update funding liquidity state based on OI velocity.
        Research concept: Monitor capital flow intensity.
        """
        self.funding_state["last_oi_velocity"] = oi_velocity
        
        # Determine liquidity regime (research concept)
        if oi_velocity > 1.5:
            regime = "EXPANSIVE"  # High capital inflow
        elif oi_velocity < -1.5:
            regime = "CONSTRICTED"  # Capital outflow
        else:
            regime = "NORMAL"
        
        if self.funding_state["liquidity_regime"] != regime:
            st.info(f"💰 Liquidity regime changed: {regime} (OI Velocity: {oi_velocity:.2f})")
        
        self.funding_state["liquidity_regime"] = regime
        self.funding_state["last_update"] = datetime.utcnow()
    
    def get_funding_state(self) -> Dict:
        """Get current funding liquidity state"""
        return self.funding_state.copy()

# ==============================
# ENHANCED SESSION MANAGER
# ==============================

class UpstoxSession:
    """
    Enhanced session manager with:
    1. Persistent token storage
    2. Automatic token refresh
    3. Funding liquidity tracking
    4. Market state awareness
    """
    
    SESSION_KEY = "upstox_access_token"
    SESSION_REFRESH_KEY = "upstox_refresh_token"
    SESSION_PROFILE_KEY = "upstox_profile"
    
    _token_storage = TokenStorage()
    _market_state = {
        "current_regime": None,
        "last_velocity": 0.0,
        "gamma_regime": None,  # POSITIVE/NEGATIVE from research
        "wall_levels": {"call": None, "put": None},
        "trap_zones": []
    }
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user has valid authentication"""
        if UpstoxSession.SESSION_KEY in st.session_state:
            return True
        
        # Check persistent storage
        tokens = UpstoxSession._token_storage.load_tokens()
        if tokens:
            # Load into session state
            st.session_state[UpstoxSession.SESSION_KEY] = tokens["access_token"]
            st.session_state[UpstoxSession.SESSION_REFRESH_KEY] = tokens["refresh_token"]
            return True
        
        return False
    
    @staticmethod
    def get_access_token() -> Optional[str]:
        """Get current access token, refresh if needed"""
        if not UpstoxSession.is_authenticated():
            return None
        
        # Check if token is in session state
        if UpstoxSession.SESSION_KEY in st.session_state:
            return st.session_state[UpstoxSession.SESSION_KEY]
        
        return None
    
    @staticmethod
    def logout() -> None:
        """Secure logout - clear all session and storage"""
        for key in [UpstoxSession.SESSION_KEY, 
                   UpstoxSession.SESSION_REFRESH_KEY,
                   UpstoxSession.SESSION_PROFILE_KEY]:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear persistent storage
        storage_file = Path("data/tokens/upstox_tokens.enc")
        if storage_file.exists():
            storage_file.unlink()
        
        # Clear query params
        st.query_params.clear()
        st.success("✓ Logged out successfully")
        st.rerun()
    
    # ==========================
    # TOKEN MANAGEMENT
    # ==========================
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[Tuple[str, str]]:
        """
        Refresh expired access token using refresh token.
        Returns (new_access_token, new_refresh_token) or None
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        
        try:
            response = requests.post(
                UPSTOX_TOKEN_URL,
                headers=headers,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                return token_data.get("access_token"), token_data.get("refresh_token")
            else:
                st.error(f"Token refresh failed: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Token refresh error: {e}")
            return None
    
    @staticmethod
    def exchange_code_for_token(code: str) -> Optional[Dict]:
        """Exchange authorization code for tokens with full token data"""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        data = {
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        
        try:
            response = requests.post(
                UPSTOX_TOKEN_URL,
                headers=headers,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Token exchange failed: {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Token exchange error: {e}")
            return None
    
    # ==========================
    # USER PROFILE
    # ==========================
    
    @staticmethod
    def get_user_profile(access_token: str) -> Optional[Dict]:
        """Fetch user profile for session context"""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        try:
            response = requests.get(UPSTOX_PROFILE_URL, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json().get("data", {})
            return None
        except Exception as e:
            st.warning(f"Profile fetch warning: {e}")
            return None
    
    # ==========================
    # MARKET STATE TRACKING (Research Integration)
    # ==========================
    
    @staticmethod
    def update_market_state(oi_velocity: float, gamma_exposure: float, 
                           wall_levels: Dict, spot_divergence: float):
        """
        Update comprehensive market state based on research concepts.
        
        Args:
            oi_velocity: Rate of change of Open Interest
            gamma_exposure: Net Gamma Exposure (positive/negative)
            wall_levels: Dict with 'call' and 'put' wall strike levels
            spot_divergence: Divergence between spot and derivative metrics
        """
        # Update funding liquidity state
        UpstoxSession._token_storage.update_funding_state(oi_velocity)
        
        # Determine gamma regime (research concept)
        if gamma_exposure > 0:
            gamma_regime = "POSITIVE"  # Stabilizing, pinning expected
        elif gamma_exposure < 0:
            gamma_regime = "NEGATIVE"  # Accelerating, squeezes possible
        else:
            gamma_regime = "NEUTRAL"
        
        # Detect potential traps (research concept)
        trap_zones = []
        current_price = 0  # Would be passed in production
        
        # Check if price is near walls with high OI unwinding
        for side, level in wall_levels.items():
            if level and abs(current_price - level) / level < 0.005:  # Within 0.5%
                # High probability of trap if OI velocity is negative (unwinding)
                if oi_velocity < -1.0:
                    trap_zones.append({
                        "side": side,
                        "level": level,
                        "type": "GAMMA_TRAP" if abs(gamma_exposure) > 0.5 else "OI_TRAP",
                        "confidence": min(abs(oi_velocity) / 3, 1.0)
                    })
        
        # Update market state
        UpstoxSession._market_state.update({
            "last_velocity": oi_velocity,
            "gamma_regime": gamma_regime,
            "wall_levels": wall_levels,
            "trap_zones": trap_zones,
            "spot_divergence": spot_divergence,
            "funding_regime": UpstoxSession._token_storage.get_funding_state()["liquidity_regime"],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    @staticmethod
    def get_market_state() -> Dict:
        """Get current market state analysis"""
        return UpstoxSession._market_state.copy()
    
    # ==========================
    # STREAMLIT ENTRYPOINT (ENHANCED)
    # ==========================
    

    @staticmethod
    def authenticate() -> str:
        """
        Simplified authentication flow that ALWAYS shows login button when not authenticated
        """
        
        # DEBUG: Show what's in query params
        query_params = st.query_params
        st.sidebar.write(f"🔍 Query params: {dict(query_params)}")
        
        # Check for OAuth callback FIRST (before checking anything else)
        if "code" in query_params:
            try:
                code = query_params["code"]
                st.sidebar.write(f"🔍 Got OAuth code: {code[:20]}...")
                
                # Exchange code for tokens
                token_data = UpstoxSession.exchange_code_for_token(code)
                if token_data:
                    st.sidebar.success("✅ Token exchange successful")
                    
                    # Save tokens
                    UpstoxSession._token_storage.save_tokens(
                        token_data["access_token"],
                        token_data.get("refresh_token", ""),
                        token_data.get("expires_in", 86400)
                    )
                    
                    # Store in session
                    st.session_state[UpstoxSession.SESSION_KEY] = token_data["access_token"]
                    if "refresh_token" in token_data:
                        st.session_state[UpstoxSession.SESSION_REFRESH_KEY] = token_data["refresh_token"]
                    
                    # Load profile
                    profile = UpstoxSession.get_user_profile(token_data["access_token"])
                    if profile:
                        st.session_state[UpstoxSession.SESSION_PROFILE_KEY] = profile
                    
                    # Clean URL and rerun
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error("❌ Failed to exchange code for tokens")
                    
            except Exception as e:
                st.error(f"❌ Authentication failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # Check if already authenticated
        if UpstoxSession.is_authenticated():
            token = UpstoxSession.get_access_token()
            return token
        
        # Check persistent storage
        stored_tokens = UpstoxSession._token_storage.load_tokens()
        if stored_tokens:
            # Use stored token
            st.session_state[UpstoxSession.SESSION_KEY] = stored_tokens["access_token"]
            st.session_state[UpstoxSession.SESSION_REFRESH_KEY] = stored_tokens["refresh_token"]
            
            token = UpstoxSession.get_access_token()
            profile = UpstoxSession.get_user_profile(token)
            if profile:
                st.session_state[UpstoxSession.SESSION_PROFILE_KEY] = profile
            
            return token
        
        # ==============================================
        # SHOW LOGIN INTERFACE (only if not authenticated)
        # ==============================================
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; color: white;'>
            <h2 style='color: white;'>🔐 Algorithmic Trading Platform Login</h2>
            <p style='font-size: 16px;'>
                Access real-time derivatives analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Login button
        login_url = UpstoxSession.get_login_url()
        
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="{login_url}">
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
                    transition: all 0.3s ease;
                ">
                📈 Login with Upstox
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Login Instructions:**
        1. Click the "Login with Upstox" button above
        2. You'll be redirected to Upstox authorization page
        3. Log in with your Upstox credentials
        4. Authorize the application
        5. You'll be redirected back to this app
        """)
        
        # CRITICAL: Stop execution here
        st.stop()
        return None
    
    @staticmethod
    def get_login_url() -> str:
        """Generate OAuth login URL"""
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": "order placement portfolio"
        }
        return f"{UPSTOX_AUTH_URL}?{urlencode(params)}"
    
    @staticmethod
    def get_session_info() -> Dict:
        """Get comprehensive session information"""
        return {
            "authenticated": UpstoxSession.is_authenticated(),
            "has_profile": UpstoxSession.SESSION_PROFILE_KEY in st.session_state,
            "profile": st.session_state.get(UpstoxSession.SESSION_PROFILE_KEY, {}),
            "funding_state": UpstoxSession._token_storage.get_funding_state(),
            "market_state": UpstoxSession.get_market_state(),
            "storage_path": str(UpstoxSession._token_storage.storage_dir)
        }

# ==============================
# SESSION UTILITIES
# ==============================

def display_session_status():
    """Display session status in Streamlit sidebar"""
    if UpstoxSession.is_authenticated():
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🧠 Session Status")
            
            # User info
            profile = st.session_state.get(UpstoxSession.SESSION_PROFILE_KEY, {})
            if profile:
                st.markdown(f"**User:** {profile.get('user_name', 'N/A')}")
                st.markdown(f"**Email:** {profile.get('email', 'N/A')}")
            
            # Market state
            market_state = UpstoxSession.get_market_state()
            if market_state.get("funding_regime"):
                regime = market_state["funding_regime"]
                color = {
                    "EXPANSIVE": "🟢",
                    "NORMAL": "🟡", 
                    "CONSTRICTED": "🔴"
                }.get(regime, "⚪")
                st.markdown(f"**Funding:** {color} {regime}")
            
            if market_state.get("gamma_regime"):
                gamma = market_state["gamma_regime"]
                icon = "📌" if gamma == "POSITIVE" else "🚀" if gamma == "NEGATIVE" else "⚖️"
                st.markdown(f"**Gamma:** {icon} {gamma}")
            
            # Logout button
            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                UpstoxSession.logout()

# ==============================
# INITIALIZATION
# ==============================

def initialize_session():
    """
    Initialize session with enhanced capabilities.
    Call this at the start of your app.
    """
    # Ensure token storage directory exists
    Path("data/tokens").mkdir(parents=True, exist_ok=True)
    
    # Initialize session state keys
    session_keys = [
        UpstoxSession.SESSION_KEY,
        UpstoxSession.SESSION_REFRESH_KEY,
        UpstoxSession.SESSION_PROFILE_KEY,
        "market_state",
        "funding_analysis"
    ]
    
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None if "state" in key else {}