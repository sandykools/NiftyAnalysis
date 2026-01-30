"""
ENHANCED TRADING TOOL - Research-Based Live Feature Engine
Incorporates OI Velocity, Gamma Exposure, Walls/Traps, and Divergence analysis.
"""

import os
import sys

# Fix Conda environment issue
if 'conda' in sys.version.lower():
    try:
        # Try to fix conda path issues
        os.environ['PATH'] = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin') + ':' + os.environ['PATH']
    except:
        pass
# ==============================
# STREAMLIT CONFIG FIX - ADD THIS
# ==============================
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
import time 
import threading  # <-- ADD THIS LINE
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datetime import datetime, timezone
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Enhanced core components
from core.session import UpstoxSession, initialize_session, display_session_status
from data.upstox_client import UpstoxClient, MarketAnalytics, display_market_analytics
from data.instrument_master import get_option_keys

try:
    from core.feature_pipeline import build_and_store_features, create_feature_summary
except ImportError as e:
    st.error(f"Feature pipeline import error: {e}")
    # Define dummy functions to prevent crash
    def build_and_store_features(*args, **kwargs):
        return {"error": "Feature pipeline not loaded"}
    def create_feature_summary(*args, **kwargs):
        return {"error": "Feature pipeline not loaded"}
from core.scheduler import MarketScheduler
from core.signals.state_machine import SignalStateMachine, SignalValidator, create_signal_engine
from features.breadth import build_constituents_df
from ml.feature_contract import FEATURE_VERSION

# Enhanced storage
from storage.repository import (
    fetch_latest_features,
    insert_signal,
    signal_exists,
    validate_new_signals,
    expire_old_signals,
    fetch_active_signals,
    fetch_recent_analytics,
    get_database_stats,
    fetch_signal_performance,
    insert_research_analytics
)

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Algorithmic Trading Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .research-insight {
        background: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .trap-alert {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .divergence-alert {
        background: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# APP HEADER
# ==============================

st.markdown('<div class="main-header">📊 Algorithmic Trading Research Platform</div>', unsafe_allow_html=True)
st.markdown("""
<div style='color: #6B7280; margin-bottom: 2rem;'>
    Advanced derivatives analytics with OI Velocity, Gamma Exposure, Structural Walls/Traps, and Spot Divergence detection.
    Version: Research v2.0
</div>
""", unsafe_allow_html=True)

# ==============================
# INITIALIZATION
# ==============================

def initialize_app_state():
    """Initialize application state with research components."""
    
    # Initialize session
    initialize_session()
    
    # Initialize state variables
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Market data series
        for key in ["price_series", "volume_series", "ccc_history", "oi_velocity_history", "gamma_history"]:
            if key not in st.session_state:
                st.session_state[key] = pd.Series(dtype=float)
        
        # Research analytics
        if "research_analytics" not in st.session_state:
            st.session_state.research_analytics = []
        
        # Signal engine
        if "signal_engine" not in st.session_state:
            st.session_state.signal_engine = create_signal_engine(
                signal_expiry_minutes=5,
                confidence_threshold=0.2
            )
        
        # Scheduler
        if "scheduler" not in st.session_state:
            st.session_state.scheduler = MarketScheduler(interval_seconds=30)
            # Initialize scheduler properties
            st.session_state.scheduler._last_run = None
            st.session_state.scheduler._total_executions = 0
        
        # Load Nifty weights
        if "nifty_weights" not in st.session_state:
            try:
                with open("config/nifty_weights.json", "r") as f:
                    st.session_state.nifty_weights = json.load(f)
            except:
                st.session_state.nifty_weights = {}
        
        # Instrument master path
        if "instrument_master" not in st.session_state:
            st.session_state.instrument_master = Path("G:/trading_app/data/instruments.json.gz")
        
        # Configuration - SMART EXPIRY SELECTION
        if "config" not in st.session_state:
            from utils.expiry_utils import get_trading_expiry
            
            # Get smart expiry
            underlying = "NIFTY"
            expiry_date_str = get_trading_expiry(underlying)
            
            if not expiry_date_str:
                # Fallback to next available date from instrument master
                from data.instrument_master import get_next_available_expiry
                expiry_date_str = get_next_available_expiry(underlying)
                
                if not expiry_date_str:
                    # Final fallback to a known date
                    expiry_date_str = "2026-02-03"
            
            # Convert string to datetime object
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
            
            st.session_state.config = {
                "index_symbol": "NSE_INDEX|Nifty 50",
                "underlying": underlying,
                "expiry_date": expiry_date,
                "max_option_keys": 200
            }
            
            # Show expiry info
            from utils.expiry_utils import is_market_open
            market_status = "open" if is_market_open() else "closed"
            
            # Get today's date
            today_str = datetime.now().strftime('%Y-%m-%d')
            
            # Show appropriate message
            if expiry_date_str == today_str:
                st.info(f"📅 Using TODAY's expiry: {expiry_date_str} (Market: {market_status})")
            else:
                st.info(f"📅 Using NEXT expiry: {expiry_date_str} (Today: {today_str}, Market: {market_status})")
        
        # Option keys cache
        if "option_keys" not in st.session_state:
            st.session_state.option_keys = []
        
        # Market regime tracking
        if "market_regime" not in st.session_state:
            st.session_state.market_regime = "UNKNOWN"
        
        # Trap detection history
        if "trap_detections" not in st.session_state:
            st.session_state.trap_detections = []
        
        st.success("✅ Application state initialized with research components")

# Run initialization
initialize_app_state()

# ==============================
# AUTHENTICATION
# ==============================

st.markdown('<div class="sub-header">🔐 Authentication</div>', unsafe_allow_html=True)

# Authenticate - this will show login button if not authenticated
access_token = UpstoxSession.authenticate()

# CRITICAL: Check if authenticate() actually returned a token
# If it returns None, we need to stop execution here
if not access_token:
    # authenticate() should have shown login button and stopped with st.stop()
    # But if we reach here, something went wrong
    st.error("❌ Authentication required. Please log in to continue.")
    
    # Show login button manually as fallback
    login_url = UpstoxSession.get_login_url()
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
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
                width: 60%;
                margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
            📈 Click to Login with Upstox
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # STOP the app here
    st.stop()

# If we reach here, we have a token
# Initialize client
try:
    client = UpstoxClient(access_token)
    st.success("✅ Upstox client initialized")
        
except Exception as e:
    st.error(f"❌ Error initializing Upstox client: {e}")
    st.stop()

# ==============================
# SIDEBAR - SESSION STATUS
# ==============================

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    
    # Display session status
    display_session_status()
    
    # Database stats
    st.markdown("---")
    st.markdown("### 📊 Database")
    
    db_stats = get_database_stats()
    if db_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Features", f"{db_stats.get('market_features_count', 0):,}")
        with col2:
            st.metric("Signals", f"{db_stats.get('signals_count', 0):,}")
    
    # Market hours info
    st.markdown("---")
    st.markdown("### 🕒 Market Hours")
    
    scheduler: MarketScheduler = st.session_state.scheduler
    now = datetime.now()
    
    if scheduler._is_market_open(now):
        st.success("✅ Market Open")
        next_run = scheduler._last_run + timedelta(seconds=scheduler.interval_seconds) if scheduler._last_run else now
        time_to_next = (next_run - now).total_seconds()
        if time_to_next > 0:
            st.info(f"Next cycle in: {int(time_to_next)}s")
    else:
        st.warning("⏸️ Market Closed")
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Clear Cache", width='stretch'):
        for key in list(st.session_state.keys()):
            if key not in ["initialized", "instrument_master", "config", "nifty_weights"]:
                del st.session_state[key]
        st.rerun()
    
    if st.button("📊 View Performance", width='stretch'):
        st.session_state.show_performance = True
    
    if st.button("🛠️ System Logs", width='stretch'):
        st.session_state.show_logs = True

def retry_on_failure(max_retries=3, delay=5):
    """Decorator for retrying failed executions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        print(f"⏸️ Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"❌ All {max_retries} attempts failed")
                        raise
            return None
        return wrapper
    return decorator

def log_execution_time(func):
    """Decorator for logging execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"⏱️ Starting {func.__name__}...")
        
        result = func(*args, **kwargs)
        
        execution_time = time.time() - start_time
        print(f"⏱️ {func.__name__} completed in {execution_time:.2f}s")
        
        return result
    return wrapper
# ==============================
# RESEARCH EXECUTION CYCLE
# ==============================

@log_execution_time
@retry_on_failure(max_retries=2, delay=3)
def execute_research_cycle(config=None, nifty_weights=None, option_keys=None):
    """
    Enhanced execution cycle with research analytics.
    Returns comprehensive research data.
    """
    # Use provided parameters or fall back to session state
    if config is None:
        try:
            config = st.session_state.config
        except (KeyError, AttributeError):
            print("⚠️ Config not found in session state")
            return None
    
    if nifty_weights is None:
        nifty_weights = st.session_state.get("nifty_weights", {})
    
    if option_keys is None:
        option_keys = st.session_state.get("option_keys", [])
    
    research_data = {}
    
    try:
        # ------------------------------
        # 1. INDEX QUOTE
        # ------------------------------
        print(f"📊 Fetching index quote for {config.get('index_symbol', 'NSE_INDEX|Nifty 50')}")
        index_data = client.fetch_index_quote(config["index_symbol"])
        
        if not index_data or index_data.get("ltp") is None:
            print("❌ No index data received")
            return None
        
        ltp = float(index_data["ltp"])
        open_price = float(index_data["open"])
        volume = 0.0  # index has no volume
        
        research_data["index"] = {
            "ltp": ltp,
            "open": open_price,
            "change": index_data.get("percent_change", 0),
            "timestamp": index_data.get("timestamp", datetime.now(timezone.utc).isoformat())
        }
        
        print(f"✅ Index data: {ltp} (Change: {index_data.get('percent_change', 0)}%)")
        
        # ------------------------------
        # 2. UPDATE PRICE SERIES
        # ------------------------------
        # Create new series safely
        try:
            if "price_series" not in st.session_state:
                st.session_state.price_series = pd.Series(dtype=float)
            
            if "volume_series" not in st.session_state:
                st.session_state.volume_series = pd.Series(dtype=float)
            
            # Update price series
            new_price_series = pd.concat([
                st.session_state.price_series, 
                pd.Series([ltp], index=[datetime.now()])
            ]).tail(200)
            
            # Update volume series
            new_volume_series = pd.concat([
                st.session_state.volume_series, 
                pd.Series([volume], index=[datetime.now()])
            ]).tail(200)
            
            # Store back to session state
            st.session_state.price_series = new_price_series
            st.session_state.volume_series = new_volume_series
            
            print(f"📈 Price series updated: {len(st.session_state.price_series)} data points")
            
        except Exception as e:
            print(f"⚠️ Error updating price series: {e}")
            # Initialize fresh series
            st.session_state.price_series = pd.Series([ltp], index=[datetime.now()])
            st.session_state.volume_series = pd.Series([volume], index=[datetime.now()])
        
        # ------------------------------
        # 3. BREADTH ANALYSIS (CCC)
        # ------------------------------
        print("📊 Starting breadth analysis...")
        try:
            if not nifty_weights:
                print("⚠️ No Nifty weights found, skipping breadth analysis")
                ccc_value = 0.0
                constituents_df = pd.DataFrame()
            else:
                equity_quotes_df = client.fetch_equity_quotes(
                    symbols=list(nifty_weights.keys())
                )
                
                constituents_df = build_constituents_df(
                    equity_quotes_df=equity_quotes_df,
                    weights=nifty_weights
                )
                
                if not constituents_df.empty:
                    ccc_value = (
                        constituents_df["weight"] * constituents_df["price_change"]
                    ).sum()
                    print(f"✅ CCC Value: {ccc_value:.4f}")
                else:
                    ccc_value = 0.0
                    print("⚠️ Empty constituents dataframe")
        except Exception as e:
            print(f"⚠️ Breadth analysis error: {e}")
            ccc_value = 0.0
            constituents_df = pd.DataFrame()
        
        # Update CCC history
        try:
            if "ccc_history" not in st.session_state:
                st.session_state.ccc_history = pd.Series(dtype=float)
            
            st.session_state.ccc_history = pd.concat([
                st.session_state.ccc_history, 
                pd.Series([ccc_value], index=[datetime.now()])
            ]).tail(200)
        except:
            st.session_state.ccc_history = pd.Series([ccc_value], index=[datetime.now()])
        
        research_data["breadth"] = {
            "ccc_value": ccc_value,
            "constituents_count": len(constituents_df),
            "positive_constituents": len(constituents_df[constituents_df["price_change"] > 0]) if not constituents_df.empty else 0
        }
        
        # ------------------------------
        # 4. OPTION INSTRUMENT DISCOVERY
        # ------------------------------
        print("🔍 Fetching option instruments...")
        try:
            if not option_keys:
                expiry_str = config["expiry_date"].strftime("%Y-%m-%d")
                option_keys = get_option_keys(
                    underlying=config["underlying"],
                    expiry=expiry_str,
                    max_keys=config.get("max_option_keys", 200)
                )
                st.session_state.option_keys = option_keys
                print(f"✅ Found {len(option_keys)} option keys")
        except Exception as e:
            print(f"⚠️ Option discovery error: {e}")
            option_keys = []
            st.session_state.option_keys = []
        
        if not option_keys:
            print("⚠️ No option keys found")
            return research_data  # Return what we have so far
        
        # ------------------------------
        # 5. ENHANCED OPTION CHAIN ANALYSIS
        # ------------------------------
        print("📊 Analyzing option chain...")
        option_analytics = client.fetch_option_chain_with_analytics(
            option_keys,
            ltp
        )
        
        if not option_analytics or "raw_data" not in option_analytics:
            print("❌ Failed to fetch option analytics")
            return research_data  # Return what we have so far
        
        option_chain_df = option_analytics["raw_data"]
        analytics = option_analytics.get("analytics", {})
        insights = option_analytics.get("market_insights", [])
        
        research_data["options"] = {
            "analytics": analytics,
            "insights": insights,
            "chain_size": len(option_chain_df),
            "put_call_ratio": analytics.get("put_call_ratio", 1.0) if isinstance(analytics.get("put_call_ratio"), (int, float)) else 1.0
        }
        
        print(f"✅ Option chain analyzed: {len(option_chain_df)} strikes, PCR: {analytics.get('put_call_ratio', 'N/A')}")
        
        # Store analytics for visualization
        if analytics:
            try:
                if "research_analytics" not in st.session_state:
                    st.session_state.research_analytics = []
                
                st.session_state.research_analytics.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **analytics
                })
                
                # Keep only recent analytics
                if len(st.session_state.research_analytics) > 100:
                    st.session_state.research_analytics = st.session_state.research_analytics[-100:]
                
                # Update market regime
                st.session_state.market_regime = analytics.get("market_regime", "UNKNOWN")
                
                # Store in database
                insert_research_analytics(analytics)
                
            except Exception as e:
                print(f"⚠️ Error storing analytics: {e}")
        
        # ------------------------------
        # 6. FEATURE PIPELINE
        # ------------------------------
        print("⚙️ Running feature pipeline...")
        try:
            snapshot_ts = pd.Timestamp.utcnow().floor("ms")
            
            feature_result = build_and_store_features(
                timestamp=snapshot_ts,
                option_chain_df=option_chain_df,
                spot_price=ltp,
                expiry_datetime=config["expiry_date"],
                
                ltp=ltp,
                vwap=open_price,
                price_series=st.session_state.price_series,
                volume_series=st.session_state.volume_series,
                
                constituents_df=constituents_df,
                ccc_history=st.session_state.ccc_history,
                
                # Enhanced parameters
                client=client,
                option_keys=option_keys
            )
            
            if feature_result:
                research_data["features"] = feature_result.get("features", {})
                research_data["feature_summary"] = create_feature_summary(feature_result["features"])
                print("✅ Feature pipeline completed")
            else:
                print("⚠️ Feature pipeline returned empty result")
                
        except Exception as e:
            print(f"⚠️ Feature pipeline error: {e}")
            import traceback
            traceback.print_exc()
        
        # ------------------------------
        # 7. SIGNAL GENERATION
        # ------------------------------
        print("🎯 Checking for signals...")
        try:
            feature_row = fetch_latest_features(FEATURE_VERSION)
            if feature_row is not None and not feature_row.empty:
                # Check if signal already exists
                timestamp_str = feature_row.iloc[0]["timestamp"]
                
                if not signal_exists(timestamp_str):
                    # Generate signal
                    signal_engine: SignalStateMachine = st.session_state.signal_engine
                    signal = signal_engine.build_signal(feature_row.iloc[0])
                    
                    # Validate signal
                    is_valid, reason = SignalValidator.validate_signal(signal, feature_row.iloc[0])
                    
                    if is_valid and signal["signal_type"] != "NEUTRAL":
                        # Add research context
                        signal["research_validation"] = {
                            "is_valid": is_valid,
                            "reason": reason,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Insert signal
                        insert_signal(signal)
                        
                        research_data["signal"] = {
                            "generated": True,
                            "type": signal["signal_type"],
                            "confidence": signal["confidence"],
                            "strength": signal.get("signal_strength", "UNKNOWN")
                        }
                        print(f"✅ Signal generated: {signal['signal_type']} ({signal['confidence']:.0%})")
                    else:
                        research_data["signal"] = {
                            "generated": False,
                            "reason": reason if not is_valid else "NEUTRAL signal"
                        }
                        print(f"ℹ️ No valid signal: {reason if not is_valid else 'NEUTRAL'}")
            else:
                print("ℹ️ No features available for signal generation")
        except Exception as e:
            print(f"⚠️ Signal generation error: {e}")
        
        # ------------------------------
        # 8. SIGNAL LIFECYCLE MANAGEMENT
        # ------------------------------
        try:
            validate_new_signals(confidence_threshold=0.2)
            expire_old_signals()
        except Exception as e:
            print(f"⚠️ Signal lifecycle error: {e}")
        
        # ------------------------------
        # 9. TRAP DETECTION
        # ------------------------------
        if analytics and "potential_traps" in analytics:
            traps = analytics["potential_traps"]
            if traps:
                try:
                    if "trap_detections" not in st.session_state:
                        st.session_state.trap_detections = []
                    
                    for trap in traps:
                        if trap.get("confidence", 0) > 0.6:
                            st.session_state.trap_detections.append({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "trap": trap,
                                "market_regime": analytics.get("market_regime")
                            })
                    
                    # Keep only recent traps
                    if len(st.session_state.trap_detections) > 20:
                        st.session_state.trap_detections = st.session_state.trap_detections[-20:]
                    
                    research_data["traps"] = traps
                    
                    if traps:
                        print(f"🎯 Detected {len(traps)} potential traps")
                        
                except Exception as e:
                    print(f"⚠️ Trap detection error: {e}")
        
        # ------------------------------
        # 10. SUCCESSFUL COMPLETION
        # ------------------------------
        print("✅ Research cycle completed successfully")
        return research_data
        
    except Exception as e:
        print(f"❌ Critical error in research cycle: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================
# ENHANCED AUTO-REFRESH SYSTEM
# ==============================

class AutoRefreshManager:
    """Robust auto-refresh manager with error handling and state persistence"""
    
    def __init__(self, scheduler, config, nifty_weights, option_keys):
        self.scheduler = scheduler
        self.config = config
        self.nifty_weights = nifty_weights
        self.option_keys = option_keys
        self.last_execution_time = None
        self.execution_count = 0
        self.errors = []
        self.max_retries = 3
        self.retry_delay = 5
        
    def should_execute(self):
        """Determine if execution should run based on time and market conditions"""
        now = datetime.now()
        
        # Check market hours
        if not self.scheduler._is_market_open(now):
            return False, "Market closed"
        
        # Check if first run
        if self.last_execution_time is None:
            return True, "First execution"
        
        # Check time since last execution
        time_since_last = (now - self.last_execution_time).total_seconds()
        
        # Debug log to see what's happening
        print(f"🕒 Auto-refresh timing: {int(time_since_last)}s since last, interval: {self.scheduler.interval_seconds}s")
        
        if time_since_last >= self.scheduler.interval_seconds:
            return True, f"Interval reached ({int(time_since_last)}s > {self.scheduler.interval_seconds}s)"
        
        return False, f"Not due yet ({int(self.scheduler.interval_seconds - time_since_last)}s remaining)"
    
    def execute_with_retry(self):
        """Execute research cycle with retry logic"""
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{self.max_retries} to execute research cycle")
                
                result = execute_research_cycle(
                    config=self.config,
                    nifty_weights=self.nifty_weights,
                    option_keys=self.option_keys
                )
                
                if result:
                    self.last_execution_time = datetime.now()
                    self.execution_count += 1
                    return True, result, f"Execution successful (attempt {attempt + 1})"
                else:
                    return False, None, "Research cycle returned no data"
                    
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                self.errors.append({
                    "timestamp": datetime.now(),
                    "attempt": attempt + 1,
                    "error": str(e)
                })
                
                if attempt < self.max_retries - 1:
                    print(f"⏸️ Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    return False, None, f"All retries failed. Last error: {str(e)}"
        
        return False, None, "Max retries exceeded"
    
    def get_status(self):
        """Get current status of auto-refresh manager"""
        status = {
            "last_execution": self.last_execution_time.strftime("%H:%M:%S") if self.last_execution_time else "Never",
            "execution_count": self.execution_count,
            "is_market_open": self.scheduler._is_market_open(datetime.now()),
            "interval_seconds": self.scheduler.interval_seconds,
            "error_count": len(self.errors)
        }
        
        if self.last_execution_time:
            time_since = (datetime.now() - self.last_execution_time).total_seconds()
            status["time_since_last"] = f"{int(time_since)}s"
            status["time_until_next"] = f"{max(0, int(self.scheduler.interval_seconds - time_since))}s"
        
        return status

# Initialize auto-refresh manager
if "auto_refresh_manager" not in st.session_state:
    st.session_state.auto_refresh_manager = AutoRefreshManager(
        scheduler=st.session_state.scheduler,
        config=st.session_state.config,
        nifty_weights=st.session_state.nifty_weights,
        option_keys=st.session_state.option_keys
    )

# Execute auto-refresh
def execute_auto_refresh():
    """Main auto-refresh execution function"""
    manager = st.session_state.auto_refresh_manager
    
    try:
        # Check if we should execute
        should_execute, reason = manager.should_execute()
        
        if should_execute:
            print(f"🏃‍♂️ Auto-execution triggered: {reason}")
            
            # Execute with retry logic
            success, result, message = manager.execute_with_retry()
            
            if success:
                print(f"✅ {message}")
                
                # Update UI state if needed
                if result and "options" in result and "analytics" in result["options"]:
                    analytics = result["options"]["analytics"]
                    if analytics:
                        # Update market regime
                        st.session_state.market_regime = analytics.get("market_regime", "UNKNOWN")
                        
                        # Store analytics
                        if "research_analytics" not in st.session_state:
                            st.session_state.research_analytics = []
                        st.session_state.research_analytics.append({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            **analytics
                        })
                        # Keep only recent analytics
                        if len(st.session_state.research_analytics) > 100:
                            st.session_state.research_analytics = st.session_state.research_analytics[-100:]
                
                return True, message
            else:
                print(f"❌ Auto-execution failed: {message}")
                return False, message
        else:
            # Log status periodically
            if manager.last_execution_time:
                time_since = (datetime.now() - manager.last_execution_time).total_seconds()
                if int(time_since) % 10 == 0:  # Log every 10 seconds
                    print(f"⏱️ Auto-refresh status: {reason}")
            return None, reason
            
    except Exception as e:
        error_msg = f"Auto-refresh system error: {str(e)}"
        print(f"🔥 {error_msg}")
        manager.errors.append({
            "timestamp": datetime.now(),
            "error": error_msg
        })
        return False, error_msg

# Run auto-refresh
execute_auto_refresh()

# ==============================
# ENHANCED BACKGROUND SCHEDULER
# ==============================

class BackgroundScheduler:
    """Robust background scheduler with error handling and monitoring"""
    
    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.errors = []
        self.last_check = None
        self.health_checks = []
        
    def health_check(self):
        """Perform system health check"""
        checks = []
        
        # Check scheduler
        try:
            scheduler = st.session_state.scheduler
            checks.append({
                "component": "Scheduler",
                "status": "HEALTHY" if scheduler else "UNKNOWN",
                "details": f"Interval: {scheduler.interval_seconds}s" if scheduler else "Not initialized"
            })
        except Exception as e:
            checks.append({
                "component": "Scheduler",
                "status": "ERROR",
                "details": str(e)
            })
        
        # Check auto-refresh manager
        try:
            manager = st.session_state.auto_refresh_manager
            checks.append({
                "component": "AutoRefresh",
                "status": "HEALTHY",
                "details": f"Executions: {manager.execution_count}"
            })
        except Exception as e:
            checks.append({
                "component": "AutoRefresh",
                "status": "ERROR",
                "details": str(e)
            })
        
        self.health_checks = checks
        return checks
    
    def start(self):
        """Start background scheduler thread"""
        if self.running:
            print("⚠️ Background scheduler already running")
            return
        
        def background_task():
            print("🚀 Starting enhanced background scheduler...")
            self.running = True
            
            while self.running:
                try:
                    self.last_check = datetime.now()
                    
                    # Perform health check every minute
                    if not self.health_checks or (datetime.now() - self.last_check).seconds > 60:
                        self.health_check()
                    
                    # Log status every 30 seconds
                    current_time = datetime.now()
                    if current_time.second % 30 == 0:
                        manager = st.session_state.auto_refresh_manager
                        if manager.last_execution_time:
                            time_since = (current_time - manager.last_execution_time).total_seconds()
                            print(f"📊 Background monitor: Last execution {int(time_since)}s ago, {manager.execution_count} total runs")
                    
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    error_msg = f"Background scheduler error: {str(e)}"
                    print(f"🔥 {error_msg}")
                    self.errors.append({
                        "timestamp": datetime.now(),
                        "error": error_msg
                    })
                    time.sleep(30)  # Wait longer on error
        
        self.thread = threading.Thread(target=background_task, daemon=True)
        self.thread.start()
        print("✅ Enhanced background scheduler started")
    
    def stop(self):
        """Stop background scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🛑 Background scheduler stopped")
    
    def get_status(self):
        """Get scheduler status"""
        return {
            "running": self.running,
            "check_interval": self.check_interval,
            "last_check": self.last_check.strftime("%H:%M:%S") if self.last_check else "Never",
            "error_count": len(self.errors),
            "health_checks": self.health_checks
        }

# Initialize and start background scheduler
if "background_scheduler" not in st.session_state:
    st.session_state.background_scheduler = BackgroundScheduler(check_interval=5)
    st.session_state.background_scheduler.start()


def get_scheduler_status():
    """Get comprehensive scheduler status"""
    scheduler = st.session_state.scheduler
    manager = st.session_state.auto_refresh_manager
    
    status = {
        "scheduler_active": True,
        "interval_seconds": scheduler.interval_seconds,
        "total_executions": scheduler._total_executions,
        "last_run": scheduler._last_run.strftime("%H:%M:%S") if scheduler._last_run else "Never",
        "is_market_open": scheduler._is_market_open(datetime.now()),
        "auto_refresh_status": manager.get_status()
    }
    
    # Calculate next run time
    if scheduler._last_run:
        next_run = scheduler._last_run + timedelta(seconds=scheduler.interval_seconds)
        time_until_next = (next_run - datetime.now()).total_seconds()
        status["next_run"] = next_run.strftime("%H:%M:%S") if time_until_next > 0 else "Now"
        status["time_until_next"] = f"{max(0, int(time_until_next))}s"
    else:
        status["next_run"] = "Pending"
        status["time_until_next"] = "N/A"
    
    return status

# ==============================
# CREATE TABS
# ==============================

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Analytics", 
    "🎯 Signals", 
    "🔍 Research", 
    "⚙️ Configuration"
])

# ==============================
# AUTO-REFRESH WORKING SOLUTION
# ==============================

# Create a placeholder for auto-refresh
refresh_placeholder = st.empty()

# Check if we need to auto-refresh
if "next_refresh_time" not in st.session_state:
    st.session_state.next_refresh_time = time.time() + 30  # First refresh in 30 seconds

current_time = time.time()

if current_time >= st.session_state.next_refresh_time:
    # Time to refresh!
    print(f"🔄 AUTO-REFRESH EXECUTING at {datetime.now().strftime('%H:%M:%S')}")
    
    # Execute research cycle
    with st.spinner("Auto-refreshing data..."):
        research_data = execute_research_cycle()
    
    if research_data:
        print(f"✅ Auto-refresh completed successfully")
        
        # Update next refresh time
        st.session_state.next_refresh_time = current_time + 30
        
        # Update state
        st.session_state.scheduler._last_run = datetime.now()
        st.session_state.scheduler._total_executions += 1
        st.session_state.auto_refresh_manager.last_execution_time = datetime.now()
        st.session_state.auto_refresh_manager.execution_count += 1
        
        # Show success message
        refresh_placeholder.success(f"✅ Auto-refresh completed at {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(1)
        refresh_placeholder.empty()
        
        # Force a rerun to update UI
        st.rerun()
    else:
        print("❌ Auto-refresh failed")
        refresh_placeholder.error("Auto-refresh failed")
        time.sleep(2)
        refresh_placeholder.empty()
else:
    # Show countdown
    time_until_next = st.session_state.next_refresh_time - current_time
    if time_until_next > 0:
        print(f"⏱️ Next auto-refresh in {int(time_until_next)}s")
        
        # Update countdown every 10 seconds
        if int(time_until_next) % 10 == 0:
            refresh_placeholder.info(f"Next auto-refresh in {int(time_until_next)} seconds...")


# ==============================
# TAB 1: LIVE ANALYTICS
# ==============================

with tab1:
    # Add JavaScript auto-refresh
    auto_refresh_js = """
    <script>
    // Auto-refresh every 30 seconds
    setTimeout(function() {
        window.location.reload();
    }, 30000);
    </script>
    """
    st.components.v1.html(auto_refresh_js, height=0)
    
    st.markdown('<div class="sub-header">📊 Live Market Analytics</div>', unsafe_allow_html=True)
   
    
    # Execute cycle - but this is just for UI display now
    scheduler: MarketScheduler = st.session_state.scheduler
    
    # Track last execution time
    if "last_execution_time" not in st.session_state:
        st.session_state.last_execution_time = None
    
    # Display background scheduler status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if scheduler._last_run:
            time_since_last = (datetime.now() - scheduler._last_run).total_seconds()
            status_text = "✅ Running" if time_since_last < 60 else "⚠️ Stale"
            st.metric("Scheduler", status_text, delta=f"{int(time_since_last)}s ago")
        else:
            st.metric("Scheduler", "⏳ Starting", delta="Initializing")
    
    with col2:
        if st.session_state.price_series.size > 0:
            current_price = st.session_state.price_series.iloc[-1]
            prev_price = st.session_state.price_series.iloc[-2] if st.session_state.price_series.size > 1 else current_price
            change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
            st.metric("Nifty 50", f"{current_price:,.0f}", f"{change_pct:+.2f}%")
    
    with col3:
        if st.session_state.research_analytics:
            latest = st.session_state.research_analytics[-1]
            regime = latest.get("market_regime", "UNKNOWN")
            st.metric("Market Regime", regime)
    
    with col4:
        if st.session_state.research_analytics:
            latest = st.session_state.research_analytics[-1]
            oi_vel = latest.get("oi_velocity", 0)
            st.metric("OI Velocity", f"{oi_vel:+.2f}σ")
    
    # Manual execution button - still works
    if st.button("🔄 Execute Research Cycle Now", type="primary", width='stretch'):
        with st.spinner("Executing research cycle..."):
            research_data = execute_research_cycle()
            if research_data:
                st.success("✅ Research cycle executed successfully")
                
                # Display insights if available
                if "options" in research_data and "insights" in research_data["options"]:
                    for insight in research_data["options"]["insights"][:3]:
                        st.info(insight)
            else:
                st.error("❌ Research cycle failed")
    
    st.markdown("---")
    
    # Display enhanced status
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = get_scheduler_status()
        if status["scheduler_active"]:
            if status["last_run"] != "Never":
                time_since = (datetime.now() - st.session_state.scheduler._last_run).total_seconds() if st.session_state.scheduler._last_run else 999
                if time_since < 60:
                    st.metric("Scheduler", "✅ Active", delta=f"{int(time_since)}s ago")
                else:
                    st.metric("Scheduler", "⚠️ Stale", delta=f"{int(time_since)}s ago")
            else:
                st.metric("Scheduler", "⏳ Initializing", delta="First run pending")
        else:
            st.metric("Scheduler", "❌ Inactive", delta="Check configuration")

    with col2:
        manager_status = st.session_state.auto_refresh_manager.get_status()
        st.metric("Auto Refresh", f"Run #{manager_status['execution_count']}", 
                delta=manager_status['last_execution'])

    with col3:
        if manager_status.get('time_until_next'):
            st.metric("Next Run In", manager_status['time_until_next'])

    with col4:
        error_count = manager_status.get('error_count', 0)
        if error_count == 0:
            st.metric("Errors", "✅ 0", delta="Clean")
        else:
            st.metric("Errors", f"⚠️ {error_count}", delta="Check logs")

    # Control buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Force Run Now", type="primary", width='stretch'):
            # Force immediate execution
            st.session_state.auto_refresh_manager.last_execution_time = None
            st.rerun()

    with col2:
        if st.button("⏸️ Pause Auto-Refresh", width='stretch'):
            # Temporarily disable auto-refresh
            original_interval = st.session_state.scheduler.interval_seconds
            st.session_state.scheduler.interval_seconds = 3600  # 1 hour
            st.success(f"Auto-refresh paused. Original interval: {original_interval}s")
            st.rerun()

    with col3:
        if st.button("▶️ Resume Auto-Refresh", width='stretch'):
            # Restore auto-refresh
            st.session_state.scheduler.interval_seconds = 30
            st.success("Auto-refresh resumed (30s interval)")
            st.rerun()
    # Display background thread info
    with st.expander("🔄 Background Scheduler Info", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if "scheduler_thread" in st.session_state:
                thread = st.session_state.scheduler_thread
                st.success("✅ Background thread active")
                st.info(f"Thread ID: {thread.ident}")
                st.info(f"Thread alive: {thread.is_alive()}")
            else:
                st.error("❌ Background thread not running")
        
        with col2:
            if scheduler._last_run:
                st.metric("Last Execution", scheduler._last_run.strftime("%H:%M:%S"))
                time_since = (datetime.now() - scheduler._last_run).total_seconds()
                st.metric("Time Since", f"{int(time_since)}s")
            else:
                st.metric("Last Execution", "Never")
            
            st.metric("Total Executions", scheduler._total_executions)
    
    
    # Display latest analytics if available
    if st.session_state.research_analytics:
        latest_analytics = st.session_state.research_analytics[-1]
        
        # Create columns for analytics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 OI & Gamma Analysis")
            
            # OI Velocity gauge
            oi_velocity = latest_analytics.get("oi_velocity", 0)
            fig_oi = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=oi_velocity,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "OI Velocity (σ)"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-3, 3]},
                    'steps': [
                        {'range': [-3, -1.5], 'color': "lightgray"},
                        {'range': [-1.5, 1.5], 'color': "gray"},
                        {'range': [1.5, 3], 'color': "lightgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': oi_velocity
                    }
                }
            ))
            fig_oi.update_layout(height=250)
            st.plotly_chart(fig_oi, width='stretch')
        
        with col2:
            st.markdown("#### 🧱 Structure Analysis")
            
            # Structural metrics
            wall_strength = latest_analytics.get("wall_strength", 0)
            trap_prob = latest_analytics.get("trap_probability", 0)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Wall Strength", f"{wall_strength:.2f}")
            with col_b:
                st.metric("Trap Probability", f"{trap_prob:.2%}")
            
            # Display traps if any
            traps = latest_analytics.get("potential_traps", [])
            if traps:
                st.markdown("##### 🎯 Detected Traps")
                for trap in traps[:2]:  # Show max 2
                    direction = trap.get("direction", "").upper()
                    confidence = trap.get("confidence", 0)
                    strike = trap.get("strike", 0)
                    
                    if confidence > 0.6:
                        st.markdown(f"""
                        <div class="trap-alert">
                            <strong>{direction} Trap</strong> at {strike:,.0f}<br>
                            Confidence: {confidence:.0%}<br>
                            Unwinding: {trap.get('unwinding_rate', 0):+.1f}%
                        </div>
                        """, unsafe_allow_html=True)
        
        # Price chart
        st.markdown("---")
        st.markdown("#### 📉 Price & Indicators")
        
        if len(st.session_state.price_series) > 1:
            # Create subplot for price and indicators
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Nifty 50 Price", "OI Velocity", "Gamma Exposure"),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price
            price_data = st.session_state.price_series.reset_index(drop=True)
            fig.add_trace(
                go.Scatter(
                    y=price_data.values,
                    mode='lines',
                    name='Price',
                    line=dict(color='#3B82F6', width=2)
                ),
                row=1, col=1
            )
            
            # OI Velocity (if available in analytics history)
            if len(st.session_state.research_analytics) > 1:
                oi_velocities = [a.get("oi_velocity", 0) for a in st.session_state.research_analytics[-50:]]
                fig.add_trace(
                    go.Scatter(
                        y=oi_velocities,
                        mode='lines',
                        name='OI Velocity',
                        line=dict(color='#10B981', width=1)
                    ),
                    row=2, col=1
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Gamma (if available)
            if len(st.session_state.research_analytics) > 1:
                gamma_values = [a.get("gamma_exposure", {}).get("net_gamma", 0) for a in st.session_state.research_analytics[-50:]]
                fig.add_trace(
                    go.Scatter(
                        y=gamma_values,
                        mode='lines',
                        name='Net Gamma',
                        line=dict(color='#8B5CF6', width=1)
                    ),
                    row=3, col=1
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, width='stretch')

# ==============================
# TAB 2: SIGNALS
# ==============================

with tab2:
    st.markdown('<div class="sub-header">🎯 Trading Signals</div>', unsafe_allow_html=True)
    
    # Fetch active signals
    active_signals = fetch_active_signals(limit=10)
    
    if not active_signals.empty:
        # Display signals in columns
        for _, signal in active_signals.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                signal_type = signal["signal_type"]
                color = "green" if signal_type == "BUY" else "red" if signal_type == "SELL" else "gray"
                st.markdown(f"### <span style='color:{color};'>{signal_type}</span>", unsafe_allow_html=True)
                
                # Rationale
                if pd.notna(signal["rationale"]):
                    st.caption(signal["rationale"])
            
            with col2:
                confidence = signal["confidence"]
                st.metric("Confidence", f"{confidence:.0%}")
            
            with col3:
                strength = signal.get("signal_strength", "UNKNOWN")
                st.metric("Strength", strength)
            
            with col4:
                status = signal["status"]
                badge_color = "blue" if status == "NEW" else "green" if status == "VALIDATED" else "orange"
                st.markdown(f"<div style='padding: 5px 10px; background-color:{badge_color}; color:white; border-radius:5px; text-align:center;'>{status}</div>", 
                           unsafe_allow_html=True)
            
            # Research context (expandable)
            with st.expander("Research Details"):
                if pd.notna(signal.get("research_context")):
                    try:
                        context = json.loads(signal["research_context"]) if isinstance(signal["research_context"], str) else signal["research_context"]
                        st.json(context)
                    except:
                        st.write("No research context available")
            
            st.markdown("---")
        
        # Performance metrics
        st.markdown("#### 📊 Signal Performance")
        performance = fetch_signal_performance(days=7)
        
        if performance:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Signals", performance.get("total_signals", 0))
            
            with col2:
                st.metric("Profitable", performance.get("profitable_signals", 0))
            
            with col3:
                win_rate = performance.get("win_rate", 0)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col4:
                period = performance.get("period_days", 7)
                st.metric("Period", f"{period} days")
        
        # Signal distribution chart
        if not active_signals.empty:
            st.markdown("#### 📈 Signal Distribution")
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=active_signals["signal_type"].value_counts().index,
                    values=active_signals["signal_type"].value_counts().values,
                    hole=.3
                )
            ])
            
            fig.update_layout(
                height=300,
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig, width='stretch')
    
    else:
        st.info("📭 No active signals. The system will generate signals during market hours.")

# ==============================
# TAB 3: RESEARCH
# ==============================

with tab3:
    st.markdown('<div class="sub-header">🔍 Research & Analytics</div>', unsafe_allow_html=True)
    
    # Research metrics
    if st.session_state.research_analytics:
        latest = st.session_state.research_analytics[-1]
        
        # Key research metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            net_gamma = latest.get("gamma_exposure", {}).get("net_gamma", 0)
            st.metric("Net Gamma", f"{net_gamma:,.0f}")
        
        with col2:
            pcr = latest.get("put_call_ratio", 1.0)
            st.metric("Put/Call Ratio", f"{pcr:.2f}")
        
        with col3:
            divergence = latest.get("divergence_score", 0)
            st.metric("Divergence Score", f"{divergence:.2f}")
        
        with col4:
            regime = latest.get("market_regime", "UNKNOWN")
            st.metric("Market Regime", regime)
        
        # Display market insights
        st.markdown("#### 💡 Market Insights")
        
        insights = latest.get("market_insights", [])
        if insights:
            for insight in insights[:5]:
                if "trap" in insight.lower():
                    st.markdown(f'<div class="trap-alert">{insight}</div>', unsafe_allow_html=True)
                elif "divergence" in insight.lower():
                    st.markdown(f'<div class="divergence-alert">{insight}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="research-insight">{insight}</div>', unsafe_allow_html=True)
        else:
            st.info("No insights available yet")
        
        # Structural walls
        st.markdown("#### 🧱 Structural Walls")
        
        walls = latest.get("structural_walls", [])
        if walls:
            walls_df = pd.DataFrame(walls)
            st.dataframe(walls_df, width='stretch')
        
        # Trap detection history
        st.markdown("#### 🎯 Trap Detection History")
        
        if st.session_state.trap_detections:
            traps_df = pd.DataFrame([
                {
                    "Time": pd.to_datetime(t["timestamp"]).strftime("%H:%M:%S"),
                    "Type": t["trap"].get("direction", ""),
                    "Strike": t["trap"].get("strike", 0),
                    "Confidence": t["trap"].get("confidence", 0),
                    "Regime": t.get("market_regime", "")
                }
                for t in st.session_state.trap_detections
            ])
            
            if not traps_df.empty:
                st.dataframe(traps_df, width='stretch')
        
        # Research analytics over time
        st.markdown("#### 📊 Analytics Timeline")
        
        if len(st.session_state.research_analytics) > 5:
            # Convert to DataFrame for plotting
            analytics_df = pd.DataFrame(st.session_state.research_analytics[-50:])
            
            if "timestamp" in analytics_df.columns:
                analytics_df["time"] = pd.to_datetime(analytics_df["timestamp"]).dt.strftime("%H:%M")
                
                # Plot OI Velocity and Gamma over time
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    subplot_titles=("OI Velocity", "Net Gamma"),
                    vertical_spacing=0.1
                )
                
                if "oi_velocity" in analytics_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=analytics_df["time"],
                            y=analytics_df["oi_velocity"],
                            mode='lines+markers',
                            name='OI Velocity',
                            line=dict(color='#10B981')
                        ),
                        row=1, col=1
                    )
                
                if "gamma_exposure" in analytics_df.columns:
                    # Extract net_gamma from gamma_exposure dict
                    try:
                        gamma_values = []
                        for g in analytics_df["gamma_exposure"]:
                            if isinstance(g, dict) and "net_gamma" in g:
                                gamma_values.append(g["net_gamma"])
                            else:
                                gamma_values.append(0)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=analytics_df["time"],
                                y=gamma_values,
                                mode='lines+markers',
                                name='Net Gamma',
                                line=dict(color='#8B5CF6')
                            ),
                            row=2, col=1
                        )
                    except:
                        pass
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, width='stretch')

# ==============================
# TAB 4: CONFIGURATION
# ==============================

with tab4:
    st.markdown('<div class="sub-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
    
    # Available indices and underlyings
    available_indices = [
        "NSE_INDEX|Nifty 50",
        "NSE_INDEX|Nifty Bank", 
        "NSE_INDEX|Nifty Fin Service",
        "NSE_INDEX|Nifty Midcap 100",
        "NSE_INDEX|Nifty Next 50",
        "BSE_INDEX|S&P BSE SENSEX"  
    ]
    
    available_underlyings = [
        "NIFTY",
        "BANKNIFTY", 
        "FINNIFTY",
        "MIDCPNIFTY",
        "SENSEX"  
    ]
    
    # Get available expiries for the selected underlying
    current_underlying = st.session_state.config["underlying"]
    from data.instrument_master import get_available_expiries
    
    try:
        available_expiries = get_available_expiries(current_underlying)
        if not available_expiries:
            available_expiries = ["2026-01-27", "2026-02-03", "2026-02-10"]
    except:
        available_expiries = ["2026-01-27", "2026-02-03", "2026-02-10"]
    
    # Configuration form
    with st.form("configuration_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Index Symbol Dropdown
            index_symbol = st.selectbox(
                "Index Symbol", 
                options=available_indices,
                index=available_indices.index(st.session_state.config["index_symbol"]) 
                if st.session_state.config["index_symbol"] in available_indices else 0
            )
            
            # Underlying Symbol Dropdown
            underlying = st.selectbox(
                "Underlying Symbol", 
                options=available_underlyings,
                index=available_underlyings.index(st.session_state.config["underlying"])
                if st.session_state.config["underlying"] in available_underlyings else 0
            )
            
            # Expiry Date Dropdown
            # Find today's date in available_expiries
            today_str = datetime.now().strftime("%Y-%m-%d")
            default_index = 0

            if today_str in available_expiries:
                default_index = available_expiries.index(today_str)
            elif today_str == "2026-01-28":  # Today's date
                # Add today to available expiries if it's not there
                available_expiries.insert(0, today_str)
                default_index = 0

            expiry_date = st.selectbox(
                "Expiry Date",
                options=available_expiries,
                index=default_index  # ✅ Now selects today if available
            )
            
            # Convert selected expiry to datetime
            if expiry_date:
                try:
                    expiry_datetime = datetime.strptime(expiry_date, "%Y-%m-%d")
                    days_to_expiry = (expiry_datetime.date() - datetime.now().date()).days
                    st.info(f"Days to expiry: {days_to_expiry}")
                except:
                    st.warning("Invalid expiry date format")
        
        with col2:
            max_option_keys = st.number_input(
                "Max Option Keys", 
                min_value=50, 
                max_value=500, 
                value=st.session_state.config["max_option_keys"],
                help="Maximum number of option strikes to fetch"
            )
            
            scheduler_interval = st.slider(
                "Scheduler Interval (seconds)", 
                min_value=10, 
                max_value=300, 
                value=30,
                step=5,
                help="How often to refresh data (10-300 seconds)"
            )
            
            confidence_threshold = st.slider(
                "Signal Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2, 
                step=0.05,
                format="%.2f",
                help="Minimum confidence level for signals (0.0-1.0)"
            )
            
            # Advanced options expander
            with st.expander("Advanced Options"):
                enable_live_trading = st.checkbox(
                    "Enable Live Trading", 
                    value=False,
                    help="WARNING: Enable actual order placement"
                )
                
                risk_per_trade = st.number_input(
                    "Risk per Trade (%)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="Maximum risk per trade as percentage of capital"
                )
        
        # Submit button
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            if st.form_submit_button("💾 Save Configuration", width='stretch'):
                # Update configuration
                st.session_state.config.update({
                    "index_symbol": index_symbol,
                    "underlying": underlying,
                    "expiry_date": datetime.strptime(expiry_date, "%Y-%m-%d") if expiry_date else datetime.now(timezone.utc) + timedelta(days=3),
                    "max_option_keys": max_option_keys,
                    "enable_live_trading": enable_live_trading,
                    "risk_per_trade": risk_per_trade
                })
                
                # Update scheduler
                st.session_state.scheduler = MarketScheduler(interval_seconds=scheduler_interval)
                
                # Update signal engine
                st.session_state.signal_engine = create_signal_engine(
                    signal_expiry_minutes=5,
                    confidence_threshold=confidence_threshold
                )
                
                # Clear option keys cache
                st.session_state.option_keys = []
                
                st.success("✅ Configuration saved!")
                st.rerun()
    
    st.markdown("---")
    
    # Quick Configuration Presets
    st.markdown("#### 🚀 Quick Presets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 NIFTY Weekly", width='stretch'):
            st.session_state.config.update({
                "index_symbol": "NSE_INDEX|Nifty 50",
                "underlying": "NIFTY"
            })
            st.success("Set to NIFTY Weekly")
            st.rerun()
    
    with col2:
        if st.button("🏦 BANKNIFTY Weekly", width='stretch'):
            st.session_state.config.update({
                "index_symbol": "NSE_INDEX|Nifty Bank",
                "underlying": "BANKNIFTY"
            })
            st.success("Set to BANKNIFTY Weekly")
            st.rerun()
    
    with col3:
        if st.button("💳 FINNIFTY Weekly", width='stretch'):
            st.session_state.config.update({
                "index_symbol": "NSE_INDEX|Nifty Fin Service",
                "underlying": "FINNIFTY"
            })
            st.success("Set to FINNIFTY Weekly")
            st.rerun()
    
    # Current Configuration Display
    st.markdown("---")
    st.markdown("#### 🖥️ Current Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**Trading Settings:**")
        st.text(f"Index: {st.session_state.config['index_symbol']}")
        st.text(f"Underlying: {st.session_state.config['underlying']}")
        st.text(f"Expiry: {st.session_state.config['expiry_date'].strftime('%Y-%m-%d')}")
        st.text(f"Max Options: {st.session_state.config['max_option_keys']}")
    
    with config_col2:
        st.markdown("**System Settings:**")
        st.text(f"Scheduler: {st.session_state.scheduler.interval_seconds}s")
        st.text(f"Signal Confidence: {confidence_threshold:.2f}")
        st.text(f"Live Trading: {'Enabled' if st.session_state.config.get('enable_live_trading', False) else 'Disabled'}")
        if st.session_state.config.get('enable_live_trading', False):
            st.text(f"Risk per Trade: {st.session_state.config.get('risk_per_trade', 1.0)}%")
    
    # Diagnostic tools (keep your existing code here)
    st.markdown("---")
    st.markdown("#### 🔧 Diagnostic Tools")
    # ... keep your existing diagnostic tools code ...
    
    # Diagnostic tools
    st.markdown("---")
    st.markdown("#### 🔧 Diagnostic Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Test Upstox Connection", width='stretch'):
            # Test connection by trying to fetch profile
            try:
                profile = client.fetch_profile()
                if profile:
                    st.success(f"✅ Upstox connection working")
                else:
                    st.error("❌ Failed to fetch profile")
            except Exception as e:
                st.error(f"❌ Connection test failed: {e}")
    
    with col2:
        if st.button("📊 Test Feature Pipeline", width='stretch'):
            try:
                # Get latest features
                features = fetch_latest_features(FEATURE_VERSION)
                if features is not None and not features.empty():
                    st.success(f"✅ Feature pipeline working ({len(features)} features)")
                else:
                    st.warning("⚠️ No features found")
            except Exception as e:
                st.error(f"❌ Feature pipeline error: {e}")
    
    with col3:
        if st.button("🔄 Clear Option Keys Cache", width='stretch'):
            st.session_state.option_keys = []
            st.success("✅ Option keys cache cleared")

# ==============================
# PERFORMANCE MODAL
# ==============================

if st.session_state.get("show_performance", False):
    with st.expander("📈 Performance Analytics", expanded=True):
        st.markdown("### 📈 Trading Performance")
        
        performance = fetch_signal_performance(days=30)
        
        if performance:
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", performance["total_signals"])
            
            with col2:
                st.metric("Profitable Trades", performance["profitable_signals"])
            
            with col3:
                win_rate = performance["win_rate"]
                color = "normal" if win_rate > 50 else "inverse"
                st.metric("Win Rate", f"{win_rate:.1f}%", delta_color=color)
            
            with col4:
                st.metric("Analysis Period", f"{performance['period_days']} days")
            
            # Breakdown by signal type
            st.markdown("#### Breakdown by Signal Type")
            
            if "breakdown" in performance:
                breakdown_df = pd.DataFrame(performance["breakdown"])
                st.dataframe(breakdown_df, width='stretch')
        
        st.session_state.show_performance = False

# ==============================
# LOGS MODAL
# ==============================

if st.session_state.get("show_logs", False):
    with st.expander("📋 System Logs", expanded=True):
        st.markdown("### 📋 Recent System Activity")
        
        # Could fetch from system_health table here
        st.info("Logs would be displayed here from the database.")
        
        st.session_state.show_logs = False

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
    Algorithmic Trading Research Platform v2.0 | 
    Research Features: OI Velocity, Gamma Exposure, Walls/Traps, Divergence Analysis
</div>
""", unsafe_allow_html=True)

# Simple auto-refresh timer
import time

if "last_auto_run" not in st.session_state:
    st.session_state.last_auto_run = time.time()

current_time = time.time()
time_since_last = current_time - st.session_state.last_auto_run

if time_since_last > 30:  # 30 seconds
    print(f"🔄 Simple timer triggered after {int(time_since_last)}s")
    
    # Execute research cycle
    research_data = execute_research_cycle()
    
    if research_data:
        st.session_state.last_auto_run = current_time
        print(f"✅ Auto-refresh executed successfully")
        
        # Update scheduler's last run time
        st.session_state.scheduler._last_run = datetime.now()
        st.session_state.scheduler._total_executions += 1
        
        # Update auto-refresh manager
        st.session_state.auto_refresh_manager.last_execution_time = datetime.now()
        st.session_state.auto_refresh_manager.execution_count += 1
        
        # Force UI update
        st.rerun()
    else:
        print("❌ Auto-refresh failed")
else:
    print(f"⏱️ Next auto-refresh in {int(30 - time_since_last)}s")