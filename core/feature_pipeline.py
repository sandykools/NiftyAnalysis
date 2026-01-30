"""
Enhanced Feature Pipeline with Research-Based Feature Engineering.
Implements all research concepts:
1. OI Velocity & Gamma Exposure features
2. Structural Walls & Traps features
3. Spot Divergence features
4. Market microstructure features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from datetime import timezone

from ml.feature_contract import FEATURE_VERSION, FEATURE_COLUMNS, TARGET_COLUMN
from storage.repository import insert_market_features
from data.upstox_client import UpstoxClient, MarketAnalytics

# Import feature computation modules
from features.option_features import compute_option_features
from features.price_features import compute_price_features
from features.breadth import compute_breadth_features


from enum import Enum

class PipelineMode(str, Enum):
    RESEARCH = "research"
    EXECUTION = "execution"

# ==============================
# RESEARCH-BASED FEATURE ENGINEERING
# ==============================

@dataclass
class ResearchFeatures:
    """Container for research-based features"""
    # OI Velocity features
    oi_velocity: float
    oi_velocity_ma: float  # Moving average
    oi_velocity_std: float  # Standard deviation
    oi_regime: str  # EXPANSIVE/NORMAL/CONSTRICTED
    
    # Gamma Exposure features
    net_gamma: float
    gamma_regime: str  # POSITIVE/NEGATIVE/NEUTRAL
    gamma_flip_distance: float  # Distance to nearest gamma flip
    max_gamma_strike: float
    
    # Structural features
    wall_strength: float  # Combined strength of top walls
    wall_defense_score: float  # Defense score (0-1)
    trap_probability: float  # Probability of trap formation
    
    # Divergence features
    price_oi_divergence: float  # Divergence between price and OI
    price_gamma_divergence: float  # Divergence between price and gamma
    divergence_score: float  # Combined divergence score
    
    # Market microstructure
    put_call_ratio: float
    max_pain_distance: float  # Distance to max pain
    vix_smile: float  # Volatility smile curvature
    skewness: float  # Option skew
    
    # Wyckoff-inspired features
    spring_detection: float  # Bear trap probability
    upthrust_detection: float  # Bull trap probability
    accumulation_score: float  # Accumulation phase score

class EnhancedFeatureEngine:
    """
    Enhanced feature engineering engine incorporating research concepts.
    Transforms raw market data into research-based predictive features.
    """
    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """
        Convert None / NaN / invalid values into a safe float.
        """
        if isinstance(value, (int, float)) and not np.isnan(value):
            return float(value)
        return default

    def __init__(self, lookback_periods: int = 20):
        self.lookback = lookback_periods
        self.feature_history = []
        self.oi_velocity_history = []
        self.gamma_history = []
        
    def extract_research_features(self, 
                                option_chain_analytics: Dict,
                                spot_price: float,
                                price_series: pd.Series,
                                volume_series: pd.Series,
                                constituents_df: pd.DataFrame) -> ResearchFeatures:
        """
        Extract research-based features from market analytics.
        """
        analytics = option_chain_analytics.get("analytics", {})
        
        # Add null checks
        if not analytics:
            # Return default/empty features
            return ResearchFeatures(
                oi_velocity=0.0,
                oi_velocity_ma=0.0,
                oi_velocity_std=0.0,
                oi_regime="NORMAL",
                net_gamma=0.0,
                gamma_regime="NEUTRAL",
                gamma_flip_distance=0.0,
                max_gamma_strike=0.0,
                wall_strength=0.0,
                wall_defense_score=0.0,
                trap_probability=0.0,
                price_oi_divergence=0.0,
                price_gamma_divergence=0.0,
                divergence_score=0.0,
                put_call_ratio=1.0,
                max_pain_distance=0.0,
                vix_smile=0.0,
                skewness=0.0,
                spring_detection=0.0,
                upthrust_detection=0.0,
                accumulation_score=0.0
            )
        
        # OI Velocity features
        oi_velocity = self._safe_float(analytics.get("oi_velocity"))
        oi_regime = analytics.get("oi_regime", "NORMAL")
        
        # Update history and calculate statistics
        self.oi_velocity_history.append(oi_velocity)
        if len(self.oi_velocity_history) > self.lookback:
            self.oi_velocity_history = self.oi_velocity_history[-self.lookback:]
        
        oi_velocity_ma = np.mean(self.oi_velocity_history) if self.oi_velocity_history else 0.0
        oi_velocity_std = np.std(self.oi_velocity_history) if len(self.oi_velocity_history) > 1 else 1.0
        
        # Gamma Exposure features
        gamma_data = analytics.get("gamma_exposure", {})
        net_gamma = gamma_data.get("net_gamma", 0.0)
        gamma_regime = gamma_data.get("regime", "NEUTRAL")
        flip_levels = gamma_data.get("flip_levels", [])
        max_gamma_strike = gamma_data.get("max_impact_strike", 0.0)
        
        # Calculate distance to nearest gamma flip
        gamma_flip_distance = 0.0
        if flip_levels:
            valid_levels = [
                self._safe_float(level)
                for level in flip_levels
                if isinstance(level, (int, float))
            ]

            if valid_levels and spot_price > 0:
                distances = [
                    abs(level - spot_price) / spot_price
                    for level in valid_levels
                ]
                gamma_flip_distance = min(distances) if distances else 0.0
        
        # Update gamma history
        self.gamma_history.append(self._safe_float(net_gamma))
        if len(self.gamma_history) > self.lookback:
            self.gamma_history = self.gamma_history[-self.lookback:]
        
        # Structural features
        walls = analytics.get("structural_walls", [])
        traps = analytics.get("potential_traps", [])
        
        # Wall strength (weighted by concentration and defense)
        wall_strength = 0.0
        wall_defense_score = 0.0
        if walls:
            for wall in walls:
                concentration = wall.get("concentration", 0.0)
                defended = 1.0 if wall.get("defended", False) else 0.0
                distance = wall.get("distance_pct", 100) / 100  # Normalize
                
                # Walls closer to spot have more impact
                proximity_factor = 1.0 / (1.0 + distance)
                wall_strength += concentration * proximity_factor
                wall_defense_score += defended * concentration
        
        # Normalize wall features
        wall_strength = min(wall_strength, 1.0)
        wall_defense_score = min(wall_defense_score, 1.0)
        
        # Trap probability
        trap_probability = 0.0
        if traps:
            trap_confidences = [trap.get("confidence", 0.0) for trap in traps]
            trap_probability = max(trap_confidences) if trap_confidences else 0.0
        
        # Divergence features
        divergence_data = analytics.get("spot_divergence", {})
        has_divergence = divergence_data.get("has_divergence", False)
        divergence_type = divergence_data.get("type", "")
        divergence_confidence = divergence_data.get("confidence", 0.0)
        
        # Calculate price-OI divergence
        price_change = 0.0
        if len(price_series) >= 2:
            prev_price = self._safe_float(price_series.iloc[-2])
            curr_price = self._safe_float(price_series.iloc[-1])

            if prev_price > 0:
                price_change = (curr_price - prev_price) / prev_price * 100

        
        price_oi_divergence = abs(
            (price_change or 0.0) -
            (oi_velocity or 0.0)
        )
        
        # Calculate price-gamma divergence
        price_gamma_divergence = 0.0
        if self.gamma_history and len(self.gamma_history) >= 2:
            gamma_change = (
                    self._safe_float(self.gamma_history[-1]) -
                    self._safe_float(self.gamma_history[-2])
                )
            price_gamma_divergence = abs(
                        self._safe_float(price_change) -
                        self._safe_float(gamma_change) * 100
                    )
        
        divergence_score = divergence_confidence if has_divergence else 0.0
        
        # Market microstructure features
        pcr_data = MarketAnalytics.calculate_put_call_ratio(option_chain_analytics.get("raw_data", pd.DataFrame()))
        put_call_ratio = pcr_data.get("pcr_oi")
        put_call_ratio = float(put_call_ratio) if put_call_ratio is not None else 0.0

        
        max_pain_data = MarketAnalytics.detect_max_pain(option_chain_analytics.get("raw_data", pd.DataFrame()))
        max_pain_strike = max_pain_data.get("max_pain_strike")
        if max_pain_strike is None or spot_price <= 0:
            max_pain_distance = 0.0
        else:
            max_pain_distance = abs(spot_price - max_pain_strike) / spot_price

        
        # Calculate VIX smile (simplified)
        vix_smile = self._calculate_vix_smile(option_chain_analytics.get("raw_data", pd.DataFrame()), spot_price)
        
        # Calculate skewness
        skewness = self._calculate_option_skew(option_chain_analytics.get("raw_data", pd.DataFrame()))
        
        # Wyckoff-inspired features
        spring_detection = self._detect_spring_pattern(price_series, volume_series, constituents_df)
        upthrust_detection = self._detect_upthrust_pattern(price_series, volume_series, constituents_df)
        accumulation_score = self._calculate_accumulation_score(price_series, volume_series, oi_velocity)
        
        return ResearchFeatures(
            oi_velocity=oi_velocity,
            oi_velocity_ma=oi_velocity_ma,
            oi_velocity_std=oi_velocity_std,
            oi_regime=oi_regime,
            
            net_gamma=net_gamma,
            gamma_regime=gamma_regime,
            gamma_flip_distance=gamma_flip_distance,
            max_gamma_strike=max_gamma_strike,
            
            wall_strength=wall_strength,
            wall_defense_score=wall_defense_score,
            trap_probability=trap_probability,
            
            price_oi_divergence=price_oi_divergence,
            price_gamma_divergence=price_gamma_divergence,
            divergence_score=divergence_score,
            
            put_call_ratio=put_call_ratio,
            max_pain_distance=max_pain_distance,
            vix_smile=vix_smile,
            skewness=skewness,
            
            spring_detection=spring_detection,
            upthrust_detection=upthrust_detection,
            accumulation_score=accumulation_score
        )
    
    def _calculate_vix_smile(self, option_chain_df: pd.DataFrame, spot_price: float) -> float:
        """Calculate volatility smile curvature."""
        if option_chain_df.empty or spot_price <= 0:
            return 0.0
        
        # Group by distance from spot
        option_chain_df = option_chain_df.copy()
        option_chain_df['distance_pct'] = abs(option_chain_df['strike'] - spot_price) / spot_price * 100
        
        # Bin by distance
        bins = [0, 2, 5, 10, 20]
        smiles = []
        
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]
            
            mask = (option_chain_df['distance_pct'] >= lower) & (option_chain_df['distance_pct'] < upper)
            bin_iv = option_chain_df.loc[mask, 'iv'].mean()
            
            if not np.isnan(bin_iv):
                smiles.append(bin_iv)
        
        # Calculate smile curvature (higher = steeper smile)
        if len(smiles) >= 3:
            return smiles[0] - smiles[-1]  # ATM vs far OTM
        return 0.0
    
    def _calculate_option_skew(self, option_chain_df: pd.DataFrame) -> float:
        """Calculate option skew (put IV - call IV)."""
        if option_chain_df.empty:
            return 0.0
        
        put_iv = option_chain_df[option_chain_df['option_type'] == 'PE']['iv'].mean()
        call_iv = option_chain_df[option_chain_df['option_type'] == 'CE']['iv'].mean()
        
        if np.isnan(put_iv) or np.isnan(call_iv):
            return 0.0
        
        return put_iv - call_iv  # Positive = put skew (bearish), Negative = call skew (bullish)
    
    def _detect_spring_pattern(self, price_series: pd.Series, 
                              volume_series: pd.Series,
                              constituents_df: pd.DataFrame) -> float:
        """Detect Wyckoff spring pattern (bear trap)."""
        if len(price_series) < 10 or len(volume_series) < 10:
            return 0.0
        
        # Simplified spring detection
        recent_prices = price_series.iloc[-5:].values
        recent_volumes = volume_series.iloc[-5:].values
        
        # Spring: price makes lower low but closes above previous low
        if len(recent_prices) >= 5:
            low1 = np.min(recent_prices[:3])  # First low
            low2 = np.min(recent_prices[2:])  # Second low (potential spring)
            close = recent_prices[-1]
            
            # Volume spike on second low
            volume_spike = recent_volumes[-2] > np.mean(recent_volumes[:-1]) * 1.5
            
            if low2 < low1 * 0.995 and close > low1 and volume_spike:
                # Calculate spring probability
                price_recovery = (close - low2) / low2
                return min(price_recovery * 10, 1.0)
        
        return 0.0
    
    def _detect_upthrust_pattern(self, price_series: pd.Series,
                                volume_series: pd.Series,
                                constituents_df: pd.DataFrame) -> float:
        """Detect Wyckoff upthrust pattern (bull trap)."""
        if len(price_series) < 10 or len(volume_series) < 10:
            return 0.0
        
        recent_prices = price_series.iloc[-5:].values
        recent_volumes = volume_series.iloc[-5:].values
        
        # Upthrust: price makes higher high but closes below previous high
        if len(recent_prices) >= 5:
            high1 = np.max(recent_prices[:3])  # First high
            high2 = np.max(recent_prices[2:])  # Second high (potential upthrust)
            close = recent_prices[-1]
            
            # Volume spike on second high
            volume_spike = recent_volumes[-2] > np.mean(recent_volumes[:-1]) * 1.5
            
            if high2 > high1 * 1.005 and close < high1 and volume_spike:
                # Calculate upthrust probability
                price_rejection = (high2 - close) / high2
                return min(price_rejection * 10, 1.0)
        
        return 0.0
    
    def _calculate_accumulation_score(self, price_series: pd.Series,
                                     volume_series: pd.Series,
                                     oi_velocity: float) -> float:
        """Calculate accumulation/distribution score."""
        if len(price_series) < 20 or len(volume_series) < 20:
            return 0.0
        
        # Price in trading range
        price_range = price_series.iloc[-20:]
        range_high = price_range.max()
        range_low = price_range.min()
        range_width = (range_high - range_low) / range_low
        
        # Low volatility in range (accumulation)
        volatility = price_range.pct_change().std()
        
        # Volume analysis
        volume_trend = np.polyfit(range(len(volume_series.iloc[-20:])), 
                                 volume_series.iloc[-20:].values, 1)[0]
        
        # OI building during range (accumulation)
        oi_building = oi_velocity > 0.5
        
        # Calculate accumulation score
        score = 0.0
        
        # Narrow range with low volatility
        if range_width < 0.02 and volatility < 0.005:
            score += 0.3
        
        # Volume declining or stable (not distribution)
        if volume_trend <= 0:
            score += 0.2
        
        # OI building
        if oi_building:
            score += 0.3
        
        # Price near range lows (better accumulation)
        current_price = price_series.iloc[-1]
        if current_price < range_low * 1.02:
            score += 0.2
        
        return min(score, 1.0)

# ==============================
# ENHANCED FEATURE PIPELINE
# ==============================

# Global feature engine
_feature_engine = EnhancedFeatureEngine()

def build_and_store_features(
    *,
    timestamp: pd.Timestamp,
    option_chain_df: pd.DataFrame,
    spot_price: float,
    expiry_datetime,
    ltp: float,
    vwap: float,
    price_series: pd.Series,
    volume_series: pd.Series,
    constituents_df: pd.DataFrame,
    ccc_history: pd.Series,
    client: Optional[UpstoxClient] = None,
    option_keys: Optional[list] = None,
    mode: PipelineMode = PipelineMode.RESEARCH  # ⬅️ NEW
) -> Optional[Dict]:

    """
    Enhanced feature pipeline with research-based feature engineering.
    
    Args:
        client: UpstoxClient instance (required for advanced analytics)
        option_keys: Option keys for fetching enhanced analytics
    
    Returns:
        Dictionary with features and research analytics
    """
    
    # ------------------------------
    # TIMESTAMP VALIDATION
    # ------------------------------
    if not isinstance(timestamp, pd.Timestamp):
        raise TypeError("timestamp must be pandas.Timestamp")
    
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
    
    # ------------------------------
    # FETCH ENHANCED ANALYTICS (if client provided)
    # ------------------------------
    research_analytics = None

    if mode == PipelineMode.EXECUTION:
        if not client:
            raise RuntimeError("EXECUTION mode requires UpstoxClient")

        if not option_keys:
            raise RuntimeError("EXECUTION mode requires non-empty option_keys")

        try:
            research_analytics = client.fetch_option_chain_with_analytics(
                option_keys, spot_price
            )
        except Exception as e:
            raise RuntimeError(f"Execution analytics unavailable: {e}")

    else:
        # RESEARCH MODE — best effort only
        if client and option_keys:
            try:
                research_analytics = client.fetch_option_chain_with_analytics(
                    option_keys, spot_price
                )
            except Exception as e:
                print(f"⚠️ Research analytics skipped: {e}")
                research_analytics = None

    
    # ------------------------------
    # COMPUTE BASE FEATURES
    # ------------------------------
    option_feats = compute_option_features(option_chain_df, spot_price, expiry_datetime)
    price_feats = compute_price_features(ltp, vwap, price_series, volume_series)
    breadth_feats = compute_breadth_features(constituents_df, ccc_history)
    
    # ------------------------------
    # COMPUTE RESEARCH FEATURES
    # ------------------------------
    research_feats_dict = {}
    if research_analytics:
        try:
            research_feats = _feature_engine.extract_research_features(
                research_analytics,
                spot_price,
                price_series,
                volume_series,
                constituents_df
            )
            
            # Convert research features to dictionary
            research_feats_dict = {
                # OI Velocity features
                "oi_velocity": research_feats.oi_velocity,
                "oi_velocity_ma": research_feats.oi_velocity_ma,
                "oi_velocity_std": research_feats.oi_velocity_std,
                "oi_regime_expansive": 1.0 if research_feats.oi_regime == "EXPANSIVE" else 0.0,
                "oi_regime_constricted": 1.0 if research_feats.oi_regime == "CONSTRICTED" else 0.0,
                
                # Gamma Exposure features
                "net_gamma": research_feats.net_gamma,
                "gamma_regime_positive": 1.0 if "POSITIVE" in research_feats.gamma_regime else 0.0,
                "gamma_regime_negative": 1.0 if "NEGATIVE" in research_feats.gamma_regime else 0.0,
                "gamma_flip_distance": research_feats.gamma_flip_distance,
                "max_gamma_strike_distance": abs(spot_price - research_feats.max_gamma_strike) / spot_price,
                
                # Structural features
                "wall_strength": research_feats.wall_strength,
                "wall_defense_score": research_feats.wall_defense_score,
                "trap_probability": research_feats.trap_probability,
                
                # Divergence features
                "price_oi_divergence": research_feats.price_oi_divergence,
                "price_gamma_divergence": research_feats.price_gamma_divergence,
                "divergence_score": research_feats.divergence_score,
                "has_divergence": 1.0 if research_feats.divergence_score > 0.3 else 0.0,
                
                # Market microstructure
                "put_call_ratio": research_feats.put_call_ratio,
                "max_pain_distance": research_feats.max_pain_distance,
                "vix_smile": research_feats.vix_smile,
                "skewness": research_feats.skewness,
                
                # Wyckoff features
                "spring_detection": research_feats.spring_detection,
                "upthrust_detection": research_feats.upthrust_detection,
                "accumulation_score": research_feats.accumulation_score,
                
                # Derived features
                "gamma_wall_interaction": research_feats.wall_strength * abs(research_feats.net_gamma),
                "velocity_divergence_composite": research_feats.oi_velocity * research_feats.divergence_score,
                "trap_gamma_composite": research_feats.trap_probability * (1.0 if research_feats.gamma_regime == "NEGATIVE" else 0.0)
            }
        except Exception as e:
            print(f"Warning: Research feature extraction failed: {e}")
            research_feats_dict = {}
    
    # ------------------------------
    # BUILD FEATURE ROW
    # ------------------------------
    feature_row = {
        "timestamp": timestamp_str,
        "feature_version": FEATURE_VERSION,
        "future_return_5m": None
    }
    
    # Fill ALL required feature columns explicitly
    for col in FEATURE_COLUMNS:
        if col == "timestamp":
            continue

        if col in option_feats:
            feature_row[col] = option_feats[col]
        elif col in price_feats:
            feature_row[col] = price_feats[col]
        elif col in breadth_feats:
            feature_row[col] = breadth_feats[col]
        else:
            feature_row[col] = 0.0

    
    # Add time to expiry with proper timezone handling
    def normalize_datetime(dt):
        """Normalize datetime to UTC timezone-aware."""
        if dt is None:
            return None
        
        # If it's already timezone-aware, convert to UTC
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.astimezone(timezone.utc)
        
        # If it's timezone-naive, assume UTC and make it aware
        if isinstance(dt, pd.Timestamp):
            if dt.tz is None:
                return dt.tz_localize('UTC')
            else:
                return dt.tz_convert('UTC')
        elif isinstance(dt, datetime):
            return dt.replace(tzinfo=timezone.utc)
        
        return dt
    
    # Normalize both timestamps to UTC
    if expiry_datetime and timestamp:
        expiry_utc = normalize_datetime(expiry_datetime)
        timestamp_utc = normalize_datetime(timestamp)
        
        if expiry_utc and timestamp_utc:
            time_diff = (expiry_utc - timestamp_utc).total_seconds() / 60
            feature_row["time_to_expiry_minutes"] = max(int(time_diff), 0)
        else:
            feature_row["time_to_expiry_minutes"] = 0
    else:
        feature_row["time_to_expiry_minutes"] = 0
    
    # Merge research features
    feature_row.update(research_feats_dict)
    
    # ------------------------------
    # VALIDATE AND PERSIST
    # ------------------------------
    # Create DataFrame with all columns (base + research)
    all_columns = list(feature_row.keys())
    df = pd.DataFrame([feature_row], columns=all_columns)
    
    # Debug info
    print(f"FEATURE PIPELINE: Generated {len(feature_row)} features")
    print(f"Timestamp: {timestamp_str}")
    print(f"Spot Price: {spot_price}")
    
    if research_analytics:
        print(f"Research Analytics: {len(research_feats_dict)} research features added")
        # Log key metrics
        analytics = research_analytics.get("analytics", {})
        print(f"OI Velocity: {analytics.get('oi_velocity', 'N/A')}")
        print(f"Gamma Regime: {analytics.get('gamma_exposure', {}).get('regime', 'N/A')}")
        print(f"Market Regime: {analytics.get('market_regime', 'N/A')}")
    
    # Check for NULLs
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        print(f"WARNING: NULL values in columns: {null_cols}")
        # Fill NULLs with 0 for numeric columns
        for col in null_cols:
            if col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(0.0)
    
    assert not df.empty, "Feature DataFrame is empty"
    
    # Persist to database
    insert_market_features(df)
    
    print(f"✓ Features stored successfully at {timestamp_str}")

    execution_ready = (
        mode == PipelineMode.EXECUTION and
        research_analytics is not None and
        len(research_feats_dict) > 0 and
        feature_row.get("oi_velocity_std", 0) > 0 and
        feature_row.get("time_to_expiry_minutes", 0) > 0
    )


    # Return comprehensive result
    return {
        "features": feature_row,
        "research_analytics": research_analytics.get("analytics", {}) if research_analytics else {},
        "market_insights": research_analytics.get("market_insights", []) if research_analytics else [],
        "timestamp": timestamp_str,
        "spot_price": spot_price,
        "execution_ready": execution_ready
    }


# ==============================
# UTILITY FUNCTIONS
# ==============================

def create_feature_summary(feature_row: Dict) -> Dict:
    """Create a summary of key features for display."""
    summary = {
        "timestamp": feature_row.get("timestamp", "N/A"),
        "base_features": {},
        "research_features": {},
        "signals": {}
    }
    
    # Base features
    base_keys = ["put_call_ratio", "vwap_distance", "price_momentum", 
                 "ccc_value", "ccc_slope", "time_to_expiry_minutes"]
    for key in base_keys:
        if key in feature_row:
            summary["base_features"][key] = feature_row[key]
    
    # Research features (top level)
    research_keys = ["oi_velocity", "net_gamma", "trap_probability", 
                    "divergence_score", "wall_strength"]
    for key in research_keys:
        if key in feature_row:
            summary["research_features"][key] = feature_row[key]
    
    # Generate signals
    signals = []
    
    # OI Velocity signal
    oi_vel = feature_row.get("oi_velocity", 0)
    if oi_vel > 1.5:
        signals.append("📈 Strong OI Buildup")
    elif oi_vel < -1.5:
        signals.append("📉 OI Unwinding")
    
    # Gamma signal
    gamma_regime_pos = feature_row.get("gamma_regime_positive", 0)
    gamma_regime_neg = feature_row.get("gamma_regime_negative", 0)
    if gamma_regime_pos > 0.5:
        signals.append("📌 Positive Gamma (Stabilizing)")
    elif gamma_regime_neg > 0.5:
        signals.append("🚀 Negative Gamma (Accelerating)")
    
    # Trap signal
    trap_prob = feature_row.get("trap_probability", 0)
    if trap_prob > 0.7:
        signals.append("🎯 High Trap Probability")
    elif trap_prob > 0.5:
        signals.append("⚠️ Moderate Trap Risk")
    
    # Divergence signal
    has_div = feature_row.get("has_divergence", 0)
    if has_div > 0.5:
        signals.append("🔍 Divergence Detected")
    
    summary["signals"] = signals
    
    return summary

def validate_feature_contract(feature_row: Dict) -> bool:
    """Validate feature row against contract."""
    # Check required columns
    required_base = ["timestamp", "feature_version"]
    for col in required_base:
        if col not in feature_row:
            print(f"Missing required column: {col}")
            return False
    
    # Check data types
    for col, value in feature_row.items():
        if col == "timestamp":
            continue
        if col == "feature_version":
            if not isinstance(value, str):
                print(f"Invalid type for {col}: expected str, got {type(value)}")
                return False
        elif col in ["oi_regime_expansive", "oi_regime_constricted", 
                    "gamma_regime_positive", "gamma_regime_negative", "has_divergence"]:
            # Binary features
            if not isinstance(value, (int, float)):
                print(f"Invalid type for {col}: expected numeric, got {type(value)}")
                return False
        elif isinstance(value, (int, float)):
            # Numeric features - check for extreme values
            if abs(value) > 1e6:  # Unreasonable large value
                print(f"Extreme value for {col}: {value}")
                return False
            if pd.isna(value):
                print(f"NaN value for {col}")
                return False
    
    return True