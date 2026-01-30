"""
Enhanced Option Features with Research Calculations.
Includes Gamma Exposure, OI Velocity, and structural analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
from typing import List, Dict, Tuple, Optional, Any, Union

# ==============================
# BLACK-SCHOLES CALCULATIONS
# ==============================

def black_scholes_greeks(
    S: float,           # Spot price
    K: float,           # Strike price
    T: float,           # Time to expiry (years)
    r: float,           # Risk-free rate
    sigma: float,       # Implied volatility
    option_type: str    # 'CE' or 'PE'
) -> Dict[str, float]:
    """
    Calculate Black-Scholes Greeks for options.
    """
    if T <= 0 or sigma <= 0:
        return {
            "delta": 0.5 if option_type == "CE" else -0.5,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "CE":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:  # PE
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega)
    }

# ==============================
# OPTION CHAIN FEATURES
# ==============================

def compute_option_features(
    option_chain_df: pd.DataFrame,
    spot_price: float,
    expiry_datetime: datetime
) -> Dict[str, float]:
    """
    Compute option chain features including research metrics.
    """
    if option_chain_df.empty:
        return get_default_features()
    
    # Base features
    put_call_ratio = compute_put_call_ratio(option_chain_df)
    oi_delta = compute_oi_delta(option_chain_df)
    oi_concentration = compute_oi_concentration(option_chain_df)
    atm_iv = compute_atm_iv(option_chain_df, spot_price)
    iv_skew = compute_iv_skew(option_chain_df, spot_price)
    
    # Advanced features
    gamma_profile = compute_gamma_profile(option_chain_df, spot_price, expiry_datetime)
    oi_velocity = compute_oi_velocity(option_chain_df)
    max_pain = compute_max_pain(option_chain_df)
    vix_smile = compute_vix_smile(option_chain_df, spot_price)
    
    # Combine all features
    features = {
        # Base features
        "put_call_ratio": put_call_ratio,
        "oi_delta": oi_delta,
        "oi_concentration": oi_concentration,
        "atm_iv": atm_iv,
        "iv_skew": iv_skew,
        
        # Time feature
        "time_to_expiry_minutes": compute_time_to_expiry_minutes(expiry_datetime),
        
        # Research features
        "net_gamma": gamma_profile["net_gamma"],
        "gamma_skew": gamma_profile["skew"],
        "max_gamma_strike": gamma_profile["max_strike"],
        "oi_velocity": oi_velocity,
        "max_pain_strike": max_pain,
        "vix_smile": vix_smile,
        
        # Volume and OI ratios
        "call_oi_ratio": compute_call_oi_ratio(option_chain_df),
        "put_oi_ratio": compute_put_oi_ratio(option_chain_df),
        "volume_put_call_ratio": compute_volume_pcr(option_chain_df)
    }
    
    return features

# ==============================
# BASE FEATURE CALCULATIONS
# ==============================

def compute_put_call_ratio(option_chain_df: pd.DataFrame) -> float:
    """Calculate Put-Call Ratio (OI-based)."""
    if option_chain_df.empty:
        return 1.0
    
    put_oi = option_chain_df.loc[
        option_chain_df["option_type"] == "PE", "oi"
    ].sum()
    
    call_oi = option_chain_df.loc[
        option_chain_df["option_type"] == "CE", "oi"
    ].sum()
    
    if call_oi == 0:
        return 0.0
    
    return float(put_oi / call_oi)

def compute_oi_delta(option_chain_df: pd.DataFrame) -> float:
    """Calculate OI Delta (Put OI change - Call OI change)."""
    if option_chain_df.empty:
        return 0.0
    
    put_delta = option_chain_df.loc[
        option_chain_df["option_type"] == "PE", "oi_change"
    ].sum()
    
    call_delta = option_chain_df.loc[
        option_chain_df["option_type"] == "CE", "oi_change"
    ].sum()
    
    return float(put_delta - call_delta)

def compute_oi_concentration(option_chain_df: pd.DataFrame) -> float:
    """Calculate OI concentration at maximum OI strike."""
    if option_chain_df.empty:
        return 0.0
    
    total_oi = option_chain_df["oi"].sum()
    if total_oi == 0:
        return 0.0
    
    max_strike_oi = (
        option_chain_df
        .groupby("strike")["oi"]
        .sum()
        .max()
    )
    
    return float(max_strike_oi / total_oi)

def compute_atm_iv(option_chain_df: pd.DataFrame, spot_price: float) -> float:
    """Calculate At-The-Money Implied Volatility."""
    if option_chain_df.empty:
        return 0.3  # Default IV
    
    option_chain_df = option_chain_df.copy()
    option_chain_df["dist"] = abs(option_chain_df["strike"] - spot_price)
    
    atm_row = option_chain_df.sort_values("dist").iloc[0]
    iv = atm_row["iv"]
    
    return float(iv) if not np.isnan(iv) else 0.3

def compute_iv_skew(option_chain_df: pd.DataFrame, spot_price: float) -> float:
    """Calculate IV Skew (Call IV - Put IV)."""
    if option_chain_df.empty:
        return 0.0
    
    option_chain_df = option_chain_df.copy()
    option_chain_df["dist"] = abs(option_chain_df["strike"] - spot_price)
    
    # Get nearest strikes
    nearest_strikes = option_chain_df.sort_values("dist").head(10)
    
    ce_iv = nearest_strikes.loc[
        nearest_strikes["option_type"] == "CE", "iv"
    ].mean()
    
    pe_iv = nearest_strikes.loc[
        nearest_strikes["option_type"] == "PE", "iv"
    ].mean()
    
    if np.isnan(ce_iv) or np.isnan(pe_iv):
        return 0.0
    
    return float(ce_iv - pe_iv)

def compute_time_to_expiry_minutes(expiry_datetime: datetime) -> int:
    """Calculate minutes to expiry."""
    now = datetime.utcnow()
    delta = expiry_datetime - now
    
    minutes = int(delta.total_seconds() / 60)
    return max(minutes, 0)

# ==============================
# RESEARCH FEATURE CALCULATIONS
# ==============================

def compute_gamma_profile(
    option_chain_df: pd.DataFrame,
    spot_price: float,
    expiry_datetime: datetime
) -> Dict[str, float]:
    """
    Compute Gamma Exposure profile for research.
    """
    if option_chain_df.empty:
        return {
            "net_gamma": 0.0,
            "positive_gamma": 0.0,
            "negative_gamma": 0.0,
            "skew": 0.0,
            "max_strike": 0.0
        }
    
    
    # Calculate time to expiry in years
    now = datetime.utcnow()
    T = max((expiry_datetime - now).total_seconds() / (365 * 24 * 3600), 0.001)
    r = 0.05  # Risk-free rate approximation
    
    total_gamma = 0.0
    positive_gamma = 0.0
    negative_gamma = 0.0
    gamma_strikes = []
    
    for _, row in option_chain_df.iterrows():
        strike = row['strike']
        option_type = row['option_type']
        oi = row['oi']
        iv = row.get('iv', 0.3)
        
        # Calculate gamma for this strike
        greeks = black_scholes_greeks(
            S=spot_price,
            K=strike,
            T=T,
            r=r,
            sigma=iv,
            option_type=option_type
        )
        
        gamma = greeks["gamma"]
        
        # Adjust sign based on market maker position
        # Market makers are typically short options
        if option_type == "CE":
            gamma_contribution = -gamma * oi * 100  # Negative for short calls
        else:  # PE
            gamma_contribution = gamma * oi * 100   # Positive for short puts
        
        total_gamma += gamma_contribution
        
        if gamma_contribution > 0:
            positive_gamma += gamma_contribution
        else:
            negative_gamma += gamma_contribution
        
        gamma_strikes.append((strike, gamma_contribution))
    
    # Calculate gamma skew and max_strike
    if gamma_strikes:
        strikes, gammas = zip(*gamma_strikes)
        
        # Calculate skew
        if len(gammas) >= 10:
            gamma_skew = np.mean(gammas[:5]) - np.mean(gammas[-5:])
        else:
            gamma_skew = 0.0
        
        # Find max gamma strike safely
        if len(strikes) > 0 and len(gammas) > 0:
            max_idx = np.argmax(np.abs(gammas))
            max_gamma_strike = strikes[max_idx]
            # Ensure it's not None
            if max_gamma_strike is None or pd.isnull(max_gamma_strike):
                max_gamma_strike = 0.0
        else:
            max_gamma_strike = 0.0
            gamma_skew = 0.0
    else:
        gamma_skew = 0.0
        max_gamma_strike = 0.0
    
    return {
        "net_gamma": float(total_gamma),
        "positive_gamma": float(positive_gamma),
        "negative_gamma": float(negative_gamma),
        "skew": float(gamma_skew),
        "max_strike": float(max_gamma_strike)
    }

def compute_oi_velocity(option_chain_df: pd.DataFrame) -> float:
    """
    Calculate OI Velocity (rate of change of OI).
    In production, this would use historical data.
    """
    if option_chain_df.empty:
        return 0.0
    
    # Simplified velocity calculation
    # In real implementation, compare with previous snapshot
    total_oi = option_chain_df["oi"].sum()
    oi_change = option_chain_df["oi_change"].sum()
    
    if total_oi == 0:
        return 0.0
    
    velocity = oi_change / total_oi * 100
    return float(velocity)

def compute_max_pain(option_chain_df: pd.DataFrame) -> float:
    """Calculate Max Pain strike."""
    if option_chain_df.empty:
        return 0.0
    
    # Check if we have strikes
    if 'strike' not in option_chain_df.columns:
        return 0.0
    
    strikes = sorted(option_chain_df['strike'].unique())
    if not strikes:
        return 0.0
    
    pain_values = []
    
    for strike in strikes:
        total_pain = 0
        
        # Calculate pain from puts
        puts = option_chain_df[(option_chain_df['strike'] == strike) & 
                              (option_chain_df['option_type'] == 'PE')]
        for _, put in puts.iterrows():
            # Check if put is ITM (strike > spot for put pain calculation)
            # Actually for max pain, we calculate loss for option writers
            # For puts: writers lose when spot < strike
            put_strike = put['strike']
            if isinstance(put_strike, (int, float)):
                if strike < put_strike:  # ITM puts cause pain
                    put_oi = put['oi'] if pd.notnull(put['oi']) else 0
                    total_pain += (put_strike - strike) * put_oi
        
        # Calculate pain from calls
        calls = option_chain_df[(option_chain_df['strike'] == strike) & 
                               (option_chain_df['option_type'] == 'CE')]
        for _, call in calls.iterrows():
            # For calls: writers lose when spot > strike
            call_strike = call['strike']
            if isinstance(call_strike, (int, float)):
                if strike > call_strike:  # ITM calls cause pain
                    call_oi = call['oi'] if pd.notnull(call['oi']) else 0
                    total_pain += (strike - call_strike) * call_oi
        
        pain_values.append((strike, total_pain))
    
    if pain_values:
        # Find strike with minimum total pain
        min_pain_item = min(pain_values, key=lambda x: x[1])
        max_pain_strike = min_pain_item[0]
        
        # Ensure it's a valid number
        if max_pain_strike is None or pd.isnull(max_pain_strike):
            return 0.0
        return float(max_pain_strike)
    
    return 0.0

def compute_vix_smile(option_chain_df: pd.DataFrame, spot_price: float) -> float:
    """Calculate VIX smile curvature."""
    if option_chain_df.empty or spot_price <= 0:
        return 0.0
    
    # Group by distance from spot
    option_chain_df = option_chain_df.copy()
    option_chain_df['distance_pct'] = abs(option_chain_df['strike'] - spot_price) / spot_price * 100
    
    # Get IVs at different distances
    atm_iv = option_chain_df[
        option_chain_df['distance_pct'] < 2
    ]['iv'].mean()
    
    otm_iv = option_chain_df[
        (option_chain_df['distance_pct'] >= 5) & 
        (option_chain_df['distance_pct'] < 10)
    ]['iv'].mean()
    
    if np.isnan(atm_iv) or np.isnan(otm_iv):
        return 0.0
    
    # Smile = OTM IV - ATM IV (positive = smile, negative = smirk)
    return float(otm_iv - atm_iv)

# ==============================
# SUPPORTING CALCULATIONS
# ==============================

def compute_call_oi_ratio(option_chain_df: pd.DataFrame) -> float:
    """Calculate Call OI as percentage of total OI."""
    if option_chain_df.empty:
        return 0.5
    
    total_oi = option_chain_df["oi"].sum()
    call_oi = option_chain_df[option_chain_df["option_type"] == "CE"]["oi"].sum()
    
    if total_oi == 0:
        return 0.5
    
    return float(call_oi / total_oi)

def compute_put_oi_ratio(option_chain_df: pd.DataFrame) -> float:
    """Calculate Put OI as percentage of total OI."""
    if option_chain_df.empty:
        return 0.5
    
    total_oi = option_chain_df["oi"].sum()
    put_oi = option_chain_df[option_chain_df["option_type"] == "PE"]["oi"].sum()
    
    if total_oi == 0:
        return 0.5
    
    return float(put_oi / total_oi)

def compute_volume_pcr(option_chain_df: pd.DataFrame) -> float:
    """Calculate Volume-based Put-Call Ratio."""
    if option_chain_df.empty:
        return 1.0
    
    put_volume = option_chain_df.loc[
        option_chain_df["option_type"] == "PE", "volume"
    ].sum()
    
    call_volume = option_chain_df.loc[
        option_chain_df["option_type"] == "CE", "volume"
    ].sum()
    
    if call_volume == 0:
        return 0.0
    
    return float(put_volume / call_volume)

def get_default_features() -> Dict[str, float]:
    """Return default feature values when no data is available."""
    return {
        "put_call_ratio": 1.0,
        "oi_delta": 0.0,
        "oi_concentration": 0.0,
        "atm_iv": 0.3,
        "iv_skew": 0.0,
        "time_to_expiry_minutes": 0,
        "net_gamma": 0.0,
        "gamma_skew": 0.0,
        "max_gamma_strike": 0.0,
        "oi_velocity": 0.0,
        "max_pain_strike": 0.0,
        "vix_smile": 0.0,
        "call_oi_ratio": 0.5,
        "put_oi_ratio": 0.5,
        "volume_put_call_ratio": 1.0
    }

# ==============================
# UTILITY FUNCTIONS
# ==============================

def analyze_option_skew(option_chain_df: pd.DataFrame, spot_price: float) -> Dict[str, float]:
    """
    Analyze option skew across strikes.
    """
    if option_chain_df.empty:
        return {
            "call_skew": 0.0,
            "put_skew": 0.0,
            "total_skew": 0.0
        }
    
    # Separate calls and puts
    calls = option_chain_df[option_chain_df["option_type"] == "CE"].copy()
    puts = option_chain_df[option_chain_df["option_type"] == "PE"].copy()
    
    # Calculate distance from spot
    calls["distance_pct"] = (calls["strike"] - spot_price) / spot_price * 100
    puts["distance_pct"] = (spot_price - puts["strike"]) / spot_price * 100
    
    # Calculate skew (slope of IV vs distance)
    call_skew = 0.0
    put_skew = 0.0
    
    if len(calls) >= 5:
        # Sort by distance and take top 5 OTM calls
        otm_calls = calls[calls["distance_pct"] > 0].nlargest(5, "distance_pct")
        if len(otm_calls) >= 2:
            call_skew = np.polyfit(otm_calls["distance_pct"], otm_calls["iv"], 1)[0]
    
    if len(puts) >= 5:
        # Sort by distance and take top 5 OTM puts
        otm_puts = puts[puts["distance_pct"] > 0].nlargest(5, "distance_pct")
        if len(otm_puts) >= 2:
            put_skew = np.polyfit(otm_puts["distance_pct"], otm_puts["iv"], 1)[0]
    
    return {
        "call_skew": float(call_skew),
        "put_skew": float(put_skew),
        "total_skew": float(call_skew + put_skew)
    }

def detect_gamma_flip_levels(
    option_chain_df: pd.DataFrame,
    spot_price: float,
    expiry_datetime: datetime
) -> List[float]:
    """
    Detect price levels where gamma flips sign.
    """
    if option_chain_df.empty:
        return []
    
    # Calculate gamma at different price levels
    price_levels = np.linspace(spot_price * 0.95, spot_price * 1.05, 21)
    gamma_levels = []
    
    T = max((expiry_datetime - datetime.utcnow()).total_seconds() / (365 * 24 * 3600), 0.001)
    r = 0.05
    
    for price in price_levels:
        total_gamma = 0.0
        
        for _, row in option_chain_df.iterrows():
            strike = row['strike']
            option_type = row['option_type']
            oi = row['oi']
            iv = row.get('iv', 0.3)
            
            greeks = black_scholes_greeks(
                S=price,
                K=strike,
                T=T,
                r=r,
                sigma=iv,
                option_type=option_type
            )
            
            gamma = greeks["gamma"]
            
            if option_type == "CE":
                gamma_contribution = -gamma * oi
            else:
                gamma_contribution = gamma * oi
            
            total_gamma += gamma_contribution
        
        gamma_levels.append((price, total_gamma))
    
    # Find sign changes
    flip_levels = []
    for i in range(1, len(gamma_levels)):
        prev_sign = np.sign(gamma_levels[i-1][1])
        curr_sign = np.sign(gamma_levels[i][1])
        
        if prev_sign != curr_sign and prev_sign != 0 and curr_sign != 0:
            flip_price = (gamma_levels[i-1][0] + gamma_levels[i][0]) / 2
            flip_levels.append(float(flip_price))
    
    return flip_levels