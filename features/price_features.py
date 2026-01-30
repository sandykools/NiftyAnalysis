"""
Enhanced Price Features with Research Calculations.
Includes VWAP distance, momentum, volume analysis, and divergence detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from scipy import stats

# ==============================
# PRICE FEATURE CALCULATIONS
# ==============================

def compute_price_features(
    ltp: float,
    vwap: float,
    price_series: pd.Series,
    volume_series: pd.Series
) -> Dict[str, float]:
    """
    Compute price-based features including research metrics.
    """
    # Base features
    vwap_distance = compute_vwap_distance(ltp, vwap)
    price_momentum = compute_price_momentum(price_series)
    volume_ratio = compute_volume_ratio(volume_series)
    
    # Advanced features
    trend_strength = compute_trend_strength(price_series)
    volatility = compute_volatility(price_series)
    rsi = compute_rsi(price_series)
    volume_profile = compute_volume_profile(price_series, volume_series)
    price_efficiency = compute_price_efficiency(price_series)
    
    # Divergence detection
    volume_price_divergence = compute_volume_price_divergence(price_series, volume_series)
    
    # Combine all features
    features = {
        # Base features
        "vwap_distance": vwap_distance,
        "price_momentum": price_momentum,
        "volume_ratio": volume_ratio,
        
        # Trend and momentum
        "trend_strength": trend_strength,
        "price_volatility": volatility,
        "rsi": rsi,
        "price_efficiency": price_efficiency,
        
        # Volume analysis
        "volume_trend": volume_profile["trend"],
        "volume_volatility": volume_profile["volatility"],
        "volume_clustering": volume_profile["clustering"],
        
        # Divergence
        "volume_price_divergence": volume_price_divergence,
        "has_volume_divergence": 1.0 if abs(volume_price_divergence) > 0.5 else 0.0
    }
    
    return features

# ==============================
# BASE FEATURE CALCULATIONS
# ==============================

def compute_vwap_distance(ltp: float, vwap: float) -> float:
    """Calculate distance from VWAP."""
    if vwap == 0:
        return 0.0
    
    return float((ltp - vwap) / vwap)

def compute_price_momentum(
    price_series: pd.Series,
    lookback: int = 5
) -> float:
    """Calculate price momentum over lookback period."""
    if len(price_series) <= lookback:
        return 0.0
    
    past_price = price_series.iloc[-lookback - 1]
    current_price = price_series.iloc[-1]
    
    if past_price == 0:
        return 0.0
    
    return float((current_price - past_price) / past_price)

def compute_volume_ratio(
    volume_series: pd.Series,
    lookback: int = 20
) -> float:
    """Calculate current volume relative to average."""
    if len(volume_series) < lookback:
        return 0.0
    
    current_volume = volume_series.iloc[-1]
    avg_volume = volume_series.iloc[-lookback:].mean()
    
    if avg_volume == 0:
        return 0.0
    
    return float(current_volume / avg_volume)

# ==============================
# ADVANCED FEATURE CALCULATIONS
# ==============================

def compute_trend_strength(price_series: pd.Series, lookback: int = 20) -> float:
    """
    Calculate trend strength using linear regression.
    Returns R² value (0-1) indicating trend strength.
    """
    if len(price_series) < lookback:
        return 0.0
    
    prices = price_series.iloc[-lookback:].values
    x = np.arange(len(prices))
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
    
    # R² value indicates trend strength
    r_squared = r_value ** 2
    
    return float(r_squared)

def compute_volatility(price_series: pd.Series, lookback: int = 20) -> float:
    """Calculate price volatility (annualized)."""
    if len(price_series) <= 1:
        return 0.0
    
    returns = price_series.pct_change().dropna()
    
    if len(returns) < lookback:
        sample_returns = returns
    else:
        sample_returns = returns.iloc[-lookback:]
    
    if len(sample_returns) <= 1:
        return 0.0
    
    # Annualized volatility (assuming daily data)
    daily_vol = sample_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    
    return float(annual_vol)

def compute_rsi(price_series: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(price_series) <= period:
        return 50.0  # Neutral
    
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def compute_price_efficiency(price_series: pd.Series) -> float:
    """
    Calculate price efficiency (random walk index).
    Higher values indicate more efficient/trending markets.
    """
    if len(price_series) < 10:
        return 0.5
    
    # Calculate Hurst exponent approximation
    n = min(len(price_series), 100)
    lags = range(2, min(n // 2, 20))
    
    if len(lags) < 2:
        return 0.5
    
    tau = []
    for lag in lags:
        # Calculate variance of lagged differences
        price_diff = np.diff(price_series.iloc[-n:], lag)
        if len(price_diff) > 1:
            tau.append(np.std(price_diff))
        else:
            tau.append(0)
    
    tau = np.array(tau)
    lags = np.array(lags)
    
    # Remove zeros
    mask = (tau > 0) & (lags > 0)
    if np.sum(mask) < 2:
        return 0.5
    
    # Linear regression in log space
    try:
        hurst = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
        efficiency = hurst  # H ≈ 0.5 random walk, >0.5 trending, <0.5 mean-reverting
    except:
        efficiency = 0.5
    
    return float(efficiency)

# ==============================
# VOLUME ANALYSIS
# ==============================

def compute_volume_profile(
    price_series: pd.Series,
    volume_series: pd.Series,
    lookback: int = 20
) -> Dict[str, float]:
    """
    Compute volume profile features.
    """
    if len(volume_series) < lookback or len(price_series) < lookback:
        return {
            "trend": 0.0,
            "volatility": 0.0,
            "clustering": 0.0
        }
    
    recent_volume = volume_series.iloc[-lookback:]
    recent_prices = price_series.iloc[-lookback:]
    
    # Volume trend
    x = np.arange(len(recent_volume))
    volume_trend = np.polyfit(x, recent_volume.values, 1)[0]
    volume_trend_normalized = volume_trend / (recent_volume.mean() + 1e-10)
    
    # Volume volatility
    volume_volatility = recent_volume.std() / (recent_volume.mean() + 1e-10)
    
    # Volume clustering (autocorrelation)
    if len(recent_volume) >= 5:
        volume_clustering = recent_volume.autocorr(lag=1)
        if pd.isna(volume_clustering):
            volume_clustering = 0.0
    else:
        volume_clustering = 0.0
    
    return {
        "trend": float(volume_trend_normalized),
        "volatility": float(volume_volatility),
        "clustering": float(volume_clustering)
    }

def compute_volume_price_divergence(
    price_series: pd.Series,
    volume_series: pd.Series,
    lookback: int = 10
) -> float:
    """
    Detect divergence between price and volume.
    Positive = price up, volume down (bearish divergence)
    Negative = price down, volume up (bullish divergence)
    """
    if len(price_series) < lookback or len(volume_series) < lookback:
        return 0.0
    
    # Calculate price and volume changes
    price_change = (price_series.iloc[-1] - price_series.iloc[-lookback]) / price_series.iloc[-lookback]
    volume_change = (volume_series.iloc[-1] - volume_series.iloc[-lookback]) / (volume_series.iloc[-lookback] + 1e-10)
    
    # Normalize
    price_norm = price_change / (np.std(price_series.iloc[-lookback:].pct_change().dropna()) + 1e-10)
    volume_norm = volume_change / (np.std(volume_series.iloc[-lookback:].pct_change().dropna()) + 1e-10)
    
    # Divergence score
    divergence = price_norm - volume_norm
    
    return float(divergence)

# ==============================
# SUPPORTING CALCULATIONS
# ==============================

def detect_price_patterns(price_series: pd.Series) -> Dict[str, float]:
    """
    Detect common price patterns.
    """
    if len(price_series) < 20:
        return {
            "double_top": 0.0,
            "double_bottom": 0.0,
            "head_shoulders": 0.0,
            "triangle": 0.0
        }
    
    # Simplified pattern detection
    prices = price_series.iloc[-20:].values
    
    # Calculate peaks and troughs
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(prices, prominence=np.std(prices) * 0.5)
    troughs, _ = find_peaks(-prices, prominence=np.std(prices) * 0.5)
    
    pattern_scores = {
        "double_top": 0.0,
        "double_bottom": 0.0,
        "head_shoulders": 0.0,
        "triangle": 0.0
    }
    
    # Double top detection
    if len(peaks) >= 2:
        peak1 = prices[peaks[-2]]
        peak2 = prices[peaks[-1]]
        if abs(peak1 - peak2) / peak1 < 0.02:  # Within 2%
            pattern_scores["double_top"] = 0.8
    
    # Double bottom detection
    if len(troughs) >= 2:
        trough1 = prices[troughs[-2]]
        trough2 = prices[troughs[-1]]
        if abs(trough1 - trough2) / trough1 < 0.02:
            pattern_scores["double_bottom"] = 0.8
    
    return pattern_scores

def calculate_support_resistance(
    price_series: pd.Series,
    window: int = 20
) -> Dict[str, float]:
    """
    Calculate support and resistance levels.
    """
    if len(price_series) < window:
        return {
            "support": 0.0,
            "resistance": 0.0,
            "current_position": 0.5
        }
    
    recent_prices = price_series.iloc[-window:]
    
    support = recent_prices.min()
    resistance = recent_prices.max()
    current = price_series.iloc[-1]
    
    if resistance - support == 0:
        position = 0.5
    else:
        position = (current - support) / (resistance - support)
    
    return {
        "support": float(support),
        "resistance": float(resistance),
        "current_position": float(position)
    }

# ==============================
# DIVERGENCE DETECTION
# ==============================

def detect_momentum_divergence(
    price_series: pd.Series,
    momentum_series: pd.Series
) -> Tuple[bool, str, float]:
    """
    Detect divergence between price and momentum oscillator.
    
    Returns:
        has_divergence, direction, confidence
    """
    if len(price_series) < 10 or len(momentum_series) < 10:
        return False, "NEUTRAL", 0.0
    
    # Get recent peaks in price and momentum
    price_peaks = []
    momentum_peaks = []
    
    # Simplified peak detection
    for i in range(5, len(price_series) - 5):
        if price_series.iloc[i] == price_series.iloc[i-5:i+5].max():
            price_peaks.append((i, price_series.iloc[i]))
        if momentum_series.iloc[i] == momentum_series.iloc[i-5:i+5].max():
            momentum_peaks.append((i, momentum_series.iloc[i]))
    
    if len(price_peaks) < 2 or len(momentum_peaks) < 2:
        return False, "NEUTRAL", 0.0
    
    # Check for divergence
    price_trend = price_peaks[-1][1] - price_peaks[-2][1]
    momentum_trend = momentum_peaks[-1][1] - momentum_peaks[-2][1]
    
    # Bullish divergence: price makes lower low, momentum makes higher low
    # Bearish divergence: price makes higher high, momentum makes lower high
    
    if price_trend > 0 and momentum_trend < 0:
        # Bearish divergence
        confidence = min(abs(momentum_trend) / 10, 1.0)
        return True, "BEARISH", confidence
    
    elif price_trend < 0 and momentum_trend > 0:
        # Bullish divergence
        confidence = min(abs(momentum_trend) / 10, 1.0)
        return True, "BULLISH", confidence
    
    return False, "NEUTRAL", 0.0