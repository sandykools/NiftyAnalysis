"""
Enhanced Breadth Features with Research Calculations.
Includes CCC analysis, market breadth, and constituent analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

# ==============================
# BREADTH FEATURE CALCULATIONS
# ==============================

def compute_breadth_features(
    constituents_df: pd.DataFrame,
    ccc_history: pd.Series
) -> Dict[str, float]:
    """
    Compute breadth-based features including research metrics.
    """
    # Base features
    ccc_value = compute_ccc_value(constituents_df)
    ccc_slope = compute_ccc_slope(ccc_history)
    
    # Advanced breadth features
    if not constituents_df.empty:
        breadth_metrics = compute_market_breadth(constituents_df)
        weight_distribution = analyze_weight_distribution(constituents_df)
        sector_analysis = analyze_sector_concentration(constituents_df)
        momentum_breadth = compute_momentum_breadth(constituents_df)
    else:
        breadth_metrics = {
            "advance_decline_ratio": 1.0,
            "percent_above_ma": 0.5,
            "breadth_momentum": 0.0
        }
        weight_distribution = {
            "top5_concentration": 0.3,
            "herfindahl_index": 0.1,
            "weight_skew": 0.0
        }
        sector_analysis = {
            "sector_concentration": 0.5,
            "dominant_sector_strength": 0.0
        }
        momentum_breadth = {
            "positive_momentum_ratio": 0.5,
            "momentum_dispersion": 0.0
        }
    
    # Combine all features
    features = {
        # Base CCC features
        "ccc_value": ccc_value,
        "ccc_slope": ccc_slope,
        
        # Market breadth
        "advance_decline_ratio": breadth_metrics["advance_decline_ratio"],
        "percent_above_ma": breadth_metrics["percent_above_ma"],
        "breadth_momentum": breadth_metrics["breadth_momentum"],
        
        # Weight distribution
        "top5_concentration": weight_distribution["top5_concentration"],
        "herfindahl_index": weight_distribution["herfindahl_index"],
        "weight_skew": weight_distribution["weight_skew"],
        
        # Sector analysis
        "sector_concentration": sector_analysis["sector_concentration"],
        "dominant_sector_strength": sector_analysis["dominant_sector_strength"],
        
        # Momentum breadth
        "positive_momentum_ratio": momentum_breadth["positive_momentum_ratio"],
        "momentum_dispersion": momentum_breadth["momentum_dispersion"],
        
        # Derived features
        "breadth_health": compute_breadth_health_score(
            ccc_value, ccc_slope, breadth_metrics
        ),
        "market_participation": compute_market_participation(constituents_df)
    }
    
    return features

# ==============================
# CCC CALCULATIONS
# ==============================

def compute_ccc_value(
    constituents_df: pd.DataFrame,
    weight_col: str = "weight",
    price_change_col: str = "price_change"
) -> float:
    """
    Compute Cumulative Constituent Contribution (CCC).
    """
    if constituents_df.empty:
        return 0.0
    
    required_cols = {weight_col, price_change_col}
    if not required_cols.issubset(constituents_df.columns):
        raise ValueError(
            f"Missing required columns: {required_cols - set(constituents_df.columns)}"
        )
    
    ccc = (constituents_df[weight_col] * constituents_df[price_change_col]).sum()
    return float(ccc)

def build_constituents_df(
    equity_quotes_df: pd.DataFrame,
    weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Build constituents DataFrame.
    Handles BOTH:
      - weights keyed as SYMBOL
      - weights keyed as NSE_EQ|SYMBOL
    """

    if equity_quotes_df.empty or not weights:
        return pd.DataFrame()

    rows = []

    for _, row in equity_quotes_df.iterrows():
        symbol = row.get("symbol")

        if not symbol:
            continue

        # 🔑 Normalize weight lookup
        weight = (
            weights.get(symbol) or
            weights.get(f"NSE_EQ|{symbol}")
        )

        if weight is None:
            continue

        open_price = row.get("open") or row.get("close")
        ltp = row.get("ltp")

        if open_price is None or open_price <= 0 or ltp is None:
            continue


        price_change = (ltp - open_price) / open_price

        rows.append({
            "symbol": symbol,
            "weight": weight,
            "price_change": price_change,
            "ltp": ltp,
            "open": open_price
        })

    return pd.DataFrame(rows)


def compute_ccc_slope(
    ccc_series: pd.Series,
    lookback: int = 5
) -> float:
    """
    Compute slope of CCC over last N points.
    """
    if len(ccc_series) <= lookback:
        return 0.0
    
    y = ccc_series.iloc[-lookback:]
    x = np.arange(len(y))
    
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)

# ==============================
# MARKET BREADTH ANALYSIS
# ==============================

def compute_market_breadth(constituents_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute comprehensive market breadth metrics.
    """
    if constituents_df.empty:
        return {
            "advance_decline_ratio": 1.0,
            "percent_above_ma": 0.5,
            "breadth_momentum": 0.0,
            "new_highs_lows": 0.0,
            "breadth_thrust": 0.0
        }
    
    # Advance-Decline ratio
    advances = constituents_df[constituents_df["price_change"] > 0]
    declines = constituents_df[constituents_df["price_change"] < 0]
    
    advance_count = len(advances)
    decline_count = len(declines)
    
    if decline_count == 0:
        advance_decline_ratio = float(advance_count) if advance_count > 0 else 1.0
    else:
        advance_decline_ratio = advance_count / decline_count
    
    # Percent above "moving average" (simplified as above open)
    above_open = constituents_df[constituents_df["price_change"] > 0]
    percent_above = len(above_open) / len(constituents_df)
    
    # Breadth momentum (weighted average of price changes)
    if constituents_df["weight"].sum() > 0:
        breadth_momentum = (constituents_df["weight"] * constituents_df["price_change"]).sum()
    else:
        breadth_momentum = constituents_df["price_change"].mean()
    
    # New highs/lows (simplified as extreme moves)
    price_change_std = constituents_df["price_change"].std()
    if price_change_std > 0:
        extreme_positives = constituents_df[
            constituents_df["price_change"] > 2 * price_change_std
        ]
        extreme_negatives = constituents_df[
            constituents_df["price_change"] < -2 * price_change_std
        ]
        new_highs_lows = (len(extreme_positives) - len(extreme_negatives)) / len(constituents_df)
    else:
        new_highs_lows = 0.0
    
    # Breadth thrust (rapid improvement in breadth)
    # Simplified as acceleration in advance-decline ratio
    breadth_thrust = 0.0  # Would require historical data
    
    return {
        "advance_decline_ratio": float(advance_decline_ratio),
        "percent_above_ma": float(percent_above),
        "breadth_momentum": float(breadth_momentum),
        "new_highs_lows": float(new_highs_lows),
        "breadth_thrust": float(breadth_thrust)
    }

# ==============================
# WEIGHT DISTRIBUTION ANALYSIS
# ==============================

def analyze_weight_distribution(constituents_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze weight distribution of index constituents.
    """
    if constituents_df.empty:
        return {
            "top5_concentration": 0.3,
            "herfindahl_index": 0.1,
            "weight_skew": 0.0,
            "weight_gini": 0.5
        }
    
    weights = constituents_df["weight"].values
    
    # Top 5 concentration
    weights_sorted = np.sort(weights)[::-1]
    top5_concentration = np.sum(weights_sorted[:5]) if len(weights_sorted) >= 5 else np.sum(weights_sorted)
    
    # Herfindahl-Hirschman Index (concentration measure)
    herfindahl_index = np.sum(weights ** 2)
    
    # Weight skewness
    if len(weights) >= 3:
        weight_skew = stats.skew(weights)
    else:
        weight_skew = 0.0
    
    # Gini coefficient (inequality measure)
    if len(weights) >= 2:
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)
        cumulative = np.cumsum(sorted_weights)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    else:
        gini = 0.5
    
    return {
        "top5_concentration": float(top5_concentration),
        "herfindahl_index": float(herfindahl_index),
        "weight_skew": float(weight_skew),
        "weight_gini": float(gini)
    }

# ==============================
# SECTOR ANALYSIS
# ==============================

def analyze_sector_concentration(constituents_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze sector concentration (simplified version).
    In production, this would use actual sector data.
    """
    if constituents_df.empty:
        return {
            "sector_concentration": 0.5,
            "dominant_sector_strength": 0.0,
            "sector_correlation": 0.0
        }
    
    # Simplified: group by weight quartiles
    if len(constituents_df) >= 4:
        quartiles = pd.qcut(constituents_df["weight"], 4, labels=False)
        sector_concentration = 1.0 - len(set(quartiles)) / 4  # Higher = more concentrated
    else:
        sector_concentration = 0.5
    
    # Dominant sector strength (weight of top quartile)
    if len(constituents_df) >= 4:
        top_quartile = constituents_df.nlargest(len(constituents_df) // 4, "weight")
        dominant_sector_strength = top_quartile["weight"].sum()
    else:
        dominant_sector_strength = constituents_df["weight"].max()
    
    # Sector correlation (simplified as correlation of top weights)
    if len(constituents_df) >= 5:
        top_5 = constituents_df.nlargest(5, "weight")
        # Use price changes as proxy for sector performance
        if "price_change" in top_5.columns:
            sector_correlation = top_5["price_change"].std() / (abs(top_5["price_change"].mean()) + 1e-10)
            sector_correlation = min(sector_correlation, 1.0)
        else:
            sector_correlation = 0.5
    else:
        sector_correlation = 0.5
    
    return {
        "sector_concentration": float(sector_concentration),
        "dominant_sector_strength": float(dominant_sector_strength),
        "sector_correlation": float(sector_correlation)
    }

# ==============================
# MOMENTUM BREADTH
# ==============================

def compute_momentum_breadth(constituents_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute momentum breadth across constituents.
    """
    if constituents_df.empty or "price_change" not in constituents_df.columns:
        return {
            "positive_momentum_ratio": 0.5,
            "momentum_dispersion": 0.0,
            "momentum_trend": 0.0
        }
    
    price_changes = constituents_df["price_change"].values
    weights = constituents_df["weight"].values
    
    # Positive momentum ratio
    positive_momentum_ratio = np.sum(price_changes > 0) / len(price_changes)
    
    # Momentum dispersion (standard deviation of momentum)
    if len(price_changes) >= 2:
        momentum_dispersion = np.std(price_changes)
    else:
        momentum_dispersion = 0.0
    
    # Weighted momentum trend
    if np.sum(weights) > 0:
        weighted_momentum = np.sum(weights * price_changes) / np.sum(weights)
    else:
        weighted_momentum = np.mean(price_changes)
    
    return {
        "positive_momentum_ratio": float(positive_momentum_ratio),
        "momentum_dispersion": float(momentum_dispersion),
        "momentum_trend": float(weighted_momentum)
    }

# ==============================
# DERIVED FEATURES
# ==============================

def compute_breadth_health_score(
    ccc_value: float,
    ccc_slope: float,
    breadth_metrics: Dict[str, float]
) -> float:
    """
    Compute composite breadth health score (0-1).
    """
    scores = []
    
    # CCC value score
    ccc_score = min(abs(ccc_value) * 10, 1.0)
    scores.append(ccc_score * 0.3)
    
    # CCC slope score (positive slope is healthy)
    if ccc_slope > 0:
        slope_score = min(ccc_slope * 100, 1.0)
    else:
        slope_score = max(ccc_slope * 50, -1.0)  # Negative slope penalized
    scores.append(max(slope_score, 0) * 0.2)
    
    # Advance-decline ratio score
    adr = breadth_metrics.get("advance_decline_ratio", 1.0)
    if adr > 1.0:
        adr_score = min((adr - 1.0) * 2, 1.0)  # Cap at 1.0
    else:
        adr_score = max(adr - 0.5, 0) * 2  # Below 0.5 is bad
    scores.append(adr_score * 0.2)
    
    # Percent above MA score
    percent_above = breadth_metrics.get("percent_above_ma", 0.5)
    percent_score = abs(percent_above - 0.5) * 2  # Distance from 50%
    scores.append(percent_score * 0.15)
    
    # Breadth momentum score
    breadth_momentum = breadth_metrics.get("breadth_momentum", 0.0)
    momentum_score = min(abs(breadth_momentum) * 100, 1.0)
    scores.append(momentum_score * 0.15)
    
    return float(np.sum(scores))

def compute_market_participation(constituents_df: pd.DataFrame) -> float:
    """
    Compute market participation score (0-1).
    Higher = broader market participation in moves.
    """
    if constituents_df.empty or "price_change" not in constituents_df.columns:
        return 0.5
    
    price_changes = constituents_df["price_change"].values
    
    if len(price_changes) < 2:
        return 0.5
    
    # Participation = 1 - (fraction of stocks with near-zero changes)
    near_zero = np.sum(np.abs(price_changes) < 0.001) / len(price_changes)
    participation = 1.0 - near_zero
    
    # Adjust for direction consistency
    positive_fraction = np.sum(price_changes > 0) / len(price_changes)
    direction_consistency = max(positive_fraction, 1 - positive_fraction)
    
    participation = participation * direction_consistency
    
    return float(participation)

# ==============================
# DIVERGENCE DETECTION
# ==============================

def detect_breadth_divergence(
    price_change: float,
    breadth_metrics: Dict[str, float],
    historical_breadth: List[Dict]
) -> Tuple[bool, str, float]:
    """
    Detect divergence between price and breadth indicators.
    
    Returns:
        has_divergence, direction, confidence
    """
    if len(historical_breadth) < 5:
        return False, "NEUTRAL", 0.0
    
    # Get recent breadth values
    recent_breadth = historical_breadth[-5:]
    
    # Calculate breadth momentum
    breadth_values = [b.get("advance_decline_ratio", 1.0) for b in recent_breadth]
    breadth_momentum = np.polyfit(range(len(breadth_values)), breadth_values, 1)[0]
    
    # Price momentum (simplified)
    price_momentum = price_change
    
    # Detect divergence
    if price_momentum > 0.01 and breadth_momentum < -0.1:
        # Price up but breadth deteriorating (bearish divergence)
        confidence = min(abs(breadth_momentum) * 10, 1.0)
        return True, "BEARISH", confidence
    
    elif price_momentum < -0.01 and breadth_momentum > 0.1:
        # Price down but breadth improving (bullish divergence)
        confidence = min(abs(breadth_momentum) * 10, 1.0)
        return True, "BULLISH", confidence
    
    return False, "NEUTRAL", 0.0

# ==============================
# UTILITY FUNCTIONS
# ==============================

def calculate_breadth_indicators(constituents_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate traditional breadth indicators.
    """
    if constituents_df.empty:
        return {}
    
    indicators = {}
    
    # McClellan Oscillator components
    advances = len(constituents_df[constituents_df["price_change"] > 0])
    declines = len(constituents_df[constituents_df["price_change"] < 0])
    
    indicators["advance_decline_line"] = advances - declines
    indicators["advance_decline_ratio"] = advances / declines if declines > 0 else advances
    
    # Arms Index (TRIN)
    if declines > 0:
        advance_volume = 1  # Simplified
        decline_volume = 1  # Simplified
        indicators["arms_index"] = (advances / declines) / (advance_volume / decline_volume)
    else:
        indicators["arms_index"] = 0.0
    
    # Percent above moving average (simplified)
    avg_change = constituents_df["price_change"].mean()
    indicators["percent_above_average"] = len(
        constituents_df[constituents_df["price_change"] > avg_change]
    ) / len(constituents_df)
    
    return indicators

def get_breadth_alerts(breadth_features: Dict[str, float]) -> List[str]:
    """
    Generate breadth-based alerts.
    """
    alerts = []
    
    # Check CCC value
    ccc_value = breadth_features.get("ccc_value", 0.0)
    if ccc_value > 0.01:
        alerts.append("Strong positive breadth (CCC > 1%)")
    elif ccc_value < -0.01:
        alerts.append("Strong negative breadth (CCC < -1%)")
    
    # Check advance-decline ratio
    adr = breadth_features.get("advance_decline_ratio", 1.0)
    if adr > 2.0:
        alerts.append("Extreme breadth: Advances >> Declines")
    elif adr < 0.5:
        alerts.append("Extreme breadth: Declines >> Advances")
    
    # Check breadth health
    health = breadth_features.get("breadth_health", 0.5)
    if health > 0.8:
        alerts.append("Excellent breadth health")
    elif health < 0.3:
        alerts.append("Poor breadth health")
    
    # Check market participation
    participation = breadth_features.get("market_participation", 0.5)
    if participation > 0.8:
        alerts.append("Broad market participation")
    elif participation < 0.3:
        alerts.append("Narrow market participation")
    
    return alerts[:3]  # Return top 3 alerts