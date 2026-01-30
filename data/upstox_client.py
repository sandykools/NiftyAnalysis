"""
Enhanced Upstox Client with advanced derivatives analytics.
Implements research concepts:
1. OI Velocity - Rate of change of Open Interest
2. Gamma Exposure (GEX) - Dealer hedging pressure
3. Structural Walls vs Traps detection
4. Spot Divergence analysis
5. Market microstructure insights
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime, timedelta
from scipy.stats import linregress
from dataclasses import dataclass
from enum import Enum
import streamlit as st

# ==============================
# CONSTANTS & CONFIG
# ==============================

BASE_URL = "https://api.upstox.com/v2"

# Research-based constants
OI_VELOCITY_LOOKBACK = 5  # periods for velocity calculation
GAMMA_LOOKBACK = 10  # strikes for gamma calculation
WALL_THRESHOLD = 0.15  # Min OI concentration for wall detection
TRAP_CONFIRMATION_WINDOW = 3  # periods for trap confirmation

class MarketRegime(Enum):
    """Market regimes based on research"""
    NORMAL = "NORMAL"
    EXPANSIVE = "EXPANSIVE"  # High OI velocity, capital inflow
    CONSTRICTED = "CONSTRICTED"  # Negative OI velocity, capital outflow
    GAMMA_POSITIVE = "GAMMA_POSITIVE"  # Stabilizing, pinning
    GAMMA_NEGATIVE = "GAMMA_NEGATIVE"  # Accelerating, squeezes
    TRAP_FORMING = "TRAP_FORMING"  # Wall breach with unwinding
    DIVERGENCE = "DIVERGENCE"  # Spot vs derivatives divergence

@dataclass
class WallAnalysis:
    """Analysis of structural walls"""
    strike: float
    oi_concentration: float
    option_type: str  # CE or PE
    is_defended: bool
    unwinding_rate: float  # OI velocity at this strike
    gamma_contribution: float
    distance_to_spot: float  # % distance from current spot

@dataclass
class TrapAnalysis:
    """Analysis of potential traps"""
    wall_strike: float
    breach_direction: str  # "UP" or "DOWN"
    confidence: float
    trigger_time: datetime
    gamma_impact: float
    oi_unwinding: float

@dataclass
class GammaProfile:
    """Gamma Exposure profile"""
    net_gamma: float
    positive_gamma_strikes: List[float]
    negative_gamma_strikes: List[float]
    flip_levels: List[float]  # Where gamma changes sign
    max_gamma_strike: float
    regime: str

# ==============================
# ENHANCED UPSTOX CLIENT
# ==============================

class UpstoxClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        })
        
        # State tracking for research concepts
        self.oi_history = {}  # symbol -> list of OI values
        self.price_history = {}  # symbol -> list of prices
        self.gamma_history = {}  # symbol -> list of gamma values
        self.velocity_history = {}  # symbol -> list of velocity values
        
        # Market structure tracking
        self.walls_cache = {}
        self.traps_cache = {}
        self.gex_cache = {}
        
        # Research-based analytics
        self.analytics = MarketAnalytics()
    
    # ==============================
    # CORE REQUEST HANDLER
    # ==============================
    
    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Helper method to make API requests with error handling"""
        url = f"{BASE_URL}/{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Debug logging
            # print(f"Request: {method} {url}")
            # print(f"Params: {kwargs.get('params', {})}")
            # print(f"Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error Response: {response.text}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            try:
                error_data = response.json()
                print(f"Error Details: {json.dumps(error_data, indent=2)}")
            except:
                pass
            raise
        except Exception as e:
            print(f"Request Error: {e}")
            raise
    
    # ==============================
    # RESEARCH IMPLEMENTATION: OI VELOCITY
    # ==============================
    
    def calculate_oi_velocity(self, symbol: str, current_oi: float) -> Tuple[float, str]:
        """
        Calculate Open Interest Velocity with regime classification.
        
        Research Concept: OI Velocity measures the kinetic energy of market trends.
        High positive velocity = Initiative capital entering (Buildup)
        High negative velocity = Capital exiting (Unwinding/Covering)
        
        Returns: (velocity_score, regime)
        """
        if symbol not in self.oi_history:
            self.oi_history[symbol] = []
        
        # Add current OI to history
        self.oi_history[symbol].append(current_oi)
        
        # Keep only last N periods
        if len(self.oi_history[symbol]) > OI_VELOCITY_LOOKBACK:
            self.oi_history[symbol] = self.oi_history[symbol][-OI_VELOCITY_LOOKBACK:]
        
        if len(self.oi_history[symbol]) < 2:
            return 0.0, "INSUFFICIENT_DATA"
        
        # Calculate velocity (rate of change)
        oi_series = pd.Series(self.oi_history[symbol])
        velocity = oi_series.pct_change().iloc[-1] * 100  # % change
        
        # Normalize velocity (research concept)
        if len(oi_series) >= 3:
            std_dev = oi_series.pct_change().std() * 100
            if std_dev > 0:
                normalized_velocity = velocity / std_dev
            else:
                normalized_velocity = velocity
        else:
            normalized_velocity = velocity
        
        # Classify regime
        if normalized_velocity > 1.5:
            regime = "EXPANSIVE"  # High capital inflow
        elif normalized_velocity < -1.5:
            regime = "CONSTRICTED"  # Capital outflow
        else:
            regime = "NORMAL"
        
        # Store for trend analysis
        if symbol not in self.velocity_history:
            self.velocity_history[symbol] = []
        self.velocity_history[symbol].append({
            "timestamp": datetime.utcnow(),
            "velocity": normalized_velocity,
            "regime": regime,
            "raw_oi": current_oi
        })
        
        return normalized_velocity, regime
    
    # ==============================
    # RESEARCH IMPLEMENTATION: GAMMA EXPOSURE
    # ==============================
    
    def calculate_gamma_exposure(self, option_chain_df: pd.DataFrame, spot_price: float):
        """
        Calculate Gamma Exposure (GEX) from option chain.
        """
        # DEBUG: Check input data
        print(f"DEBUG Gamma Input: DataFrame shape: {option_chain_df.shape}")
        print(f"DEBUG Gamma Input: Columns: {option_chain_df.columns.tolist()}")
        print(f"DEBUG Gamma Input: Sample data:")
        print(option_chain_df.head(3))
        print(f"DEBUG Gamma Input: Spot price: {spot_price}")
        
        if option_chain_df.empty:
            print("DEBUG Gamma: Empty option chain")
            return GammaProfile(net_gamma=0.0, positive_gamma_strikes=[], negative_gamma_strikes=[], 
                            flip_levels=[], max_gamma_strike=0.0, regime="NEUTRAL")
        
        # Check for required columns
        required_cols = ['strike', 'option_type', 'oi']
        missing_cols = [col for col in required_cols if col not in option_chain_df.columns]
        
        if missing_cols:
            print(f"DEBUG Gamma: Missing columns: {missing_cols}")
            return GammaProfile(net_gamma=0.0, positive_gamma_strikes=[], negative_gamma_strikes=[],
                            flip_levels=[], max_gamma_strike=0.0, regime="NEUTRAL")
        
        # Calculate gamma for each strike (improved calculation)
        gamma_values = []
        flip_levels = []
        
        for _, row in option_chain_df.iterrows():
            strike = row['strike']
            option_type = row['option_type']
            oi = row['oi']
            
            # FIX 1: Get IV with proper default
            iv = row.get('iv', 0.0)
            if iv <= 0 or pd.isna(iv):
                iv = 0.3  # Reasonable default for NIFTY options
            
            # FIX 2: Calculate distance properly
            distance = abs(strike - spot_price) / spot_price
            
            # FIX 3: Improved gamma formula that doesn't blow up with small IV
            # Gamma is highest ATM (distance = 0) and decays exponentially
            # Using normalized formula that works even with small IV
            gamma_atm = 1.0 / (strike * 0.01)  # Base gamma at ATM
            
            # Apply distance decay
            gamma = gamma_atm * np.exp(-distance * 50)
            
            # Apply IV adjustment (higher IV = lower gamma)
            if iv > 0:
                gamma = gamma / (iv * np.sqrt(365/252))  # Annualized
            
            # Adjust for option type
            # Calls have positive gamma for buyers, negative for sellers
            # Puts have negative gamma for buyers, positive for sellers
            # Market makers (who are typically net sellers) have opposite gamma
            if option_type.upper() in ['PE', 'PUT']:
                gamma = -gamma  # Put writers have short gamma
            
            # Weight by OI (scaled down for reasonable numbers)
            gamma_weighted = gamma * oi / 1000000
            
            gamma_values.append((strike, gamma_weighted))
            
            # Track sign changes for flip levels
            if len(gamma_values) > 1:
                prev_sign = np.sign(gamma_values[-2][1])
                curr_sign = np.sign(gamma_weighted)
                if prev_sign != curr_sign and prev_sign != 0 and curr_sign != 0:
                    flip_level = (strike + gamma_values[-2][0]) / 2
                    flip_levels.append(flip_level)
        
        if not gamma_values:
            return GammaProfile(0.0, [], [], [], 0.0, "NEUTRAL")
        
        # Calculate net gamma
        strikes, gammas = zip(*gamma_values)
        net_gamma = sum(gammas)
        
        # DEBUG: Print gamma statistics
        print(f"DEBUG Gamma: Net gamma: {net_gamma:.2f}")
        print(f"DEBUG Gamma: Min gamma: {min(gammas):.2f}")
        print(f"DEBUG Gamma: Max gamma: {max(gammas):.2f}")
        
        # Separate positive and negative gamma strikes
        positive_strikes = [s for s, g in gamma_values if g > 0]
        negative_strikes = [s for s, g in gamma_values if g < 0]
        
        # Find strike with maximum gamma impact
        if gammas:
            max_gamma_idx = np.argmax(np.abs(gammas))
            max_gamma_strike = strikes[max_gamma_idx]
            max_gamma_value = gammas[max_gamma_idx]
            print(f"DEBUG Gamma: Max gamma strike: {max_gamma_strike} (value: {max_gamma_value:.2f})")
        else:
            max_gamma_strike = 0.0
        
        # Determine regime with better thresholds
        if net_gamma > 10000:
            regime = "GAMMA_POSITIVE"
        elif net_gamma < -10000:
            regime = "GAMMA_NEGATIVE"
        else:
            regime = "NEUTRAL"
        
        print(f"DEBUG Gamma: Regime: {regime}")
        
        return GammaProfile(
            net_gamma=net_gamma,
            positive_gamma_strikes=positive_strikes[:5],  # Top 5
            negative_gamma_strikes=negative_strikes[:5],
            flip_levels=flip_levels,
            max_gamma_strike=max_gamma_strike,
            regime=regime
        )
    # ==============================
    # RESEARCH IMPLEMENTATION: WALLS VS TRAPS
    # ==============================
    
    def analyze_structural_walls(self, option_chain_df: pd.DataFrame, 
                                 spot_price: float) -> List[WallAnalysis]:
        """
        Identify structural walls with defense analysis.
        
        Research Concept: Walls are high OI concentrations that act as barriers.
        Traps form when walls breach with OI unwinding.
        """
        if option_chain_df.empty:
            return []
        
        walls = []
        
        # Group by strike to find OI concentrations
        strike_groups = option_chain_df.groupby('strike')
        
        for strike, group in strike_groups:
            total_oi = group['oi'].sum()
            call_oi = group[group['option_type'] == 'CE']['oi'].sum()
            put_oi = group[group['option_type'] == 'PE']['oi'].sum()
            
            # Determine dominant option type at this strike
            if call_oi > put_oi:
                dominant_type = "CE"
                dominant_oi = call_oi
            else:
                dominant_type = "PE"
                dominant_oi = put_oi
            
            # Calculate OI concentration (research concept)
            total_all_oi = option_chain_df['oi'].sum()
            if total_all_oi > 0:
                oi_concentration = dominant_oi / total_all_oi
            else:
                oi_concentration = 0
            
            # Check if this is a wall (high concentration)
            if oi_concentration > WALL_THRESHOLD:
                # Calculate distance from spot
                distance_pct = abs(strike - spot_price) / spot_price * 100
                
                # Analyze defense strength
                # Defense is stronger if:
                # 1. High OI concentration
                # 2. Close to spot price
                # 3. Recent OI increase (build-up)
                
                # Calculate unwinding rate (OI velocity at this strike)
                strike_key = f"{strike}_{dominant_type}"
                if strike_key in self.oi_history:
                    strike_oi_series = pd.Series(self.oi_history[strike_key])
                    if len(strike_oi_series) > 1:
                        unwinding_rate = strike_oi_series.pct_change().iloc[-1] * 100
                    else:
                        unwinding_rate = 0
                else:
                    unwinding_rate = 0
                
                # Determine if wall is being defended
                # Negative unwinding = defending (adding positions)
                # Positive unwinding = abandoning (closing positions)
                is_defended = unwinding_rate < 0  # Adding OI = defending
                
                walls.append(WallAnalysis(
                    strike=strike,
                    oi_concentration=oi_concentration,
                    option_type=dominant_type,
                    is_defended=is_defended,
                    unwinding_rate=unwinding_rate,
                    gamma_contribution=0,  # Would calculate in production
                    distance_to_spot=distance_pct
                ))
        
        # Sort by OI concentration (highest first)
        walls.sort(key=lambda x: x.oi_concentration, reverse=True)
        return walls[:5]  # Return top 5 walls
    
    def detect_traps(self, walls: List[WallAnalysis], spot_price: float,
                     oi_velocity: float) -> List[TrapAnalysis]:
        """
        Detect potential traps forming at walls.
        
        Research Concept: Traps occur when:
        1. Price breaches a wall
        2. OI starts unwinding (velocity negative)
        3. Gamma is negative (accelerating)
        4. High confidence of squeeze
        """
        traps = []
        
        for wall in walls:
            # Check if price is near wall (±1%)
            price_to_wall_ratio = spot_price / wall.strike
            is_near_wall = 0.99 <= price_to_wall_ratio <= 1.01
            
            if is_near_wall:
                # Check for trap conditions
                trap_confidence = 0.0
                
                # Condition 1: OI unwinding (negative velocity)
                if wall.unwinding_rate > 1.0:  # Rapid unwinding
                    trap_confidence += 0.3
                
                # Condition 2: Overall OI velocity negative
                if oi_velocity < -1.0:
                    trap_confidence += 0.3
                
                # Condition 3: Wall not defended
                if not wall.is_defended:
                    trap_confidence += 0.2
                
                # Condition 4: Price already breached (for detection)
                if (wall.option_type == "CE" and spot_price > wall.strike) or \
                   (wall.option_type == "PE" and spot_price < wall.strike):
                    trap_confidence += 0.2
                
                if trap_confidence > 0.5:  # Minimum confidence threshold
                    breach_direction = "UP" if wall.option_type == "CE" else "DOWN"
                    
                    trap = TrapAnalysis(
                        wall_strike=wall.strike,
                        breach_direction=breach_direction,
                        confidence=trap_confidence,
                        trigger_time=datetime.utcnow(),
                        gamma_impact=0,  # Would calculate actual gamma
                        oi_unwinding=wall.unwinding_rate
                    )
                    traps.append(trap)
        
        return traps
    
    # ==============================
    # RESEARCH IMPLEMENTATION: SPOT DIVERGENCE
    # ==============================
    
    def analyze_spot_divergence(self, spot_price: float, spot_velocity: float,
                               oi_velocity: float, gamma_profile: GammaProfile) -> Dict:
        """
        Analyze divergence between spot price and derivatives metrics.
        
        Research Concept: Divergence reveals internal market weakness.
        Bullish divergence: Price down but OI/Gamma improving
        Bearish divergence: Price up but OI/Gamma deteriorating
        """
        divergence_analysis = {
            "has_divergence": False,
            "type": None,  # BULLISH/BEARISH
            "confidence": 0.0,
            "metrics": {}
        }
        
        # Price vs OI divergence
        if spot_velocity > 0 and oi_velocity < -0.5:
            # Price up but OI unwinding (bearish divergence)
            divergence_analysis["has_divergence"] = True
            divergence_analysis["type"] = "BEARISH"
            divergence_analysis["confidence"] = min(abs(oi_velocity) / 2, 1.0)
            divergence_analysis["metrics"]["price_oi_divergence"] = {
                "price_change": spot_velocity,
                "oi_change": oi_velocity,
                "interpretation": "Hollow rally - price rising on covering, not new buying"
            }
        
        elif spot_velocity < 0 and oi_velocity > 0.5:
            # Price down but OI building (bullish divergence)
            divergence_analysis["has_divergence"] = True
            divergence_analysis["type"] = "BULLISH"
            divergence_analysis["confidence"] = min(oi_velocity / 2, 1.0)
            divergence_analysis["metrics"]["price_oi_divergence"] = {
                "price_change": spot_velocity,
                "oi_change": oi_velocity,
                "interpretation": "Accumulation - smart money buying despite price drop"
            }
        
        # Price vs Gamma divergence
        if gamma_profile.regime == "GAMMA_POSITIVE" and spot_velocity > 1.0:
            # Positive gamma (stabilizing) but price moving fast
            divergence_analysis["has_divergence"] = True
            divergence_analysis["type"] = "BEARISH"
            divergence_analysis["confidence"] = max(divergence_analysis["confidence"], 0.3)
            divergence_analysis["metrics"]["price_gamma_divergence"] = {
                "gamma_regime": gamma_profile.regime,
                "price_velocity": spot_velocity,
                "interpretation": "Price moving against gamma regime - likely to revert"
            }
        
        return divergence_analysis
    
    # ==============================
    # ENHANCED OPTION CHAIN FETCH
    # ==============================
    
    def fetch_option_chain_with_analytics(self, option_keys: list[str], 
                                         spot_price: float) -> Dict:
        """
        Enhanced option chain fetch with full research analytics.
        Returns comprehensive analysis including OI velocity, GEX, walls, traps.
        """
        # Fetch raw option chain
        option_chain_df = self.fetch_option_chain(option_keys)
        
        # DEBUG: Add diagnostic logging
        print(f"DEBUG: Fetched option chain shape: {option_chain_df.shape}")
        print(f"DEBUG: Option chain columns: {option_chain_df.columns.tolist() if not option_chain_df.empty else 'EMPTY'}")
        if not option_chain_df.empty:
            print(f"DEBUG: Option types: {option_chain_df['option_type'].unique()}")
            print(f"DEBUG: OI sample: {option_chain_df['oi'].head(5).tolist()}")
        
        if option_chain_df.empty:
            return {
                "raw_data": option_chain_df,
                "analytics": {},
                "warnings": ["No option data available"]
            }
        
        # Calculate total OI for velocity
        total_oi = option_chain_df['oi'].sum()
        print(f"DEBUG: Total OI calculated: {total_oi}")
        
        # OI Velocity analysis
        oi_velocity, oi_regime = self.calculate_oi_velocity("INDEX", total_oi)
        print(f"DEBUG: OI Velocity: {oi_velocity}, Regime: {oi_regime}")
        
        # Gamma Exposure analysis
        gamma_profile = self.calculate_gamma_exposure(option_chain_df, spot_price)
        print(f"DEBUG: Gamma profile net_gamma: {gamma_profile.net_gamma if gamma_profile else 'N/A'}")
        
        # Structural walls analysis
        walls = self.analyze_structural_walls(option_chain_df, spot_price)
        print(f"DEBUG: Found {len(walls)} structural walls")
        
        # Trap detection
        traps = self.detect_traps(walls, spot_price, oi_velocity)
        print(f"DEBUG: Detected {len(traps)} potential traps")
        
        # Spot price velocity (approximate)
        if "INDEX" not in self.price_history:
            self.price_history["INDEX"] = []
        self.price_history["INDEX"].append(spot_price)
        if len(self.price_history["INDEX"]) > OI_VELOCITY_LOOKBACK:
            self.price_history["INDEX"] = self.price_history["INDEX"][-OI_VELOCITY_LOOKBACK:]
        
        spot_velocity = 0
        if len(self.price_history["INDEX"]) >= 2:
            price_series = pd.Series(self.price_history["INDEX"])
            spot_velocity = price_series.pct_change().iloc[-1] * 100
        
        # Spot divergence analysis
        divergence = self.analyze_spot_divergence(spot_price, spot_velocity, 
                                                 oi_velocity, gamma_profile)
        
        # Market regime synthesis
        market_regime = self._synthesize_market_regime(
            oi_regime, gamma_profile.regime, divergence
        )
        
        # Calculate Put/Call Ratio properly
        try:
            if not option_chain_df.empty:
                # Ensure we have proper filtering for PUT and CALL
                # First check what option types we have
                option_types = option_chain_df['option_type'].unique()
                print(f"DEBUG: Available option types: {option_types}")
                
                # Standardize option type strings
                option_chain_df['option_type_std'] = option_chain_df['option_type'].str.upper()
                
                # Calculate OI by type
                put_oi = option_chain_df[option_chain_df['option_type_std'].isin(['PUT', 'PE'])]['oi'].sum()
                call_oi = option_chain_df[option_chain_df['option_type_std'].isin(['CALL', 'CE'])]['oi'].sum()
                
                print(f"DEBUG: PUT OI total: {put_oi}")
                print(f"DEBUG: CALL OI total: {call_oi}")
                
                if call_oi > 0:
                    put_call_ratio = put_oi / call_oi
                elif put_oi > 0:
                    put_call_ratio = 99.99  # Extreme put dominance
                else:
                    put_call_ratio = 0.0  # No data
                
                print(f"DEBUG: Put/Call Ratio calculated: {put_call_ratio:.4f}")
            else:
                put_call_ratio = 0.0
        except Exception as e:
            print(f"DEBUG: Error calculating PCR: {e}")
            import traceback
            traceback.print_exc()
            put_call_ratio = 0.0
        
        # Compile comprehensive analysis
        analytics = {
            "timestamp": datetime.utcnow().isoformat(),
            "spot_price": spot_price,
            
            # OI Analysis
            "oi_velocity": round(oi_velocity, 3),
            "oi_regime": oi_regime,
            "total_oi": int(total_oi),
            "put_call_ratio": round(put_call_ratio, 4),  # Added PCR
            
            # Gamma Analysis
            "gamma_exposure": {
                "net_gamma": round(gamma_profile.net_gamma, 3) if gamma_profile else 0.0,
                "regime": gamma_profile.regime if gamma_profile else "UNKNOWN",
                "flip_levels": [round(x, 2) for x in gamma_profile.flip_levels] if gamma_profile else [],
                "max_impact_strike": gamma_profile.max_gamma_strike if gamma_profile else 0.0
            },
            
            # Structure Analysis
            "structural_walls": [
                {
                    "strike": wall.strike,
                    "type": wall.option_type,
                    "concentration": round(wall.oi_concentration, 3),
                    "defended": wall.is_defended,
                    "distance_pct": round(wall.distance_to_spot, 2)
                }
                for wall in walls
            ],
            
            # Trap Analysis
            "potential_traps": [
                {
                    "strike": trap.wall_strike,
                    "direction": trap.breach_direction,
                    "confidence": round(trap.confidence, 3),
                    "unwinding_rate": round(trap.oi_unwinding, 2)
                }
                for trap in traps
            ],
            
            # Divergence Analysis
            "spot_divergence": divergence,
            
            # Market Regime
            "market_regime": market_regime,
            "regime_confidence": self._calculate_regime_confidence(
                oi_velocity, gamma_profile.net_gamma if gamma_profile else 0.0, divergence
            )
        }
        
        # DEBUG: Print final analytics keys
        print(f"DEBUG: Analytics keys: {analytics.keys()}")
        print(f"DEBUG: PCR in analytics: {analytics.get('put_call_ratio', 'NOT FOUND')}")
        print(f"DEBUG: PCR type: {type(analytics.get('put_call_ratio'))}")
        
        # Store in cache for trend analysis
        self.walls_cache[datetime.utcnow()] = walls
        self.traps_cache[datetime.utcnow()] = traps
        self.gex_cache[datetime.utcnow()] = gamma_profile
        
        return {
            "raw_data": option_chain_df,
            "analytics": analytics,
            "market_insights": self._generate_market_insights(analytics)
        }
    
    # ==============================
    # ORIGINAL METHODS (ENHANCED)
    # ==============================
    
    def fetch_option_chain(self, option_keys: list[str]) -> pd.DataFrame:
        """
        Fetch option chain data for given option keys.
        """
        try:
            if not option_keys:
                return pd.DataFrame()
            
            # Fetch quotes in batches
            batch_size = 100
            all_quotes = []
            
            for i in range(0, len(option_keys), batch_size):
                batch = option_keys[i:i + batch_size]
                
                print(f"DEBUG: Fetching batch {i//batch_size + 1}, {len(batch)} keys")
                print(f"DEBUG: Sample keys: {batch[:3]}")
                
                quotes_data = self.fetch_quotes(batch)
                
                if quotes_data and 'data' in quotes_data:
                    batch_quotes = quotes_data['data']
                    
                    # Handle different response formats
                    if isinstance(batch_quotes, dict):
                        # The keys in the dict are the instrument identifiers
                        for instrument_id, quote_data in batch_quotes.items():
                            if isinstance(quote_data, dict):
                                # Add the instrument_id to the quote data
                                quote_data['instrument_id'] = instrument_id
                                all_quotes.append(quote_data)
                    elif isinstance(batch_quotes, list):
                        all_quotes.extend(batch_quotes)
                    else:
                        print(f"DEBUG: Unexpected batch_quotes type: {type(batch_quotes)}")
            
            if not all_quotes:
                print("DEBUG: No quotes returned from API")
                return pd.DataFrame()
            
            # Parse option chain data
            option_chain_data = []
            
            for quote in all_quotes:
                if not isinstance(quote, dict):
                    print(f"DEBUG: Skipping non-dict quote: {type(quote)}")
                    continue
                
                # Get instrument identifier
                instrument_id = quote.get('instrument_id', '')
                
                #print(f"DEBUG: Parsing instrument_id: '{instrument_id}'")
                
                if not instrument_id:
                    print(f"DEBUG: No instrument identifier found")
                    continue
                
                # Parse the instrument ID
                strike = 0.0
                option_type = ''
                
                # Format from logs: 'NSE_FO:NIFTY2620325300PE'
                # Let's parse this format
                if ':' in instrument_id:
                    parts = instrument_id.split(':')
                    if len(parts) >= 2:
                        full_code = parts[1]  # NIFTY2620325300PE
                        
                        # Check if it's a NIFTY option
                        if full_code.startswith('NIFTY'):
                            # Remove 'NIFTY' prefix
                            code = full_code[5:]  # 2620325300PE
                            
                            # Parse using simple logic
                            # Find option type first
                            if 'CE' in code:
                                option_type = 'CE'
                                # Remove 'CE' to get numbers only
                                numbers_part = code.replace('CE', '')
                            elif 'PE' in code:
                                option_type = 'PE'
                                numbers_part = code.replace('PE', '')
                            elif code.endswith('C'):
                                option_type = 'CE'
                                numbers_part = code[:-1]
                            elif code.endswith('P'):
                                option_type = 'PE'
                                numbers_part = code[:-1]
                            else:
                                print(f"DEBUG: Could not find option type in: {code}")
                                continue
                            
                            # Now extract strike from numbers_part
                            # numbers_part should be like: 2620325300
                            # The last 4-5 digits are the strike
                            
                            # Try different strike lengths
                            strike_candidates = []
                            
                            # Try 5-digit strike (e.g., 25300)
                            if len(numbers_part) >= 5:
                                candidate = numbers_part[-5:]
                                if candidate.isdigit():
                                    strike_candidates.append(float(candidate))
                            
                            # Try 4-digit strike (e.g., 5300)
                            if len(numbers_part) >= 4:
                                candidate = numbers_part[-4:]
                                if candidate.isdigit():
                                    strike_candidates.append(float(candidate))
                            
                            # Try 3-digit strike
                            if len(numbers_part) >= 3:
                                candidate = numbers_part[-3:]
                                if candidate.isdigit():
                                    strike_candidates.append(float(candidate))
                            
                            # Choose the most reasonable strike
                            if strike_candidates:
                                # For NIFTY, strikes are usually multiples of 50
                                # Prioritize strikes that are multiples of 50
                                valid_strikes = [s for s in strike_candidates if s % 50 == 0]
                                if valid_strikes:
                                    strike = max(valid_strikes)  # Take the largest valid strike
                                else:
                                    strike = max(strike_candidates)  # Fallback to largest
                            
                            if strike > 0:
                                print(f"DEBUG: Parsed - Strike: {strike}, Type: {option_type}")
                            else:
                                print(f"DEBUG: Could not parse strike from: {numbers_part}")
                
                # If still no strike/type, try symbol field
                if strike <= 0 or not option_type:
                    symbol = quote.get('symbol', '')
                    print(f"DEBUG: Trying symbol field: '{symbol}'")
                    
                    # Try to extract from symbol
                    if symbol:
                        import re
                        numbers = re.findall(r'\d+', symbol)
                        if numbers:
                            # Get the largest number (likely the strike)
                            strike_str = max(numbers, key=lambda x: (len(x), int(x)))
                            try:
                                strike = float(strike_str)
                            except:
                                strike = 0.0
                        
                        # Determine option type from symbol
                        if 'CE' in symbol.upper() or 'CALL' in symbol.upper():
                            option_type = 'CE'
                        elif 'PE' in symbol.upper() or 'PUT' in symbol.upper():
                            option_type = 'PE'
                        
                        if strike > 0 and option_type:
                            print(f"DEBUG: Symbol parse successful - Strike: {strike}, Type: {option_type}")
                
                if strike <= 0 or not option_type:
                    print(f"DEBUG: Could not parse strike/type from: {instrument_id}")
                    continue
                
                # Get OI value
                oi_value = quote.get('oi', 0)
                if oi_value is None:
                    oi_value = 0
                
                # Get last price
                last_price = quote.get('last_price', 0)
                if last_price is None:
                    last_price = 0
                
                option_chain_data.append({
                    'strike': strike,
                    'option_type': option_type,
                    'oi': float(oi_value),
                    'oi_change': 0.0,  # Not available in current response
                    'iv': float(quote.get('iv', 0) or 0),
                    'ltp': float(last_price),
                    'volume': int(quote.get('volume', 0) or 0),
                    'timestamp': quote.get('timestamp', '')
                })
            
            df = pd.DataFrame(option_chain_data)
            
            # DEBUG: Report findings
            print(f"DEBUG: Total options fetched: {len(df)}")
            if not df.empty:
                print(f"DEBUG: Option type distribution: {df['option_type'].value_counts().to_dict()}")
                print(f"DEBUG: Unique strikes: {len(df['strike'].unique())}")
                print(f"DEBUG: Min strike: {df['strike'].min()}")
                print(f"DEBUG: Max strike: {df['strike'].max()}")
                print(f"DEBUG: Total OI: {df['oi'].sum():.0f}")
                
                # Display first few rows
                if len(df) > 0:
                    print(f"DEBUG: First 3 rows:")
                    print(df[['strike', 'option_type', 'oi', 'ltp']].head(3).to_string())
            else:
                print("DEBUG: Option DataFrame is EMPTY")
            
            return df
            
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def parse_nifty_instrument_id(self, instrument_id: str):
        """
        Helper to parse NIFTY instrument IDs.
        Format: NSE_FO:NIFTY2620325300PE
        Returns: (strike, option_type)
        """
        try:
            if ':' not in instrument_id:
                return 0.0, ''
            
            parts = instrument_id.split(':')
            if len(parts) < 2:
                return 0.0, ''
            
            code = parts[1]  # NIFTY2620325300PE
            
            if not code.startswith('NIFTY'):
                return 0.0, ''
            
            # Remove 'NIFTY' prefix
            code = code[5:]  # 2620325300PE
            
            # Find option type
            option_type = ''
            if 'CE' in code:
                option_type = 'CE'
                code = code.replace('CE', '')
            elif 'PE' in code:
                option_type = 'PE'
                code = code.replace('PE', '')
            
            if not option_type:
                return 0.0, ''
            
            # Extract strike (last 4-5 digits)
            # Remove any non-digits
            import re
            numbers = re.findall(r'\d+', code)
            if not numbers:
                return 0.0, ''
            
            # Join all numbers
            all_digits = ''.join(numbers)
            
            # Try to extract strike
            strike = 0.0
            
            # Try different lengths from the end
            for length in [5, 4, 3]:
                if len(all_digits) >= length:
                    candidate = all_digits[-length:]
                    try:
                        strike_val = float(candidate)
                        # Check if it's a reasonable strike (multiple of 50 for NIFTY)
                        if strike_val >= 1000 and strike_val <= 50000:  # Reasonable range
                            if strike_val % 50 == 0:  # NIFTY strikes are multiples of 50
                                strike = strike_val
                                break
                    except:
                        continue
            
            return strike, option_type
            
        except Exception as e:
            print(f"DEBUG: Error in parse_nifty_instrument_id: {e}")
            return 0.0, ''
    
    def parse_nifty_option_code(self, option_code: str) -> Tuple[float, str]:
        """
        Parse NIFTY option code like 'NIFTY2620325300PE' to (strike, option_type)
        
        Format: NIFTY + YY + MM + DD + STRIKE + TYPE
        Example: NIFTY2620325300PE = NIFTY + 26 + 20 + 3 + 25300 + PE
        """
        try:
            # Remove 'NIFTY' prefix
            if option_code.startswith('NIFTY'):
                code = option_code[5:]  # Remove 'NIFTY'
            else:
                code = option_code
            
            # Extract option type (last 2 chars)
            if code.endswith('CE'):
                option_type = 'CE'
                code = code[:-2]
            elif code.endswith('PE'):
                option_type = 'PE'
                code = code[:-2]
            else:
                # Try single char type
                if code.endswith('C'):
                    option_type = 'CE'
                    code = code[:-1]
                elif code.endswith('P'):
                    option_type = 'PE'
                    code = code[:-1]
                else:
                    return 0.0, ''
            
            # The remaining code should be: YYMMDD + STRIKE
            # For NIFTY, strike is usually the last 5 digits
            if len(code) >= 5:
                # Try to extract strike (last 5 digits)
                strike_str = code[-5:]
                
                # Convert to float
                strike = float(strike_str)
                
                # Validate strike (NIFTY strikes are multiples of 50)
                if strike > 0 and strike % 50 == 0:
                    return strike, option_type
            
            # Alternative: try to find numbers
            import re
            numbers = re.findall(r'\d+', code)
            if numbers:
                # Take the largest number as strike
                max_num = max(numbers, key=len)
                if len(max_num) >= 4:  # Reasonable strike length
                    strike = float(max_num)
                    return strike, option_type
            
            return 0.0, ''
            
        except Exception as e:
            print(f"DEBUG: Error parsing option code {option_code}: {e}")
            return 0.0, ''
        
    def fetch_index_quote(self, symbol: str = "NSE_INDEX|Nifty 50") -> Optional[dict]:
        """Original method - fetch index quote."""
        try:
            data = self._make_request(
                "GET",
                "market-quote/quotes",
                params={"symbol": symbol}
            )
            
            if data.get("status") == "success":
                response_key = symbol.replace("|", ":")
                quote_data = data.get("data", {}).get(response_key)
                
                if not quote_data:
                    quote_data = data.get("data", {}).get(symbol)
                
                if not quote_data and data.get("data"):
                    first_key = list(data["data"].keys())[0]
                    quote_data = data["data"][first_key]
                
                if not quote_data:
                    return None
                
                ohlc = quote_data.get("ohlc", {})
                
                return {
                    "symbol": symbol,
                    "ltp": quote_data.get("last_price"),
                    "open": ohlc.get("open"),
                    "high": ohlc.get("high"),
                    "low": ohlc.get("low"),
                    "close": ohlc.get("close"),
                    "change": quote_data.get("change"),
                    "net_change": quote_data.get("net_change"),
                    "percent_change": quote_data.get("percent_change"),
                    "volume": quote_data.get("volume"),
                    "timestamp": quote_data.get("timestamp")
                }
            else:
                print(f"API Error: {data}")
                return None
                
        except Exception as e:
            print(f"Error in fetch_index_quote: {e}")
            return None
    
    
    
    
    
    
    def fetch_quotes(self, instrument_keys: list[str]) -> Optional[dict]:
        """
        Fetch quotes for multiple instrument keys.
        Used by fetch_option_chain.
        """
        try:
            if not instrument_keys:
                return None
            
            # Join instrument keys with comma
            instrument_key_str = ",".join(instrument_keys)
            
            data = self._make_request(
                "GET",
                "market-quote/quotes",
                params={"instrument_key": instrument_key_str}
            )
            
            # FIX: Debug the response structure
            print(f"DEBUG fetch_quotes: Response keys: {data.keys() if isinstance(data, dict) else 'Not dict'}")
            if isinstance(data, dict) and 'data' in data:
                print(f"DEBUG fetch_quotes: Data type: {type(data['data'])}")
                if isinstance(data['data'], dict):
                    print(f"DEBUG fetch_quotes: First 3 keys in data: {list(data['data'].keys())[:3] if data['data'] else 'Empty'}")
            
            return data
            
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            return None

    def fetch_equity_quotes(self, symbols: list[str]) -> pd.DataFrame:
        """
        Fetch equity quotes from Upstox.
        Accepts BOTH:
        - SYMBOL (e.g. HDFCBANK)
        - NSE_EQ|SYMBOL

        Internally normalizes to NSE_EQ|SYMBOL (Upstox requirement).
        NEVER raises validation errors for symbol format.
        """

        # ------------------------------
        # DEBUG: Log input
        # ------------------------------
        print(f"DEBUG Equity Quotes: Starting fetch with {len(symbols)} symbols")
        print(f"DEBUG Equity Quotes: First 3 symbols: {symbols[:3] if symbols else 'EMPTY'}")

        # ------------------------------
        # EMPTY GUARD
        # ------------------------------
        if not symbols:
            print("DEBUG Equity Quotes: No symbols provided")
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )

        # ------------------------------
        # 🔑 NORMALIZE SYMBOLS (CRITICAL FIX)
        # ------------------------------
        normalized_symbols: list[str] = []
        symbol_mapping = {}  # Track original -> normalized mapping

        for s in symbols:
            if not isinstance(s, str):
                print(f"DEBUG Equity Quotes: Skipping non-string symbol: {s}")
                continue

            s = s.strip().upper()
            
            # DEBUG: Track original
            original = s
            
            # Convert SYMBOL → NSE_EQ|SYMBOL
            if "|" not in s:
                normalized = f"NSE_EQ:{s}"
                normalized_symbols.append(normalized)
                symbol_mapping[normalized] = s  # Store mapping
            else:
                # Already formatted → enforce correct prefix
                parts = s.split("|", 1)
                if len(parts) == 2:
                    prefix, sym = parts
                    if prefix != "NSE_EQ":
                        normalized = f"NSE_EQ|{sym}"
                        normalized_symbols.append(normalized)
                        symbol_mapping[normalized] = sym  # Store mapping
                    else:
                        normalized_symbols.append(s)
                        symbol_mapping[s] = sym  # Store mapping
                else:
                    print(f"DEBUG Equity Quotes: Invalid format for {s}, skipping")
                    continue
            
            # DEBUG: Show mapping
            print(f"DEBUG Equity Quotes: Original: '{original}' → Normalized: '{normalized_symbols[-1]}'")

        if not normalized_symbols:
            print("DEBUG Equity Quotes: No valid symbols after normalization")
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )

        # Upstox limit safeguard
        if len(normalized_symbols) > 200:
            print(f"DEBUG Equity Quotes: Trimming symbols from {len(normalized_symbols)} to 200")
            normalized_symbols = normalized_symbols[:200]

        print(f"DEBUG Equity Quotes: Final normalized symbols: {len(normalized_symbols)}")
        print(f"DEBUG Equity Quotes: Sample normalized: {normalized_symbols[:3]}")

        # ------------------------------
        # API CALL
        # ------------------------------
        try:
            # Batch symbols for Upstox API (max 50 per request)
            batch_size = 50
            all_data = {}
            
            for i in range(0, len(normalized_symbols), batch_size):
                batch = normalized_symbols[i:i + batch_size]
                batch_str = ",".join(batch)
                
                print(f"DEBUG Equity Quotes: Batch {i//batch_size + 1}: {len(batch)} symbols")
                print(f"DEBUG Equity Quotes: Batch sample: {batch[:2]}")
                # In fetch_equity_quotes method, after the API call:
                print(f"DEBUG Equity Quotes: API response status: {data.get('status') if data else 'No response'}")
                if data and 'data' in data:
                    print(f"DEBUG Equity Quotes: Number of symbols returned: {len(data['data'])}")
                    # Print first few items to see structure
                    for i, (key, value) in enumerate(data['data'].items()):
                        if i < 3:
                            print(f"DEBUG Equity Quotes: Key {i}: {key}, Value type: {type(value)}")
                            if isinstance(value, dict):
                                print(f"DEBUG Equity Quotes:   Last price: {value.get('last_price')}")
                                print(f"DEBUG Equity Quotes:   OI: {value.get('oi')}")

                data = self._make_request(
                    "GET",
                    "market-quote/quotes",
                    params={"instrument_key": batch_str}
                )
                
                if data and 'data' in data:
                    print(f"DEBUG Equity Quotes: Batch response keys: {list(data['data'].keys())[:3] if data['data'] else 'EMPTY'}")
                    all_data.update(data['data'])
                else:
                    print(f"DEBUG Equity Quotes: Batch {i//batch_size + 1} returned no data or error")
            
            print(f"DEBUG Equity Quotes: Total keys in response: {len(all_data)}")
            
            if not all_data:
                print("DEBUG Equity Quotes: No data in any batch response")
                return pd.DataFrame(
                    columns=[
                        "symbol", "ltp", "open", "high", "low",
                        "close", "change", "percent_change",
                        "volume", "timestamp"
                    ]
                )

        except Exception as e:
            print(f"DEBUG Equity Quotes: API call error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )

        # ------------------------------
        # PARSE RESPONSE
        # ------------------------------
        rows = []
        success_count = 0
        error_count = 0

        for key, payload in all_data.items():
            try:
                # DEBUG: Check payload structure
                if not payload:
                    print(f"DEBUG Equity Quotes: Empty payload for key: {key}")
                    error_count += 1
                    continue
                
                # key format: NSE_EQ|HDFCBANK
                if "|" in key:
                    symbol = key.split("|", 1)[1]
                else:
                    symbol = key
                
                # Get original symbol for mapping (if available)
                display_symbol = symbol_mapping.get(key, symbol)
                
                # Extract data
                ohlc = payload.get("ohlc", {}) or {}
                
                # DEBUG: Show data being extracted
                ltp = payload.get("last_price")
                print(f"DEBUG Equity Quotes: Parsing {display_symbol} → LTP: {ltp}")
                
                rows.append({
                    "symbol": display_symbol,  # Use mapped/cleaned symbol
                    "ltp": float(ltp) if ltp is not None else 0.0,
                    "open": float(ohlc.get("open", 0)) if ohlc.get("open") is not None else 0.0,
                    "high": float(ohlc.get("high", 0)) if ohlc.get("high") is not None else 0.0,
                    "low": float(ohlc.get("low", 0)) if ohlc.get("low") is not None else 0.0,
                    "close": float(ohlc.get("close", 0)) if ohlc.get("close") is not None else 0.0,
                    "change": float(payload.get("change", 0)) if payload.get("change") is not None else 0.0,
                    "percent_change": float(payload.get("percent_change", 0)) if payload.get("percent_change") is not None else 0.0,
                    "volume": int(payload.get("volume", 0)) if payload.get("volume") is not None else 0,
                    "timestamp": payload.get("timestamp", "")
                })
                success_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"⚠️ DEBUG Equity Quotes: Failed to parse equity quote {key}: {e}")
                import traceback
                traceback.print_exc()

        print(f"DEBUG Equity Quotes: Parsed {success_count} successful, {error_count} errors")
        
        if not rows:
            print("DEBUG Equity Quotes: No rows parsed successfully")
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )
        
        df = pd.DataFrame(rows)
        print(f"DEBUG Equity Quotes: Returning dataframe with shape: {df.shape}")
        print(f"DEBUG Equity Quotes: Columns: {df.columns.tolist()}")
        print(f"DEBUG Equity Quotes: Sample data (first 3):")
        if not df.empty:
            print(df.head(3))
        
        return df

    def fetch_equity_quote(self, symbol: str) -> Optional[dict]:
        """
        Fetch single equity quote.
        """
        try:
            # Normalize symbol
            if "|" not in symbol:
                symbol = f"NSE_EQ|{symbol.upper()}"
            else:
                parts = symbol.split("|")
                if len(parts) == 2 and parts[0] != "NSE_EQ":
                    symbol = f"NSE_EQ|{parts[1]}"
            
            data = self._make_request(
                "GET",
                "market-quote/quotes",
                params={"instrument_key": symbol}
            )
            
            if data.get("status") == "success":
                quote_data = data.get("data", {}).get(symbol)
                if not quote_data:
                    return None
                
                ohlc = quote_data.get("ohlc", {})
                
                # Extract symbol without prefix
                display_symbol = symbol.split("|")[-1] if "|" in symbol else symbol
                
                return {
                    "symbol": display_symbol,
                    "ltp": quote_data.get("last_price"),
                    "open": ohlc.get("open"),
                    "high": ohlc.get("high"),
                    "low": ohlc.get("low"),
                    "close": ohlc.get("close"),
                    "change": quote_data.get("change"),
                    "percent_change": quote_data.get("percent_change"),
                    "volume": quote_data.get("volume"),
                    "timestamp": quote_data.get("timestamp")
                }
            else:
                print(f"API Error for {symbol}: {data}")
                return None
                
        except Exception as e:
            print(f"Error fetching equity quote for {symbol}: {e}")
            return None

    def fetch_profile(self) -> Optional[dict]:
        """
        Fetch user profile to test authentication.
        """
        try:
            data = self._make_request("GET", "user/profile")
            return data.get("data", {})
        except Exception as e:
            print(f"Error fetching profile: {e}")
            return None
    
    def fetch_equity_quotes(self, symbols: list[str]) -> pd.DataFrame:
        """
        Fetch equity quotes from Upstox.
        Accepts BOTH:
        - SYMBOL (e.g. HDFCBANK)
        - NSE_EQ|SYMBOL

        Internally normalizes to NSE_EQ|SYMBOL (Upstox requirement).
        NEVER raises validation errors for symbol format.
        """

        # ------------------------------
        # DEBUG: Log input
        # ------------------------------
        print(f"DEBUG Equity Quotes: Starting fetch with {len(symbols)} symbols")
        print(f"DEBUG Equity Quotes: First 3 symbols: {symbols[:3] if symbols else 'EMPTY'}")

        # ------------------------------
        # EMPTY GUARD
        # ------------------------------
        if not symbols:
            print("DEBUG Equity Quotes: No symbols provided")
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )

        # ------------------------------
        # 🔑 NORMALIZE SYMBOLS (CRITICAL FIX)
        # ------------------------------
        normalized_symbols: list[str] = []
        symbol_mapping = {}  # Track original -> normalized mapping

        for s in symbols:
            if not isinstance(s, str):
                print(f"DEBUG Equity Quotes: Skipping non-string symbol: {s}")
                continue

            s = s.strip().upper()
            
            # DEBUG: Track original
            original = s
            
            # Convert SYMBOL → NSE_EQ|SYMBOL
            if "|" not in s:
                normalized = f"NSE_EQ|{s}"
                normalized_symbols.append(normalized)
                symbol_mapping[normalized] = s  # Store mapping
            else:
                # Already formatted → enforce correct prefix
                parts = s.split("|", 1)
                if len(parts) == 2:
                    prefix, sym = parts
                    if prefix != "NSE_EQ":
                        normalized = f"NSE_EQ|{sym}"
                        normalized_symbols.append(normalized)
                        symbol_mapping[normalized] = sym  # Store mapping
                    else:
                        normalized_symbols.append(s)
                        symbol_mapping[s] = sym  # Store mapping
                else:
                    print(f"DEBUG Equity Quotes: Invalid format for {s}, skipping")
                    continue
            
            # DEBUG: Show mapping
            print(f"DEBUG Equity Quotes: Original: '{original}' → Normalized: '{normalized_symbols[-1]}'")

        if not normalized_symbols:
            print("DEBUG Equity Quotes: No valid symbols after normalization")
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )

        # Upstox limit safeguard
        if len(normalized_symbols) > 200:
            print(f"DEBUG Equity Quotes: Trimming symbols from {len(normalized_symbols)} to 200")
            normalized_symbols = normalized_symbols[:200]

        print(f"DEBUG Equity Quotes: Final normalized symbols: {len(normalized_symbols)}")
        print(f"DEBUG Equity Quotes: Sample normalized: {normalized_symbols[:3]}")

        # ------------------------------
        # API CALL
        # ------------------------------
        try:
            # Batch symbols for Upstox API (max 50 per request)
            batch_size = 50
            all_data = {}
            
            for i in range(0, len(normalized_symbols), batch_size):
                batch = normalized_symbols[i:i + batch_size]
                batch_str = ",".join(batch)
                
                print(f"DEBUG Equity Quotes: Batch {i//batch_size + 1}: {len(batch)} symbols")
                print(f"DEBUG Equity Quotes: Batch sample: {batch[:2]}")
                
                data = self._make_request(
                    "GET",
                    "market-quote/quotes",
                    params={"instrument_key": batch_str}
                )
                
                if data and 'data' in data:
                    print(f"DEBUG Equity Quotes: Batch response keys: {list(data['data'].keys())[:3] if data['data'] else 'EMPTY'}")
                    all_data.update(data['data'])
                else:
                    print(f"DEBUG Equity Quotes: Batch {i//batch_size + 1} returned no data or error")
            
            print(f"DEBUG Equity Quotes: Total keys in response: {len(all_data)}")
            
            if not all_data:
                print("DEBUG Equity Quotes: No data in any batch response")
                return pd.DataFrame(
                    columns=[
                        "symbol", "ltp", "open", "high", "low",
                        "close", "change", "percent_change",
                        "volume", "timestamp"
                    ]
                )

        except Exception as e:
            print(f"DEBUG Equity Quotes: API call error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )

        # ------------------------------
        # PARSE RESPONSE
        # ------------------------------
        rows = []
        success_count = 0
        error_count = 0

        for key, payload in all_data.items():
            try:
                # DEBUG: Check payload structure
                if not payload:
                    print(f"DEBUG Equity Quotes: Empty payload for key: {key}")
                    error_count += 1
                    continue
                
                # key format: NSE_EQ|HDFCBANK
                if "|" in key:
                    symbol = key.split("|", 1)[1]
                else:
                    symbol = key
                
                # Get original symbol for mapping (if available)
                display_symbol = symbol_mapping.get(key, symbol)
                
                # Extract data
                ohlc = payload.get("ohlc", {}) or {}
                
                # DEBUG: Show data being extracted
                ltp = payload.get("last_price")
                print(f"DEBUG Equity Quotes: Parsing {display_symbol} → LTP: {ltp}")
                
                rows.append({
                    "symbol": display_symbol,  # Use mapped/cleaned symbol
                    "ltp": float(ltp) if ltp is not None else 0.0,
                    "open": float(ohlc.get("open", 0)) if ohlc.get("open") is not None else 0.0,
                    "high": float(ohlc.get("high", 0)) if ohlc.get("high") is not None else 0.0,
                    "low": float(ohlc.get("low", 0)) if ohlc.get("low") is not None else 0.0,
                    "close": float(ohlc.get("close", 0)) if ohlc.get("close") is not None else 0.0,
                    "change": float(payload.get("change", 0)) if payload.get("change") is not None else 0.0,
                    "percent_change": float(payload.get("percent_change", 0)) if payload.get("percent_change") is not None else 0.0,
                    "volume": int(payload.get("volume", 0)) if payload.get("volume") is not None else 0,
                    "timestamp": payload.get("timestamp", "")
                })
                success_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"⚠️ DEBUG Equity Quotes: Failed to parse equity quote {key}: {e}")
                import traceback
                traceback.print_exc()

        print(f"DEBUG Equity Quotes: Parsed {success_count} successful, {error_count} errors")
        
        if not rows:
            print("DEBUG Equity Quotes: No rows parsed successfully")
            return pd.DataFrame(
                columns=[
                    "symbol", "ltp", "open", "high", "low",
                    "close", "change", "percent_change",
                    "volume", "timestamp"
                ]
            )
        
        df = pd.DataFrame(rows)
        print(f"DEBUG Equity Quotes: Returning dataframe with shape: {df.shape}")
        print(f"DEBUG Equity Quotes: Columns: {df.columns.tolist()}")
        print(f"DEBUG Equity Quotes: Sample data (first 3):")
        if not df.empty:
            print(df.head(3))
        
        return df


    
    # ==============================
    # HELPER METHODS
    # ==============================
    
    def _synthesize_market_regime(self, oi_regime: str, gamma_regime: str, 
                                 divergence: Dict) -> str:
        """Synthesize overall market regime from multiple indicators."""
        regimes = []
        
        # OI-based regime
        if oi_regime == "EXPANSIVE":
            regimes.append("CAPITAL_INFLOW")
        elif oi_regime == "CONSTRICTED":
            regimes.append("CAPITAL_OUTFLOW")
        
        # Gamma-based regime
        if "POSITIVE" in gamma_regime:
            regimes.append("STABILIZING")
        elif "NEGATIVE" in gamma_regime:
            regimes.append("ACCELERATING")
        
        # Divergence-based
        if divergence["has_divergence"]:
            regimes.append(f"DIVERGENCE_{divergence['type']}")
        
        if not regimes:
            return "NEUTRAL"
        
        # Combine regimes
        if len(regimes) == 1:
            return regimes[0]
        else:
            return f"{regimes[0]}_{regimes[1]}"
    
    def _calculate_regime_confidence(self, oi_velocity: float, 
                                    net_gamma: float, divergence: Dict) -> float:
        """Calculate confidence score for market regime."""
        confidence = 0.5  # Base confidence
        
        # OI velocity confidence
        oi_confidence = min(abs(oi_velocity) / 3, 1.0)
        confidence = 0.7 * confidence + 0.3 * oi_confidence
        
        # Gamma confidence
        gamma_confidence = min(abs(net_gamma) / 1000, 1.0)  # Scale appropriately
        confidence = 0.7 * confidence + 0.3 * gamma_confidence
        
        # Divergence confidence
        if divergence["has_divergence"]:
            div_confidence = divergence["confidence"]
            confidence = 0.8 * confidence + 0.2 * div_confidence
        
        return round(confidence, 3)
    
    def _generate_market_insights(self, analytics: Dict) -> List[str]:
        """Generate human-readable market insights."""
        insights = []
        
        # OI insights
        oi_vel = analytics["oi_velocity"]
        if oi_vel > 1.5:
            insights.append("📈 Strong OI buildup - New capital entering market")
        elif oi_vel < -1.5:
            insights.append("📉 OI unwinding - Positions being closed, watch for reversals")
        
        # Gamma insights
        gamma_regime = analytics["gamma_exposure"]["regime"]
        if gamma_regime == "GAMMA_POSITIVE":
            insights.append("📌 Positive Gamma - Market likely to pin/stabilize")
        elif gamma_regime == "GAMMA_NEGATIVE":
            insights.append("🚀 Negative Gamma - Accelerating moves possible, watch for squeezes")
        
        # Wall insights
        walls = analytics["structural_walls"]
        if walls:
            top_wall = walls[0]
            if top_wall["defended"]:
                insights.append(f"🛡️ Strong wall at {top_wall['strike']} ({top_wall['type']}) - Being defended")
            else:
                insights.append(f"⚠️ Wall at {top_wall['strike']} weakening - Monitor for breach")
        
        # Trap insights
        traps = analytics["potential_traps"]
        if traps:
            trap = traps[0]
            insights.append(f"🎯 Potential {trap['direction']} trap at {trap['strike']} - Confidence: {trap['confidence']*100:.0f}%")
        
        # Divergence insights
        if analytics["spot_divergence"]["has_divergence"]:
            div_type = analytics["spot_divergence"]["type"]
            if div_type == "BULLISH":
                insights.append("🔍 Bullish divergence detected - Price down but smart money accumulating")
            else:
                insights.append("🔍 Bearish divergence detected - Price up but internal weakness")
        
        return insights
    
    def get_market_analytics_summary(self) -> Dict:
        """Get summary of all market analytics."""
        return {
            "oi_velocity_history": self.velocity_history,
            "gamma_history": self.gex_cache,
            "walls_history": self.walls_cache,
            "traps_history": self.traps_cache,
            "current_regime": getattr(self, 'current_regime', 'UNKNOWN'),
            "last_update": datetime.utcnow().isoformat()
        }


# ==============================
# MARKET ANALYTICS HELPER CLASS
# ==============================

class MarketAnalytics:
    """Helper class for advanced market analytics."""
    
    @staticmethod
    def calculate_put_call_ratio(option_chain_df: pd.DataFrame) -> Dict:
        """Calculate Put-Call Ratio with breakdown."""
        put_oi = option_chain_df[option_chain_df["option_type"] == "PE"]["oi"].sum()
        call_oi = option_chain_df[option_chain_df["option_type"] == "CE"]["oi"].sum()
        put_volume = option_chain_df[option_chain_df["option_type"] == "PE"]["volume"].sum()
        call_volume = option_chain_df[option_chain_df["option_type"] == "CE"]["volume"].sum()
        
        pcr_oi = put_oi / call_oi if call_oi > 0 else 0
        pcr_volume = put_volume / call_volume if call_volume > 0 else 0
        
        return {
            "pcr_oi": round(pcr_oi, 3),
            "pcr_volume": round(pcr_volume, 3),
            "sentiment": "BEARISH" if pcr_oi > 1.2 else "BULLISH" if pcr_oi < 0.8 else "NEUTRAL",
            "put_oi": int(put_oi),
            "call_oi": int(call_oi),
            "total_oi": int(put_oi + call_oi)
        }
    
    @staticmethod
    def detect_max_pain(option_chain_df: pd.DataFrame) -> Dict:
        """Calculate Max Pain strike."""
        if option_chain_df.empty:
            return {"max_pain_strike": 0, "total_pain": 0}
        
        strikes = sorted(option_chain_df['strike'].unique())
        pain_values = []
        
        for strike in strikes:
            total_pain = 0
            
            # Calculate pain from puts
            puts = option_chain_df[(option_chain_df['strike'] == strike) & 
                                  (option_chain_df['option_type'] == 'PE')]
            for _, put in puts.iterrows():
                if strike < put['strike']:  # ITM
                    total_pain += (put['strike'] - strike) * put['oi']
            
            # Calculate pain from calls
            calls = option_chain_df[(option_chain_df['strike'] == strike) & 
                                   (option_chain_df['option_type'] == 'CE')]
            for _, call in calls.iterrows():
                if strike > call['strike']:  # ITM
                    total_pain += (strike - call['strike']) * call['oi']
            
            pain_values.append((strike, total_pain))
        
        if pain_values:
            max_pain_strike, min_pain = min(pain_values, key=lambda x: x[1])
            return {
                "max_pain_strike": max_pain_strike,
                "total_pain": int(min_pain),
                "all_pain_levels": pain_values[:10]  # Top 10
            }
        
        return {"max_pain_strike": 0, "total_pain": 0}


# ==============================
# UTILITY FUNCTIONS
# ==============================

def display_market_analytics(analytics: Dict):
    """Display market analytics in Streamlit."""
    if not analytics:
        st.info("No analytics available yet")
        return
    
    st.markdown("### 📊 Advanced Market Analytics")
    
    # OI Velocity
    col1, col2, col3 = st.columns(3)
    with col1:
        oi_vel = analytics.get("oi_velocity", 0)
        regime = analytics.get("oi_regime", "N/A")
        color = "green" if oi_vel > 0 else "red"
        st.metric("OI Velocity", f"{oi_vel:.2f}σ", regime, delta_color="off")
    
    # Gamma Exposure
    with col2:
        gamma = analytics.get("gamma_exposure", {}).get("net_gamma", 0)
        gamma_regime = analytics.get("gamma_exposure", {}).get("regime", "N/A")
        icon = "📌" if "POSITIVE" in gamma_regime else "🚀" if "NEGATIVE" in gamma_regime else "⚖️"
        st.metric("Gamma Exposure", f"{icon} {gamma_regime}", f"{gamma:.2f}")
    
    # Market Regime
    with col3:
        regime = analytics.get("market_regime", "N/A")
        confidence = analytics.get("regime_confidence", 0) * 100
        st.metric("Market Regime", regime, f"{confidence:.0f}% confidence")
    
    # Insights
    insights = analytics.get("market_insights", [])
    if insights:
        st.markdown("#### 💡 Market Insights")
        for insight in insights[:5]:  # Top 5 insights
            st.info(insight)
    
    # Structural Walls
    walls = analytics.get("structural_walls", [])
    if walls:
        st.markdown("#### 🧱 Structural Walls")
        wall_df = pd.DataFrame(walls)
        st.dataframe(wall_df, use_container_width=True)
    
    # Traps
    traps = analytics.get("potential_traps", [])
    if traps:
        st.markdown("#### 🎯 Potential Traps")
        for trap in traps:
            direction = trap.get("direction", "")
            strike = trap.get("strike", 0)
            confidence = trap.get("confidence", 0) * 100
            st.warning(f"{direction} trap at {strike} ({confidence:.0f}% confidence)")