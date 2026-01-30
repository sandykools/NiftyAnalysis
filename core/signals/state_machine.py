"""
Enhanced Signal State Machine with Research-Based Decision Making.
Implements advanced signal generation using:
1. OI Velocity & Gamma Exposure analysis
2. Structural Walls & Traps detection
3. Spot Divergence analysis
4. Wyckoff pattern recognition
5. Market regime awareness
"""

from datetime import datetime, timedelta
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import json

# Import research features
from ml.feature_contract import FEATURE_VERSION, RESEARCH_FEATURES

# ==============================
# ENUMS & DATA CLASSES
# ==============================

class SignalStrength(Enum):
    """Signal strength classification"""
    WEAK = "WEAK"        # Low confidence, minor edge
    MODERATE = "MODERATE" # Decent confidence, clear edge
    STRONG = "STRONG"    # High confidence, strong edge
    VERY_STRONG = "VERY_STRONG" # Very high confidence, major edge

class MarketRegime(Enum):
    """Market regime classification"""
    ACCUMULATION = "ACCUMULATION"      # Wyckoff accumulation
    DISTRIBUTION = "DISTRIBUTION"      # Wyckoff distribution
    UPTREND = "UPTREND"                # Strong uptrend
    DOWNTREND = "DOWNTREND"            # Strong downtrend
    RANGING = "RANGING"                # Range-bound
    BREAKOUT = "BREAKOUT"              # Breakout from range
    SQUEEZE = "SQUEEZE"                # Gamma/OI squeeze
    REVERSAL = "REVERSAL"              # Trend reversal

class TrapType(Enum):
    """Types of market traps"""
    GAMMA_TRAP = "GAMMA_TRAP"          # Gamma-induced squeeze
    OI_TRAP = "OI_TRAP"                # OI unwinding trap
    WYCKOFF_SPRING = "WYCKOFF_SPRING"  # Wyckoff spring (bear trap)
    WYCKOFF_UPTHRUST = "WYCKOFF_UPTHRUST" # Wyckoff upthrust (bull trap)
    DIVERGENCE_TRAP = "DIVERGENCE_TRAP" # Divergence-based trap

@dataclass
class SignalComponents:
    """Components of signal decision"""
    # Core components
    trend_score: float                # -1 to 1 (bearish to bullish)
    momentum_score: float             # -1 to 1 (weak to strong)
    
    # Research components
    oi_velocity_score: float          # -1 to 1 (negative to positive velocity)
    gamma_score: float               # -1 to 1 (negative to positive gamma)
    wall_interaction_score: float    # -1 to 1 (wall defense vs trap)
    divergence_score: float          # -1 to 1 (bearish to bullish divergence)
    
    # Wyckoff components
    wyckoff_phase_score: float       # -1 to 1 (distribution to accumulation)
    pattern_score: float             # -1 to 1 (bearish to bullish patterns)
    
    # Composite scores
    composite_score: float           # Overall score (-1 to 1)
    confidence: float               # Signal confidence (0-1)
    
    # Regime classification
    market_regime: MarketRegime
    regime_confidence: float

@dataclass
class TrapAnalysis:
    """Analysis of potential trap"""
    trap_type: TrapType
    strike_level: float
    direction: str  # "BULLISH" or "BEARISH"
    confidence: float
    trigger_conditions: List[str]
    expected_move_pct: float

# ==============================
# ENHANCED SIGNAL STATE MACHINE
# ==============================

class SignalStateMachine:
    """
    Advanced signal generator with research-based decision making.
    Incorporates OI velocity, gamma exposure, walls/traps, divergence.
    """
    
    def __init__(
        self,
        signal_expiry_minutes: int = 5,
        confidence_threshold: float = 0.2,
        trap_confidence_threshold: float = 0.6
    ):
        self.signal_expiry_minutes = signal_expiry_minutes
        self.confidence_threshold = confidence_threshold
        self.trap_confidence_threshold = trap_confidence_threshold
        
        # State tracking
        self.signal_history = []
        self.regime_history = []
        self.trap_detections = []
        
        # Thresholds (configurable)
        self.thresholds = {
            "oi_velocity_high": 1.5,      # σ
            "oi_velocity_low": -1.5,      # σ
            "gamma_high": 500,           # Net gamma threshold
            "gamma_low": -500,           # Net gamma threshold
            "trap_prob_high": 0.7,       # High trap probability
            "divergence_high": 0.5,      # Significant divergence
            "wall_strength_high": 0.3,   # Strong wall
            "spring_detection_high": 0.6, # Strong spring pattern
            "upthrust_detection_high": 0.6 # Strong upthrust pattern
        }
    
    # ==============================
    # RESEARCH FEATURE EXTRACTION
    # ==============================
    
    def extract_research_features(self, feature_row: pd.Series) -> Dict:
        """
        Extract and normalize research features from feature row.
        """
        feats = {}
        
        # OI Velocity features
        feats["oi_velocity"] = feature_row.get("oi_velocity", 0.0)
        feats["oi_regime_expansive"] = feature_row.get("oi_regime_expansive", 0.0)
        feats["oi_regime_constricted"] = feature_row.get("oi_regime_constricted", 0.0)
        
        # Gamma Exposure features
        feats["net_gamma"] = feature_row.get("net_gamma", 0.0)
        feats["gamma_regime_positive"] = feature_row.get("gamma_regime_positive", 0.0)
        feats["gamma_regime_negative"] = feature_row.get("gamma_regime_negative", 0.0)
        feats["gamma_flip_distance"] = feature_row.get("gamma_flip_distance", 0.0)
        
        # Structural features
        feats["wall_strength"] = feature_row.get("wall_strength", 0.0)
        feats["wall_defense_score"] = feature_row.get("wall_defense_score", 0.0)
        feats["trap_probability"] = feature_row.get("trap_probability", 0.0)
        
        # Divergence features
        feats["price_oi_divergence"] = feature_row.get("price_oi_divergence", 0.0)
        feats["price_gamma_divergence"] = feature_row.get("price_gamma_divergence", 0.0)
        feats["divergence_score"] = feature_row.get("divergence_score", 0.0)
        feats["has_divergence"] = feature_row.get("has_divergence", 0.0)
        
        # Wyckoff features
        feats["spring_detection"] = feature_row.get("spring_detection", 0.0)
        feats["upthrust_detection"] = feature_row.get("upthrust_detection", 0.0)
        feats["accumulation_score"] = feature_row.get("accumulation_score", 0.0)
        
        # Composite features
        feats["gamma_wall_interaction"] = feature_row.get("gamma_wall_interaction", 0.0)
        feats["velocity_divergence_composite"] = feature_row.get("velocity_divergence_composite", 0.0)
        feats["trap_gamma_composite"] = feature_row.get("trap_gamma_composite", 0.0)
        
        # Base features (still important)
        feats["put_call_ratio"] = feature_row.get("put_call_ratio", 1.0)
        feats["price_momentum"] = feature_row.get("price_momentum", 0.0)
        feats["ccc_slope"] = feature_row.get("ccc_slope", 0.0)
        feats["vwap_distance"] = feature_row.get("vwap_distance", 0.0)
        
        return feats
    
    # ==============================
    # COMPONENT SCORING
    # ==============================
    
    def score_oi_velocity(self, feats: Dict) -> Tuple[float, str]:
        """
        Score OI velocity component.
        
        Returns:
            score (-1 to 1), analysis
        """
        velocity = feats.get("oi_velocity", 0.0)
        expansive = feats.get("oi_regime_expansive", 0.0)
        constricted = feats.get("oi_regime_constricted", 0.0)
        
        # Velocity scoring
        if velocity > self.thresholds["oi_velocity_high"]:
            score = 1.0  # Strong buildup
            analysis = "Strong OI buildup - initiative capital entering"
        elif velocity > 0.5:
            score = 0.5  # Moderate buildup
            analysis = "Moderate OI buildup"
        elif velocity < self.thresholds["oi_velocity_low"]:
            score = -1.0  # Strong unwinding
            analysis = "Strong OI unwinding - capital exiting"
        elif velocity < -0.5:
            score = -0.5  # Moderate unwinding
            analysis = "Moderate OI unwinding"
        else:
            score = 0.0  # Neutral
            analysis = "OI velocity neutral"
        
        # Regime adjustment
        if expansive > 0.5:
            score = max(score, 0.3)  # Bias bullish
            analysis += " (EXPANSIVE regime)"
        elif constricted > 0.5:
            score = min(score, -0.3)  # Bias bearish
            analysis += " (CONSTRICTED regime)"
        
        return score, analysis
    
    def score_gamma_exposure(self, feats: Dict) -> Tuple[float, str]:
        """
        Score Gamma Exposure component.
        
        Returns:
            score (-1 to 1), analysis
        """
        net_gamma = feats.get("net_gamma", 0.0)
        gamma_pos = feats.get("gamma_regime_positive", 0.0)
        gamma_neg = feats.get("gamma_regime_negative", 0.0)
        flip_distance = feats.get("gamma_flip_distance", 1.0)
        
        # Gamma scoring
        if gamma_neg > 0.5 and abs(net_gamma) > self.thresholds["gamma_high"]:
            score = -1.0  # Strong negative gamma (accelerating)
            analysis = "Strong negative gamma - volatility acceleration likely"
        elif gamma_neg > 0.5:
            score = -0.7  # Moderate negative gamma
            analysis = "Negative gamma regime - trending moves possible"
        elif gamma_pos > 0.5 and abs(net_gamma) > self.thresholds["gamma_high"]:
            score = 0.3  # Strong positive gamma (stabilizing)
            analysis = "Strong positive gamma - range-bound/pinning likely"
        elif gamma_pos > 0.5:
            score = 0.1  # Moderate positive gamma
            analysis = "Positive gamma regime - mean reversion favored"
        else:
            score = 0.0  # Neutral
            analysis = "Gamma exposure neutral"
        
        # Flip distance adjustment (closer to flip = more uncertainty)
        if flip_distance < 0.01:  # Very close to flip
            score *= 0.5  # Reduce confidence near flip
            analysis += " (Near gamma flip)"
        
        return score, analysis
    
    def score_structure(self, feats: Dict) -> Tuple[float, str, Optional[TrapAnalysis]]:
        """
        Score structural components (walls, traps, Wyckoff).
        
        Returns:
            score (-1 to 1), analysis, trap_analysis
        """
        wall_strength = feats.get("wall_strength", 0.0)
        wall_defense = feats.get("wall_defense_score", 0.0)
        trap_prob = feats.get("trap_probability", 0.0)
        spring = feats.get("spring_detection", 0.0)
        upthrust = feats.get("upthrust_detection", 0.0)
        accumulation = feats.get("accumulation_score", 0.0)
        
        score = 0.0
        analysis = []
        trap_analysis = None
        
        # Wall analysis
        if wall_strength > self.thresholds["wall_strength_high"]:
            if wall_defense > 0.7:
                score += 0.3  # Strong defense = continuation
                analysis.append(f"Strong wall defense ({wall_strength:.2f})")
            else:
                score -= 0.2  # Weak defense = potential break
                analysis.append(f"Weak wall defense ({wall_strength:.2f})")
        
        # Trap analysis
        if trap_prob > self.thresholds["trap_prob_high"]:
            # High trap probability
            trap_type = self._classify_trap_type(feats)
            if trap_type in [TrapType.GAMMA_TRAP, TrapType.WYCKOFF_SPRING]:
                # Bullish traps
                trap_analysis = TrapAnalysis(
                    trap_type=trap_type,
                    strike_level=0,  # Would be actual strike in production
                    direction="BULLISH",
                    confidence=trap_prob,
                    trigger_conditions=["Price breach", "OI unwinding"],
                    expected_move_pct=2.0  # Estimated move
                )
                score += 0.5  # Bullish bias
                analysis.append(f"{trap_type.value} detected (conf: {trap_prob:.2f})")
            elif trap_type in [TrapType.OI_TRAP, TrapType.WYCKOFF_UPTHRUST]:
                # Bearish traps
                trap_analysis = TrapAnalysis(
                    trap_type=trap_type,
                    strike_level=0,
                    direction="BEARISH",
                    confidence=trap_prob,
                    trigger_conditions=["Price rejection", "OI buildup"],
                    expected_move_pct=-2.0
                )
                score -= 0.5  # Bearish bias
                analysis.append(f"{trap_type.value} detected (conf: {trap_prob:.2f})")
        
        # Wyckoff pattern analysis
        if spring > self.thresholds["spring_detection_high"]:
            score += 0.4  # Spring = bullish
            analysis.append(f"Wyckoff Spring detected (strength: {spring:.2f})")
        
        if upthrust > self.thresholds["upthrust_detection_high"]:
            score -= 0.4  # Upthrust = bearish
            analysis.append(f"Wyckoff Upthrust detected (strength: {upthrust:.2f})")
        
        # Accumulation/distribution
        if accumulation > 0.7:
            score += 0.3  # Accumulation = bullish
            analysis.append(f"Accumulation phase (score: {accumulation:.2f})")
        elif accumulation < 0.3:
            score -= 0.3  # Distribution = bearish
            analysis.append(f"Distribution phase (score: {accumulation:.2f})")
        
        # Clamp score
        score = max(-1.0, min(1.0, score))
        
        return score, " | ".join(analysis) if analysis else "Structure neutral", trap_analysis
    
    def score_divergence(self, feats: Dict) -> Tuple[float, str]:
        """
        Score divergence components.
        
        Returns:
            score (-1 to 1), analysis
        """
        divergence_score = feats.get("divergence_score", 0.0)
        has_divergence = feats.get("has_divergence", 0.0)
        price_oi_div = feats.get("price_oi_divergence", 0.0)
        price_gamma_div = feats.get("price_gamma_divergence", 0.0)
        
        if has_divergence < 0.5:
            return 0.0, "No significant divergence"
        
        score = 0.0
        analysis = []
        
        # Price-OI divergence (research concept)
        if abs(price_oi_div) > 1.0:  # Significant divergence
            if price_oi_div > 0:  # Price up, OI down = bearish divergence
                score -= 0.4
                analysis.append("Bearish Price-OI divergence")
            else:  # Price down, OI up = bullish divergence
                score += 0.4
                analysis.append("Bullish Price-OI divergence")
        
        # Price-Gamma divergence
        if abs(price_gamma_div) > 0.5:
            if price_gamma_div > 0:  # Price moving against gamma regime
                score -= 0.3
                analysis.append("Price-Gamma regime divergence")
            else:
                score += 0.3
                analysis.append("Price-Gamma regime convergence")
        
        # Overall divergence score
        if divergence_score > self.thresholds["divergence_high"]:
            # High divergence confidence
            if divergence_score > 0:  # Bullish divergence
                score += 0.3
                analysis.append(f"Strong bullish divergence (conf: {divergence_score:.2f})")
            else:  # Bearish divergence
                score -= 0.3
                analysis.append(f"Strong bearish divergence (conf: {abs(divergence_score):.2f})")
        
        # Clamp score
        score = max(-1.0, min(1.0, score))
        
        return score, " | ".join(analysis) if analysis else "Divergence neutral"
    
    def score_trend_momentum(self, feats: Dict) -> Tuple[float, float, str]:
        """
        Score traditional trend and momentum components.
        
        Returns:
            trend_score, momentum_score, analysis
        """
        price_momentum = feats.get("price_momentum", 0.0)
        ccc_slope = feats.get("ccc_slope", 0.0)
        vwap_distance = feats.get("vwap_distance", 0.0)
        put_call_ratio = feats.get("put_call_ratio", 1.0)
        
        # Trend score (directional bias)
        trend_score = 0.0
        
        # Price momentum
        if price_momentum > 0.01:
            trend_score += 0.3
        elif price_momentum < -0.01:
            trend_score -= 0.3
        
        # Breadth momentum (CCC slope)
        if ccc_slope > 0.001:
            trend_score += 0.2
        elif ccc_slope < -0.001:
            trend_score -= 0.2
        
        # VWAP position
        if vwap_distance > 0.005:
            trend_score += 0.2  # Above VWAP = bullish
        elif vwap_distance < -0.005:
            trend_score -= 0.2  # Below VWAP = bearish
        
        # Put-Call ratio sentiment
        if put_call_ratio < 0.8:
            trend_score += 0.1  # Low PCR = bullish
        elif put_call_ratio > 1.2:
            trend_score -= 0.1  # High PCR = bearish
        
        # Momentum score (strength of move)
        momentum_score = abs(price_momentum) * 10  # Scale to 0-1 range
        momentum_score = min(momentum_score, 1.0)
        
        # Analysis
        analysis_parts = []
        if price_momentum > 0:
            analysis_parts.append(f"Price momentum: +{price_momentum*100:.1f}%")
        else:
            analysis_parts.append(f"Price momentum: {price_momentum*100:.1f}%")
        
        if ccc_slope > 0:
            analysis_parts.append(f"Breadth improving (CCC: +{ccc_slope:.3f})")
        elif ccc_slope < 0:
            analysis_parts.append(f"Breadth weakening (CCC: {ccc_slope:.3f})")
        
        analysis = " | ".join(analysis_parts) if analysis_parts else "Momentum neutral"
        
        return trend_score, momentum_score, analysis
    
    # ==============================
    # TRAP CLASSIFICATION
    # ==============================
    
    def _classify_trap_type(self, feats: Dict) -> TrapType:
        """
        Classify trap type based on feature combination.
        """
        trap_prob = feats.get("trap_probability", 0.0)
        gamma_neg = feats.get("gamma_regime_negative", 0.0)
        spring = feats.get("spring_detection", 0.0)
        upthrust = feats.get("upthrust_detection", 0.0)
        divergence = feats.get("has_divergence", 0.0)
        
        # Check for specific trap types
        if gamma_neg > 0.5 and trap_prob > 0.6:
            return TrapType.GAMMA_TRAP
        
        if trap_prob > 0.6 and gamma_neg < 0.5:
            return TrapType.OI_TRAP
        
        if spring > self.thresholds["spring_detection_high"]:
            return TrapType.WYCKOFF_SPRING
        
        if upthrust > self.thresholds["upthrust_detection_high"]:
            return TrapType.WYCKOFF_UPTHRUST
        
        if divergence > 0.5 and trap_prob > 0.5:
            return TrapType.DIVERGENCE_TRAP
        
        # Default
        return TrapType.OI_TRAP
    
    # ==============================
    # MARKET REGIME DETECTION
    # ==============================
    
    def detect_market_regime(self, feats: Dict, components: SignalComponents) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime based on features.
        
        Returns:
            regime, confidence (0-1)
        """
        regime_scores = {
            MarketRegime.ACCUMULATION: 0.0,
            MarketRegime.DISTRIBUTION: 0.0,
            MarketRegime.UPTREND: 0.0,
            MarketRegime.DOWNTREND: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.BREAKOUT: 0.0,
            MarketRegime.SQUEEZE: 0.0,
            MarketRegime.REVERSAL: 0.0
        }
        
        # Extract key features
        accumulation = feats.get("accumulation_score", 0.0)
        trap_prob = feats.get("trap_probability", 0.0)
        gamma_neg = feats.get("gamma_regime_negative", 0.0)
        divergence = feats.get("has_divergence", 0.0)
        price_momentum = feats.get("price_momentum", 0.0)
        
        # Score regimes
        # Accumulation/Distribution
        if accumulation > 0.7:
            regime_scores[MarketRegime.ACCUMULATION] = accumulation
        elif accumulation < 0.3:
            regime_scores[MarketRegime.DISTRIBUTION] = 1 - accumulation
        
        # Trend detection
        if abs(price_momentum) > 0.02:  # Strong trend
            if price_momentum > 0:
                regime_scores[MarketRegime.UPTREND] = abs(price_momentum) * 10
            else:
                regime_scores[MarketRegime.DOWNTREND] = abs(price_momentum) * 10
        else:
            regime_scores[MarketRegime.RANGING] = 0.7
        
        # Breakout/Squeeze
        if trap_prob > 0.6:
            if gamma_neg > 0.5:
                regime_scores[MarketRegime.SQUEEZE] = trap_prob
            else:
                regime_scores[MarketRegime.BREAKOUT] = trap_prob
        
        # Reversal
        if divergence > 0.5 and abs(price_momentum) > 0.01:
            regime_scores[MarketRegime.REVERSAL] = divergence
        
        # Find highest scoring regime
        best_regime_item = max(regime_scores.items(), key=lambda x: x[1])
        best_regime = best_regime_item[0]
        regime_confidence = best_regime_item[1]
        
        # If no clear regime, default to RANGING
        if regime_confidence < 0.3:
            return MarketRegime.RANGING, 0.3
        
        return best_regime, min(regime_confidence, 1.0)
    
    # ==============================
    # COMPOSITE SIGNAL GENERATION
    # ==============================
    
    def compute_composite_signal(self, components: SignalComponents) -> Tuple[str, float, SignalStrength]:
        """
        Compute final signal from components.
        
        Returns:
            signal_type, confidence, strength
        """
        composite = components.composite_score
        confidence = components.confidence
        
        # Determine signal type
        if composite > 0.3 and confidence > self.confidence_threshold:
            signal_type = "BUY"
            if composite > 0.6 and confidence > 0.7:
                strength = SignalStrength.VERY_STRONG
            elif composite > 0.4:
                strength = SignalStrength.STRONG
            elif composite > 0.3:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
                
        elif composite < -0.3 and confidence > self.confidence_threshold:
            signal_type = "SELL"
            if composite < -0.6 and confidence > 0.7:
                strength = SignalStrength.VERY_STRONG
            elif composite < -0.4:
                strength = SignalStrength.STRONG
            elif composite < -0.3:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
                
        else:
            signal_type = "NEUTRAL"
            strength = SignalStrength.WEAK
        
        return signal_type, confidence, strength
    
    # ==============================
    # MAIN DECISION ENGINE
    # ==============================
    
    def decide(self, feature_row: pd.Series) -> SignalComponents:
        """
        Main decision engine with research-based scoring.
        """
        # Extract research features
        feats = self.extract_research_features(feature_row)
        
        # Score all components
        trend_score, momentum_score, trend_analysis = self.score_trend_momentum(feats)
        oi_score, oi_analysis = self.score_oi_velocity(feats)
        gamma_score, gamma_analysis = self.score_gamma_exposure(feats)
        structure_score, structure_analysis, trap_analysis = self.score_structure(feats)
        divergence_score, divergence_analysis = self.score_divergence(feats)
        
        # Calculate Wyckoff phase score
        wyckoff_score = 0.0
        pattern_score = 0.0
        
        accumulation = feats.get("accumulation_score", 0.0)
        spring = feats.get("spring_detection", 0.0)
        upthrust = feats.get("upthrust_detection", 0.0)
        
        if accumulation > 0.5:
            wyckoff_score = accumulation - 0.5  # 0 to 0.5 range
        
        pattern_score = spring - upthrust  # Positive for springs, negative for upthrusts
        
        # Calculate composite score with weights
        weights = {
            "trend": 0.15,
            "momentum": 0.10,
            "oi_velocity": 0.20,     # High weight for OI velocity (research important)
            "gamma": 0.20,           # High weight for gamma (research important)
            "structure": 0.15,
            "divergence": 0.10,
            "wyckoff": 0.05,
            "pattern": 0.05
        }
        
        weighted_sum = (
            trend_score * weights["trend"] +
            momentum_score * np.sign(trend_score) * weights["momentum"] +  # Momentum amplifies trend
            oi_score * weights["oi_velocity"] +
            gamma_score * weights["gamma"] +
            structure_score * weights["structure"] +
            divergence_score * weights["divergence"] +
            wyckoff_score * weights["wyckoff"] +
            pattern_score * weights["pattern"]
        )
        
        # Calculate confidence
        component_scores = [
            abs(trend_score), abs(oi_score), abs(gamma_score),
            abs(structure_score), abs(divergence_score)
        ]
        confidence = np.mean([s for s in component_scores if s > 0.1]) if any(s > 0.1 for s in component_scores) else 0.0
        
        # Detect market regime
        market_regime, regime_confidence = self.detect_market_regime(feats, SignalComponents(
            trend_score=trend_score,
            momentum_score=momentum_score,
            oi_velocity_score=oi_score,
            gamma_score=gamma_score,
            wall_interaction_score=structure_score,
            divergence_score=divergence_score,
            wyckoff_phase_score=wyckoff_score,
            pattern_score=pattern_score,
            composite_score=weighted_sum,
            confidence=confidence,
            market_regime=MarketRegime.RANGING,  # Temporary placeholder
            regime_confidence=0.5  # Temporary placeholder
        ))
        
        # Store trap analysis if found
        if trap_analysis:
            self.trap_detections.append({
                "timestamp": feature_row.get("timestamp", ""),
                "trap_analysis": trap_analysis,
                "features": feats
            })
        
        return SignalComponents(
            trend_score=trend_score,
            momentum_score=momentum_score,
            oi_velocity_score=oi_score,
            gamma_score=gamma_score,
            wall_interaction_score=structure_score,
            divergence_score=divergence_score,
            wyckoff_phase_score=wyckoff_score,
            pattern_score=pattern_score,
            composite_score=weighted_sum,
            confidence=confidence,
            market_regime=market_regime,
            regime_confidence=regime_confidence
        )
    
    # ==============================
    # SIGNAL OBJECT GENERATION
    # ==============================
    
    def build_signal(
        self,
        feature_row: pd.Series,
        model_version: str = "research_v2"
    ) -> Dict:
        """
        Build complete signal record with research context.
        """

        # --- EXECUTION SAFETY GATE ---
        mode = feature_row.get("pipeline_mode", "research")
        execution_ready = feature_row.get("execution_ready", False)

        if mode == "execution" and not execution_ready:
            return {
                "signal_type": "NEUTRAL",
                "confidence": 0.0,
                "signal_strength": SignalStrength.WEAK.value,
                "reason": "Execution not allowed: feature row not execution-ready",
                "timestamp": feature_row.get("timestamp"),
                "feature_version": feature_row.get("feature_version"),
                "model_version": model_version,
                "status": "BLOCKED"
            }



        # Run decision engine
        components = self.decide(feature_row)
        
        # Compute final signal
        signal_type, confidence, strength = self.compute_composite_signal(components)
        
        # Generate signal ID
        signal_id = str(uuid.uuid4())
        
        # Current time
        now = datetime.utcnow()
        
        # Build research context
        research_context = {
            "components": {
                "trend_score": round(components.trend_score, 3),
                "momentum_score": round(components.momentum_score, 3),
                "oi_velocity_score": round(components.oi_velocity_score, 3),
                "gamma_score": round(components.gamma_score, 3),
                "wall_interaction_score": round(components.wall_interaction_score, 3),
                "divergence_score": round(components.divergence_score, 3),
                "wyckoff_phase_score": round(components.wyckoff_phase_score, 3),
                "pattern_score": round(components.pattern_score, 3),
                "composite_score": round(components.composite_score, 3)
            },
            "market_regime": components.market_regime.value,
            "regime_confidence": round(components.regime_confidence, 3),
            "signal_strength": strength.value,
            "thresholds_used": self.thresholds
        }
        
        # Build analytics summary
        analytics_summary = {
            "oi_velocity": feature_row.get("oi_velocity", 0),
            "net_gamma": feature_row.get("net_gamma", 0),
            "trap_probability": feature_row.get("trap_probability", 0),
            "divergence_score": feature_row.get("divergence_score", 0),
            "wall_strength": feature_row.get("wall_strength", 0),
            "confidence_breakdown": {
                "component_confidence": round(components.confidence, 3),
                "regime_confidence": round(components.regime_confidence, 3),
                "composite_confidence": round(confidence, 3)
            }
        }
        
        # Build rationale
        rationale_parts = []

        # Extract values from feature_row (not from components)
        trap_probability = feature_row.get("trap_probability", 0.0)
        has_divergence = feature_row.get("has_divergence", 0.0)

        # Add component analysis
        if abs(components.oi_velocity_score) > 0.3:
            direction = "bullish" if components.oi_velocity_score > 0 else "bearish"
            rationale_parts.append(f"{direction.capitalize()} OI velocity")

        if abs(components.gamma_score) > 0.3:
            regime = "negative" if components.gamma_score < 0 else "positive"
            rationale_parts.append(f"{regime.capitalize()} gamma regime")

        if trap_probability > 0.5:
            rationale_parts.append(f"High trap probability ({trap_probability:.2f})")

        if has_divergence > 0.5:
            rationale_parts.append("Significant divergence detected")

        rationale = " | ".join(rationale_parts) if rationale_parts else "rule_based_research_v2"
        
        # Build complete signal
        signal = {
            "signal_id": signal_id,
            "timestamp": feature_row["timestamp"],
            "feature_version": feature_row.get("feature_version", FEATURE_VERSION),
            "model_version": model_version,
            
            "signal_type": signal_type,
            "confidence": round(confidence, 3),
            
            "market_state": components.market_regime.value,
            "rationale": rationale,
            
            "expiry_time": (now + timedelta(minutes=self.signal_expiry_minutes)).isoformat(),
            "status": "NEW",
            
            "created_at": now.isoformat(),
            
            # Enhanced fields
            "research_context": research_context,
            "analytics_summary": analytics_summary,
            "signal_strength": strength.value,
            
            # Additional metadata
            "spot_price": feature_row.get("spot_price", 0),
            "put_call_ratio": feature_row.get("put_call_ratio", 1.0),
            "oi_velocity": feature_row.get("oi_velocity", 0),
            "net_gamma": feature_row.get("net_gamma", 0)
        }
        
        # Store in history
        self.signal_history.append({
            "timestamp": now.isoformat(),
            "signal": signal,
            "components": components
        })
        
        # Keep history manageable
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        return signal
    
    # ==============================
    # UTILITY METHODS
    # ==============================
    
    def get_signal_history(self, limit: int = 10) -> List[Dict]:
        """Get recent signal history."""
        return self.signal_history[-limit:] if self.signal_history else []
    
    def get_recent_trap_detections(self, limit: int = 5) -> List[Dict]:
        """Get recent trap detections."""
        return self.trap_detections[-limit:] if self.trap_detections else []
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics for recent signals."""
        if not self.signal_history:
            return {}
        
        recent_signals = self.signal_history[-100:]  # Last 100 signals
        
        buy_signals = [s for s in recent_signals if s["signal"]["signal_type"] == "BUY"]
        sell_signals = [s for s in recent_signals if s["signal"]["signal_type"] == "SELL"]
        
        metrics = {
            "total_signals": len(recent_signals),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "neutral_signals": len(recent_signals) - len(buy_signals) - len(sell_signals),
            "avg_confidence": np.mean([s["signal"]["confidence"] for s in recent_signals]) if recent_signals else 0,
            "recent_regimes": {},
            "trap_detections": len(self.trap_detections)
        }
        
        # Count recent regimes
        for signal in recent_signals[-20:]:  # Last 20 signals
            regime = signal["signal"]["market_state"]
            metrics["recent_regimes"][regime] = metrics["recent_regimes"].get(regime, 0) + 1
        
        return metrics
    
    def reset(self):
        """Reset state machine history."""
        self.signal_history = []
        self.regime_history = []
        self.trap_detections = []

# ==============================
# SIGNAL VALIDATION & FILTERING
# ==============================

class SignalValidator:
    """
    Validates signals based on research criteria.
    """
    
    @staticmethod
    def validate_signal(signal: Dict, feature_row: pd.Series) -> Tuple[bool, str]:
        """
        Validate signal against research criteria.
        
        Returns:
            is_valid, reason
        """
        # Check confidence threshold
        if signal.get("confidence", 0) < 0.2:
            return False, "Confidence below threshold"
        
        # Check for trap confirmation
        trap_prob = feature_row.get("trap_probability", 0)
        if trap_prob > 0.7 and signal.get("signal_type") != "NEUTRAL":
            # High trap probability requires careful validation
            gamma_neg = feature_row.get("gamma_regime_negative", 0)
            if gamma_neg > 0.5:
                # Negative gamma with high trap = likely valid
                return True, "Gamma trap confirmed"
            else:
                return False, "High trap probability without negative gamma"
        
        # Check divergence consistency
        has_divergence = feature_row.get("has_divergence", 0)
        if has_divergence > 0.5:
            # Signal should align with divergence direction
            divergence_score = feature_row.get("divergence_score", 0)
            signal_type = signal.get("signal_type")
            
            if divergence_score > 0 and signal_type != "BUY":
                return False, "Signal contradicts bullish divergence"
            elif divergence_score < 0 and signal_type != "SELL":
                return False, "Signal contradicts bearish divergence"
        
        # Check OI velocity consistency
        oi_velocity = feature_row.get("oi_velocity", 0)
        if abs(oi_velocity) > 1.5:  # Strong OI movement
            if oi_velocity > 0 and signal.get("signal_type") != "BUY":
                return False, "Signal contradicts strong OI buildup"
            elif oi_velocity < 0 and signal.get("signal_type") != "SELL":
                return False, "Signal contradicts strong OI unwinding"
        
        return True, "Signal validated"
    
    @staticmethod
    def filter_weak_signals(signals: List[Dict], min_strength: str = "MODERATE") -> List[Dict]:
        """
        Filter signals by strength.
        
        Args:
            signals: List of signals
            min_strength: Minimum strength required
        
        Returns:
            Filtered signals
        """
        strength_order = {
            "WEAK": 0,
            "MODERATE": 1,
            "STRONG": 2,
            "VERY_STRONG": 3
        }
        
        min_strength_value = strength_order.get(min_strength, 0)
        
        filtered = []
        for signal in signals:
            signal_strength = signal.get("signal_strength", "WEAK")
            if strength_order.get(signal_strength, 0) >= min_strength_value:
                filtered.append(signal)
        
        return filtered

# ==============================
# SIGNAL ANALYTICS
# ==============================

def analyze_signal_patterns(signals: List[Dict]) -> Dict:
    """
    Analyze patterns in signal generation.
    """
    if not signals:
        return {}
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(signals)
    
    analysis = {
        "total_signals": len(df),
        "signal_distribution": df["signal_type"].value_counts().to_dict(),
        "avg_confidence": df["confidence"].mean() if "confidence" in df.columns else 0,
        "strength_distribution": df["signal_strength"].value_counts().to_dict() if "signal_strength" in df.columns else {},
        "regime_distribution": df["market_state"].value_counts().to_dict() if "market_state" in df.columns else {}
    }
    
    # Calculate success rate if PNL data available
    if "pnl" in df.columns and not df["pnl"].isna().all():
        profitable = df[df["pnl"] > 0]
        analysis["profitable_signals"] = len(profitable)
        analysis["success_rate"] = len(profitable) / len(df) * 100 if len(df) > 0 else 0
        analysis["avg_pnl"] = df["pnl"].mean()
    
    return analysis

# ==============================
# INITIALIZATION
# ==============================

def create_signal_engine(
    signal_expiry_minutes: int = 5,
    confidence_threshold: float = 0.2
) -> SignalStateMachine:
    """
    Factory function to create signal engine.
    """
    return SignalStateMachine(
        signal_expiry_minutes=signal_expiry_minutes,
        confidence_threshold=confidence_threshold
    )