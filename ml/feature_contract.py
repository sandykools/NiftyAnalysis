"""
ENHANCED FEATURE CONTRACT - SINGLE SOURCE OF TRUTH
Incorporates research concepts: OI Velocity, Gamma Exposure, Walls/Traps, Spot Divergence

Used by:
- Live data collection
- ML training
- ML inference
- Backtesting
- Signal generation
"""

# ==============================
# FEATURE VERSION
# ==============================

FEATURE_VERSION = "v2.0"  # Updated for research features

# ==============================
# FEATURE CATEGORIES
# ==============================

# Base features (original set)
BASE_FEATURES = [
    "timestamp",
    
    # Option structure (original)
    "put_call_ratio",
    "oi_delta",
    "oi_concentration",
    "atm_iv",
    "iv_skew",
    
    # Price & flow (original)
    "vwap_distance",
    "price_momentum",
    "volume_ratio",
    
    # Breadth (original)
    "ccc_value",
    "ccc_slope",
    
    # Time context
    "time_to_expiry_minutes"
]

# Research features (enhanced)
RESEARCH_FEATURES = [
    # OI Velocity features
    "oi_velocity",
    "oi_velocity_ma",
    "oi_velocity_std",
    "oi_regime_expansive",      # 1.0 if EXPANSIVE, else 0.0
    "oi_regime_constricted",    # 1.0 if CONSTRICTED, else 0.0
    
    # Gamma Exposure features
    "net_gamma",
    "gamma_regime_positive",    # 1.0 if POSITIVE, else 0.0
    "gamma_regime_negative",    # 1.0 if NEGATIVE, else 0.0
    "gamma_flip_distance",      # Distance to nearest gamma flip (normalized)
    "max_gamma_strike_distance",# Distance to max gamma strike (normalized)
    
    # Structural features (Walls & Traps)
    "wall_strength",            # Combined strength of top walls (0-1)
    "wall_defense_score",       # How well walls are defended (0-1)
    "trap_probability",         # Probability of trap formation (0-1)
    
    # Divergence features
    "price_oi_divergence",      # Divergence between price and OI
    "price_gamma_divergence",   # Divergence between price and gamma
    "divergence_score",         # Combined divergence score (0-1)
    "has_divergence",           # 1.0 if significant divergence, else 0.0
    
    # Market microstructure
    "max_pain_distance",        # Distance to max pain (normalized)
    "vix_smile",                # Volatility smile curvature
    "skewness",                 # Option skew (put IV - call IV)
    
    # Wyckoff-inspired features
    "spring_detection",         # Bear trap probability (0-1)
    "upthrust_detection",       # Bull trap probability (0-1)
    "accumulation_score",       # Accumulation phase score (0-1)
    
    # Derived composite features
    "gamma_wall_interaction",   # wall_strength * abs(net_gamma)
    "velocity_divergence_composite",  # oi_velocity * divergence_score
    "trap_gamma_composite"      # trap_probability * gamma_regime_negative
]

# ==============================
# COMPLETE FEATURE SET
# ==============================

# All features (base + research)
FEATURE_COLUMNS = BASE_FEATURES + RESEARCH_FEATURES

# Target variable
TARGET_COLUMN = "future_return_5m"

# Metadata columns
METADATA_COLUMNS = [
    "feature_version",
    "timestamp"
]

# Primary keys for database
PRIMARY_KEYS = ["timestamp"]

# ==============================
# FEATURE GROUPS FOR ANALYSIS
# ==============================

FEATURE_GROUPS = {
    "option_structure": [
        "put_call_ratio",
        "oi_delta", 
        "oi_concentration",
        "atm_iv",
        "iv_skew",
        "skewness",
        "vix_smile"
    ],
    
    "price_momentum": [
        "vwap_distance",
        "price_momentum",
        "volume_ratio",
        "ccc_value",
        "ccc_slope"
    ],
    
    "oi_analysis": [
        "oi_velocity",
        "oi_velocity_ma",
        "oi_velocity_std",
        "oi_regime_expansive",
        "oi_regime_constricted"
    ],
    
    "gamma_exposure": [
        "net_gamma",
        "gamma_regime_positive",
        "gamma_regime_negative",
        "gamma_flip_distance",
        "max_gamma_strike_distance"
    ],
    
    "structure_analysis": [
        "wall_strength",
        "wall_defense_score",
        "trap_probability",
        "spring_detection",
        "upthrust_detection",
        "accumulation_score"
    ],
    
    "divergence_analysis": [
        "price_oi_divergence",
        "price_gamma_divergence",
        "divergence_score",
        "has_divergence"
    ],
    
    "composite_features": [
        "gamma_wall_interaction",
        "velocity_divergence_composite",
        "trap_gamma_composite"
    ],
    
    "context_features": [
        "time_to_expiry_minutes",
        "max_pain_distance"
    ]
}

# ==============================
# FEATURE DESCRIPTIONS
# ==============================

FEATURE_DESCRIPTIONS = {
    # Base features
    "put_call_ratio": "Put OI / Call OI ratio. >1.0 = bearish sentiment",
    "oi_delta": "Net OI change (Put OI change - Call OI change)",
    "oi_concentration": "Maximum OI concentration at any single strike",
    "atm_iv": "At-the-money implied volatility",
    "iv_skew": "Call IV - Put IV (positive = call skew, negative = put skew)",
    "vwap_distance": "(LTP - VWAP) / VWAP. Positive = above VWAP (bullish)",
    "price_momentum": "Normalized price momentum over lookback period",
    "volume_ratio": "Current volume / average volume",
    "ccc_value": "Cumulative Constituent Contribution - breadth indicator",
    "ccc_slope": "Slope of CCC over lookback period",
    "time_to_expiry_minutes": "Minutes remaining until option expiry",
    
    # OI Velocity features
    "oi_velocity": "Normalized rate of change of Open Interest (σ)",
    "oi_velocity_ma": "Moving average of OI velocity",
    "oi_velocity_std": "Standard deviation of OI velocity",
    "oi_regime_expansive": "1.0 if OI velocity > 1.5σ (capital inflow), else 0.0",
    "oi_regime_constricted": "1.0 if OI velocity < -1.5σ (capital outflow), else 0.0",
    
    # Gamma Exposure features
    "net_gamma": "Net Gamma Exposure of market makers",
    "gamma_regime_positive": "1.0 if net_gamma > 0 (stabilizing/pinning), else 0.0",
    "gamma_regime_negative": "1.0 if net_gamma < 0 (accelerating), else 0.0",
    "gamma_flip_distance": "Distance to nearest gamma flip level (normalized)",
    "max_gamma_strike_distance": "Distance to strike with maximum gamma impact",
    
    # Structural features
    "wall_strength": "Combined strength of structural walls (0-1 scale)",
    "wall_defense_score": "How strongly walls are being defended (0-1)",
    "trap_probability": "Probability of trap/squeeze formation (0-1)",
    "spring_detection": "Wyckoff spring pattern detection (0-1)",
    "upthrust_detection": "Wyckoff upthrust pattern detection (0-1)",
    "accumulation_score": "Accumulation phase score (0-1)",
    
    # Divergence features
    "price_oi_divergence": "Divergence between price change and OI change",
    "price_gamma_divergence": "Divergence between price change and gamma change",
    "divergence_score": "Combined divergence confidence (0-1)",
    "has_divergence": "1.0 if significant divergence detected, else 0.0",
    
    # Market microstructure
    "max_pain_distance": "Distance to max pain strike (normalized)",
    "vix_smile": "Volatility smile curvature (ATM IV - OTM IV)",
    "skewness": "Option skew (Put IV - Call IV)",
    
    # Composite features
    "gamma_wall_interaction": "Interaction between gamma and wall strength",
    "velocity_divergence_composite": "OI velocity multiplied by divergence score",
    "trap_gamma_composite": "Trap probability weighted by negative gamma regime"
}

# ==============================
# FEATURE IMPORTANCE GUIDELINES
# ==============================

# Expected impact on returns (for initial model weighting)
FEATURE_IMPACT = {
    "high_impact": [
        "oi_velocity",
        "net_gamma",
        "trap_probability",
        "divergence_score",
        "wall_strength"
    ],
    
    "medium_impact": [
        "put_call_ratio",
        "price_momentum",
        "ccc_slope",
        "gamma_regime_negative",
        "has_divergence",
        "spring_detection"
    ],
    
    "low_impact": [
        "oi_concentration",
        "vwap_distance",
        "volume_ratio",
        "max_pain_distance",
        "skewness"
    ],
    
    "contextual": [
        "time_to_expiry_minutes",
        "atm_iv",
        "vix_smile",
        "accumulation_score"
    ]
}

# ==============================
# FEATURE VALIDATION RULES
# ==============================

FEATURE_VALIDATION = {
    "value_ranges": {
        "put_call_ratio": (0, 5),
        "oi_velocity": (-10, 10),
        "net_gamma": (-1e6, 1e6),
        "trap_probability": (0, 1),
        "divergence_score": (0, 1),
        "wall_strength": (0, 1)
    },
    
    "required_features": [
        "timestamp",
        "feature_version",
        "put_call_ratio",
        "vwap_distance",
        "price_momentum",
        "ccc_value"
    ],
    
    "derived_features": [
        "oi_regime_expansive",
        "oi_regime_constricted",
        "gamma_regime_positive",
        "gamma_regime_negative",
        "has_divergence"
    ]
}

# ==============================
# MODEL INPUT CONFIGURATION
# ==============================

# Features for different model types
MODEL_FEATURE_SETS = {
    "full_model": FEATURE_COLUMNS,
    
    "research_model": RESEARCH_FEATURES,
    
    "momentum_model": [
        "price_momentum",
        "volume_ratio",
        "ccc_slope",
        "oi_velocity",
        "net_gamma"
    ],
    
    "divergence_model": [
        "price_oi_divergence",
        "price_gamma_divergence",
        "divergence_score",
        "has_divergence",
        "trap_probability"
    ],
    
    "structure_model": [
        "wall_strength",
        "wall_defense_score",
        "trap_probability",
        "spring_detection",
        "upthrust_detection"
    ],
    
    "quick_model": [
        "put_call_ratio",
        "oi_velocity",
        "net_gamma",
        "trap_probability",
        "divergence_score"
    ]
}

# ==============================
# UTILITY FUNCTIONS
# ==============================

def get_feature_group(feature_name: str) -> str:
    """Get the group name for a feature."""
    for group_name, features in FEATURE_GROUPS.items():
        if feature_name in features:
            return group_name
    return "unknown"

def validate_feature_name(feature_name: str) -> bool:
    """Check if a feature name is valid."""
    return feature_name in FEATURE_COLUMNS or feature_name in METADATA_COLUMNS

def get_feature_description(feature_name: str) -> str:
    """Get description for a feature."""
    return FEATURE_DESCRIPTIONS.get(feature_name, "No description available")

def get_features_by_impact(impact_level: str) -> list:
    """Get features by impact level."""
    return FEATURE_IMPACT.get(impact_level, [])

def get_model_features(model_type: str = "full_model") -> list:
    """Get feature set for specific model type."""
    return MODEL_FEATURE_SETS.get(model_type, FEATURE_COLUMNS)

def print_feature_summary():
    """Print summary of all features."""
    print(f"=== FEATURE CONTRACT v{FEATURE_VERSION} ===")
    print(f"Total features: {len(FEATURE_COLUMNS)}")
    print(f"Base features: {len(BASE_FEATURES)}")
    print(f"Research features: {len(RESEARCH_FEATURES)}")
    print()
    
    print("Feature Groups:")
    for group_name, features in FEATURE_GROUPS.items():
        print(f"  {group_name}: {len(features)} features")
    
    print()
    print("Top Impact Features:")
    for feature in FEATURE_IMPACT["high_impact"][:5]:
        print(f"  • {feature}: {FEATURE_DESCRIPTIONS.get(feature, '')}")

# ==============================
# DATABASE SCHEMA
# ==============================

# Schema for market_features table
MARKET_FEATURES_SCHEMA = {
    "table_name": "market_features",
    "columns": {
        "timestamp": "TEXT PRIMARY KEY",
        "feature_version": "TEXT",
        "future_return_5m": "REAL",
        
        # Base features
        "put_call_ratio": "REAL",
        "oi_delta": "REAL",
        "oi_concentration": "REAL",
        "atm_iv": "REAL",
        "iv_skew": "REAL",
        "vwap_distance": "REAL",
        "price_momentum": "REAL",
        "volume_ratio": "REAL",
        "ccc_value": "REAL",
        "ccc_slope": "REAL",
        "time_to_expiry_minutes": "INTEGER",
        
        # Research features (added in v2.0)
        "oi_velocity": "REAL",
        "oi_velocity_ma": "REAL",
        "oi_velocity_std": "REAL",
        "oi_regime_expansive": "REAL",
        "oi_regime_constricted": "REAL",
        "net_gamma": "REAL",
        "gamma_regime_positive": "REAL",
        "gamma_regime_negative": "REAL",
        "gamma_flip_distance": "REAL",
        "max_gamma_strike_distance": "REAL",
        "wall_strength": "REAL",
        "wall_defense_score": "REAL",
        "trap_probability": "REAL",
        "price_oi_divergence": "REAL",
        "price_gamma_divergence": "REAL",
        "divergence_score": "REAL",
        "has_divergence": "REAL",
        "max_pain_distance": "REAL",
        "vix_smile": "REAL",
        "skewness": "REAL",
        "spring_detection": "REAL",
        "upthrust_detection": "REAL",
        "accumulation_score": "REAL",
        "gamma_wall_interaction": "REAL",
        "velocity_divergence_composite": "REAL",
        "trap_gamma_composite": "REAL"
    },
    "indexes": [
        "CREATE INDEX idx_timestamp ON market_features(timestamp)",
        "CREATE INDEX idx_feature_version ON market_features(feature_version)",
        "CREATE INDEX idx_oi_velocity ON market_features(oi_velocity)",
        "CREATE INDEX idx_net_gamma ON market_features(net_gamma)"
    ]
}

# ==============================
# MIGRATION UTILITIES
# ==============================

def get_migration_sql(from_version: str, to_version: str) -> list:
    """
    Get SQL migration statements for feature version updates.
    
    Args:
        from_version: Current feature version
        to_version: Target feature version
    
    Returns:
        List of SQL statements to migrate
    """
    migrations = []
    
    if from_version == "v1.0" and to_version == "v2.0":
        # Add research feature columns
        research_columns = [
            "oi_velocity REAL DEFAULT 0.0",
            "oi_velocity_ma REAL DEFAULT 0.0",
            "oi_velocity_std REAL DEFAULT 0.0",
            "oi_regime_expansive REAL DEFAULT 0.0",
            "oi_regime_constricted REAL DEFAULT 0.0",
            "net_gamma REAL DEFAULT 0.0",
            "gamma_regime_positive REAL DEFAULT 0.0",
            "gamma_regime_negative REAL DEFAULT 0.0",
            "gamma_flip_distance REAL DEFAULT 0.0",
            "max_gamma_strike_distance REAL DEFAULT 0.0",
            "wall_strength REAL DEFAULT 0.0",
            "wall_defense_score REAL DEFAULT 0.0",
            "trap_probability REAL DEFAULT 0.0",
            "price_oi_divergence REAL DEFAULT 0.0",
            "price_gamma_divergence REAL DEFAULT 0.0",
            "divergence_score REAL DEFAULT 0.0",
            "has_divergence REAL DEFAULT 0.0",
            "max_pain_distance REAL DEFAULT 0.0",
            "vix_smile REAL DEFAULT 0.0",
            "skewness REAL DEFAULT 0.0",
            "spring_detection REAL DEFAULT 0.0",
            "upthrust_detection REAL DEFAULT 0.0",
            "accumulation_score REAL DEFAULT 0.0",
            "gamma_wall_interaction REAL DEFAULT 0.0",
            "velocity_divergence_composite REAL DEFAULT 0.0",
            "trap_gamma_composite REAL DEFAULT 0.0"
        ]
        
        for column_def in research_columns:
            column_name = column_def.split()[0]
            migrations.append(f"ALTER TABLE market_features ADD COLUMN {column_def}")
        
        # Add indexes for new features
        migrations.append("CREATE INDEX idx_oi_velocity ON market_features(oi_velocity)")
        migrations.append("CREATE INDEX idx_net_gamma ON market_features(net_gamma)")
        migrations.append("CREATE INDEX idx_trap_probability ON market_features(trap_probability)")
    
    return migrations

# ==============================
# INITIALIZATION
# ==============================

if __name__ == "__main__":
    print_feature_summary()