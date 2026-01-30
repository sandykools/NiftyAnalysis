-- ================================
-- FEATURE STORE (IMMUTABLE)
-- ================================
CREATE TABLE IF NOT EXISTS market_features (
    timestamp TEXT NOT NULL,
    feature_version TEXT NOT NULL,

    -- Option structure
    put_call_ratio REAL NOT NULL,
    oi_delta REAL NOT NULL,
    oi_concentration REAL NOT NULL,
    atm_iv REAL NOT NULL,
    iv_skew REAL NOT NULL,

    -- Price & flow
    vwap_distance REAL NOT NULL,
    price_momentum REAL NOT NULL,
    volume_ratio REAL NOT NULL,

    -- Breadth
    ccc_value REAL NOT NULL,
    ccc_slope REAL NOT NULL,

    -- Time context
    time_to_expiry_minutes INTEGER NOT NULL,

    -- ML target (NULL in live mode)
    future_return_5m REAL,

    PRIMARY KEY (timestamp, feature_version)
);

-- ================================
-- SIGNAL STATE MACHINE
-- ================================
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,

    feature_version TEXT NOT NULL,
    model_version TEXT NOT NULL,

    signal_type TEXT CHECK(signal_type IN ('BUY','SELL','NEUTRAL')) NOT NULL,
    confidence REAL NOT NULL,

    market_state TEXT NOT NULL,
    rationale TEXT,

    expiry_time TEXT NOT NULL,
    status TEXT CHECK(status IN ('NEW','VALIDATED','EXPIRED','EVALUATED')) NOT NULL,

    created_at TEXT NOT NULL
);

-- ================================
-- TRADE EXECUTION (FUTURE)
-- ================================
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,

    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity INTEGER NOT NULL,

    stop_loss REAL,
    take_profit REAL,

    pnl REAL,
    executed_at TEXT,
    closed_at TEXT,

    FOREIGN KEY(signal_id) REFERENCES signals(signal_id)
);

-- ================================
-- MODEL REGISTRY
-- ================================
CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY,
    feature_version TEXT NOT NULL,

    algorithm TEXT NOT NULL,
    trained_on_start TEXT NOT NULL,
    trained_on_end TEXT NOT NULL,

    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- ================================
-- SYSTEM HEALTH & DEBUG
-- ================================
CREATE TABLE IF NOT EXISTS system_health (
    timestamp TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT
);
