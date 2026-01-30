"""
Enhanced Storage Repository with Research Features Support - FIXED VERSION
Handles database operations for market features, signals, and analytics.
Includes migration system for feature version updates.
"""

import sqlite3
import pandas as pd
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np
from contextlib import contextmanager

# Import feature contract for schema
from ml.feature_contract import (
    FEATURE_VERSION, 
    FEATURE_COLUMNS, 
    METADATA_COLUMNS,
    MARKET_FEATURES_SCHEMA,
    get_migration_sql
)

# =========================
# DATABASE CONFIG
# =========================

DB_PATH = Path("G:/trading_app/storage/trading.db")

# Ensure directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# CONNECTION MANAGEMENT
# =========================

@contextmanager
def get_connection() -> sqlite3.Connection:
    """
    Context manager for database connections with automatic cleanup.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def initialize_database():
    """
    Initialize database with all required tables and indexes.
    """
    with get_connection() as conn:
        # Check if market_features table exists and get its columns
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_features'")
        table_exists = cur.fetchone() is not None
        
        if table_exists:
            # Get existing columns
            cur = conn.execute("PRAGMA table_info(market_features)")
            existing_columns = {row[1] for row in cur.fetchall()}
            print(f"Existing columns in market_features: {existing_columns}")
        else:
            existing_columns = set()
        
        # Get expected columns from schema
        expected_columns = set(MARKET_FEATURES_SCHEMA["columns"].keys())
        print(f"Expected columns in market_features: {expected_columns}")
        
        # Create table if it doesn't exist
        if not table_exists:
            print("Creating market_features table from scratch...")
            columns = MARKET_FEATURES_SCHEMA["columns"]
            column_defs = [f"{name} {dtype}" for name, dtype in columns.items()]
            
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS market_features (
                    {', '.join(column_defs)}
                )
            """
            
            conn.execute(create_table_sql)
            print("market_features table created successfully")
        
        # Check for missing columns and add them
        missing_columns = expected_columns - existing_columns
        if missing_columns:
            print(f"Adding missing columns: {missing_columns}")
            
            for column in missing_columns:
                dtype = MARKET_FEATURES_SCHEMA["columns"][column]
                try:
                    conn.execute(f"ALTER TABLE market_features ADD COLUMN {column} {dtype}")
                    print(f"  ✓ Added column: {column}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" in str(e):
                        print(f"  ⚠️ Column {column} already exists")
                    else:
                        raise
        
        # Create other tables
        print("Creating other tables...")
        
        # Create signals table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                model_version TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL,
                market_state TEXT,
                rationale TEXT,
                expiry_time TEXT,
                status TEXT,
                created_at TEXT,
                research_context TEXT,
                analytics_summary TEXT,
                pnl REAL DEFAULT NULL,
                exit_time TEXT,
                exit_price REAL,
                stop_loss_hit INTEGER DEFAULT 0,
                take_profit_hit INTEGER DEFAULT 0,
                trade_duration_seconds INTEGER
            )
        """)
        
        # Create model_registry table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                model_version TEXT PRIMARY KEY,
                feature_version TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                trained_on_start TEXT,
                trained_on_end TEXT,
                metrics_json TEXT,
                feature_importance_json TEXT,
                created_at TEXT,
                is_active INTEGER DEFAULT 0
            )
        """)
        
        # Create system_health table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                details_json TEXT
            )
        """)
        
        # Create research_analytics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_analytics (
                timestamp TEXT PRIMARY KEY,
                oi_velocity REAL,
                oi_regime TEXT,
                net_gamma REAL,
                gamma_regime TEXT,
                wall_strength REAL,
                trap_probability REAL,
                divergence_score REAL,
                market_regime TEXT,
                confidence REAL,
                insights_json TEXT,
                walls_json TEXT,
                traps_json TEXT,
                created_at TEXT
            )
        """)
        
        # Create feature_history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                change_pct REAL,
                created_at TEXT
            )
        """)
        
        # Create database_info table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS database_info (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        
        # Create indexes (skip if they already exist)
        print("Creating indexes...")
        index_sqls = [
            # market_features indexes
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON market_features(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_feature_version ON market_features(feature_version)",
            "CREATE INDEX IF NOT EXISTS idx_oi_velocity ON market_features(oi_velocity)",
            "CREATE INDEX IF NOT EXISTS idx_net_gamma ON market_features(net_gamma)",
            
            # signals indexes
            "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)",
            "CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)",
            
            # research_analytics indexes
            "CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON research_analytics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_regime ON research_analytics(market_regime)"
        ]
        
        for index_sql in index_sqls:
            try:
                conn.execute(index_sql)
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    print(f"  ⚠️ Could not create index: {e}")
        
        # Initialize database info
        current_version = get_current_feature_version(conn)
        if not current_version:
            conn.execute(
                "INSERT INTO database_info (key, value, updated_at) VALUES (?, ?, ?)",
                ("feature_version", FEATURE_VERSION, datetime.utcnow().isoformat())
            )
            print(f"✓ Set feature version to {FEATURE_VERSION}")
        else:
            print(f"✓ Current feature version: {current_version}")
        
        # Log initialization
        _log_system_health_conn(
            conn,
            "database",
            "INITIALIZED",
            f"Database initialized with feature version {FEATURE_VERSION}"
        )
        
        print(f"✓ Database initialized at {DB_PATH}")

def get_current_feature_version(conn: sqlite3.Connection) -> Optional[str]:
    """Get current feature version from database."""
    try:
        cur = conn.execute("SELECT value FROM database_info WHERE key = 'feature_version'")
        row = cur.fetchone()
        return row["value"] if row else None
    except:
        return None

# =========================
# DATABASE MIGRATION
# =========================

def migrate_database(target_version: str = FEATURE_VERSION) -> bool:
    """
    Migrate database to target feature version.
    """
    try:
        with get_connection() as conn:
            # Get current version
            current_version = get_current_feature_version(conn)
            if not current_version:
                current_version = "v1.0"
            
            if current_version == target_version:
                print(f"Database already at version {target_version}")
                return True
            
            print(f"Migrating database from {current_version} to {target_version}")
            
            # Get migration SQL
            migrations = get_migration_sql(current_version, target_version)
            
            if not migrations:
                print("No migration required")
                return True
            
            # Execute migrations
            for migration_sql in migrations:
                try:
                    print(f"Executing: {migration_sql[:80]}...")
                    conn.execute(migration_sql)
                except sqlite3.OperationalError as e:
                    if "already exists" in str(e) or "duplicate column" in str(e):
                        print(f"  ⚠️ Skipping (already exists): {migration_sql[:50]}...")
                    else:
                        print(f"  ❌ Error: {e}")
                        raise
            
            # Update version
            conn.execute(
                "INSERT OR REPLACE INTO database_info (key, value, updated_at) VALUES (?, ?, ?)",
                (target_version, target_version, datetime.utcnow().isoformat())
            )
            
            # Log migration
            _log_system_health_conn(
                conn,
                "database",
                "MIGRATED",
                f"Migrated from {current_version} to {target_version}",
                json.dumps({"migrations_applied": len(migrations)})
            )
            
            print(f"✓ Database migrated to version {target_version}")
            return True
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# =========================
# SYSTEM HEALTH LOGGING (PRIVATE HELPER)
# =========================

def _log_system_health_conn(
    conn: sqlite3.Connection,
    component: str,
    status: str,
    message: Optional[str] = None,
    details_json: Optional[str] = None
) -> None:
    """
    PRIVATE: Log system health status (requires connection).
    """
    query = """
        INSERT INTO system_health (timestamp, component, status, message, details_json)
        VALUES (?, ?, ?, ?, ?)
    """
    
    params = (
        datetime.utcnow().isoformat(),
        component,
        status,
        message,
        details_json
    )
    
    try:
        conn.execute(query, params)
    except Exception as e:
        print(f"❌ Error logging system health: {e}")

# =========================
# PUBLIC SYSTEM HEALTH LOGGING
# =========================

def log_system_health(
    component: str,
    status: str,
    message: Optional[str] = None,
    details_json: Optional[str] = None
) -> None:
    """
    PUBLIC: Log system health status.
    """
    with get_connection() as conn:
        _log_system_health_conn(conn, component, status, message, details_json)

# =========================
# INITIALIZATION (MODIFIED)
# =========================

def initialize_storage():
    """
    Initialize storage system.
    """
    print("Initializing storage system...")
    
    try:
        # Initialize database structure
        initialize_database()
        
        # Get current version
        with get_connection() as conn:
            current_version = get_current_feature_version(conn)
        
        # Check and run migrations if needed
        if current_version != FEATURE_VERSION:
            print(f"Current version: {current_version}, Target: {FEATURE_VERSION}")
            print(f"Migrating from {current_version} to {FEATURE_VERSION}")
            success = migrate_database(FEATURE_VERSION)
            
            if not success:
                print("⚠️ Migration may have issues, but continuing...")
        
        print("✓ Storage system initialized")
        
    except Exception as e:
        print(f"❌ Error initializing storage: {e}")
        import traceback
        traceback.print_exc()
        raise

# =========================
# MARKET FEATURES OPERATIONS
# =========================

def insert_market_features(df: pd.DataFrame) -> None:
    """
    Insert feature rows into market_features table.
    """
    if df.empty:
        return
    
    # Ensure feature_version is set
    if "feature_version" not in df.columns:
        df["feature_version"] = FEATURE_VERSION
    
    # Ensure timestamp is string
    df["timestamp"] = df["timestamp"].astype(str)
    
    # Get expected columns from schema
    expected_columns = set(MARKET_FEATURES_SCHEMA["columns"].keys())
    
    # Add any missing columns with NULL
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
    
    # Select only columns that exist in schema
    df_to_insert = df[list(expected_columns)].copy()
    
    # Convert NaN to None for SQL
    df_to_insert = df_to_insert.replace({np.nan: None})
    
    try:
        with get_connection() as conn:
            # Insert data
            df_to_insert.to_sql(
                name="market_features",
                con=conn,
                if_exists="append",
                index=False
            )
            
            print(f"✓ Inserted {len(df)} feature rows")
            
    except Exception as e:
        print(f"❌ Error inserting features: {e}")
        raise

def fetch_latest_features(feature_version: str = FEATURE_VERSION) -> Optional[pd.DataFrame]:
    """
    Fetch the most recent feature row for inference.
    """
    query = """
        SELECT *
        FROM market_features
        WHERE feature_version = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """
    
    try:
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(feature_version,))
            return df if not df.empty else None
            
    except Exception as e:
        print(f"❌ Error fetching features: {e}")
        return None

def fetch_features_by_time_range(
    start_time: str,
    end_time: str,
    feature_version: str = FEATURE_VERSION
) -> pd.DataFrame:
    """
    Fetch features within a time range.
    
    Args:
        start_time: Start timestamp (inclusive)
        end_time: End timestamp (inclusive)
        feature_version: Feature version filter
    
    Returns:
        DataFrame with features in range
    """
    query = """
        SELECT *
        FROM market_features
        WHERE feature_version = ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    """
    
    try:
        with get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(feature_version, start_time, end_time)
            )
            
            log_system_health(
                "features",
                "FETCHED_RANGE",
                f"Fetched {len(df)} features from {start_time} to {end_time}"
            )
            
            return df
            
    except Exception as e:
        print(f"❌ Error fetching features by range: {e}")
        return pd.DataFrame()

def fetch_training_data(
    feature_version: str = FEATURE_VERSION,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    min_confidence: float = 0.0
) -> pd.DataFrame:
    """
    Fetch labeled data for ML training with research features.
    
    Args:
        feature_version: Feature version
        start_ts: Start timestamp
        end_ts: End timestamp
        min_confidence: Minimum signal confidence for training
    
    Returns:
        DataFrame with features and labels
    """
    # Build WHERE clause
    conditions = ["feature_version = ?", "future_return_5m IS NOT NULL"]
    params = [feature_version]
    
    if start_ts:
        conditions.append("timestamp >= ?")
        params.append(start_ts)
    
    if end_ts:
        conditions.append("timestamp <= ?")
        params.append(end_ts)
    
    where_clause = " AND ".join(conditions)
    
    query = f"""
        SELECT mf.*, s.confidence as signal_confidence
        FROM market_features mf
        LEFT JOIN signals s ON mf.timestamp = s.timestamp AND s.status = 'VALIDATED'
        WHERE {where_clause}
        ORDER BY mf.timestamp ASC
    """
    
    try:
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
            # Filter by confidence if needed
            if min_confidence > 0 and 'signal_confidence' in df.columns:
                df = df[df['signal_confidence'] >= min_confidence]
            
            log_system_health(
                "training",
                "FETCHED",
                f"Fetched {len(df)} training samples"
            )
            
            return df
            
    except Exception as e:
        print(f"❌ Error fetching training data: {e}")
        return pd.DataFrame()

# =========================
# RESEARCH ANALYTICS OPERATIONS
# =========================

def insert_research_analytics(analytics: Dict) -> None:
    """
    Insert research analytics for tracking and visualization.
    
    Args:
        analytics: Dictionary with research analytics data
    """
    if not analytics:
        return
    
    # Prepare data
    timestamp = analytics.get("timestamp", datetime.utcnow().isoformat())
    
    data = {
        "timestamp": timestamp,
        "oi_velocity": analytics.get("oi_velocity"),
        "oi_regime": analytics.get("oi_regime"),
        "net_gamma": analytics.get("gamma_exposure", {}).get("net_gamma"),
        "gamma_regime": analytics.get("gamma_exposure", {}).get("regime"),
        "wall_strength": analytics.get("structural_walls", [{}])[0].get("concentration", 0) if analytics.get("structural_walls") else 0,
        "trap_probability": analytics.get("potential_traps", [{}])[0].get("confidence", 0) if analytics.get("potential_traps") else 0,
        "divergence_score": analytics.get("spot_divergence", {}).get("confidence", 0),
        "market_regime": analytics.get("market_regime"),
        "confidence": analytics.get("regime_confidence"),
        "insights_json": json.dumps(analytics.get("market_insights", [])),
        "walls_json": json.dumps(analytics.get("structural_walls", [])),
        "traps_json": json.dumps(analytics.get("potential_traps", [])),
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Clean NaN values
    for key, value in data.items():
        if isinstance(value, float) and np.isnan(value):
            data[key] = None
    
    try:
        with get_connection() as conn:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            values = list(data.values())
            
            conn.execute(
                f"INSERT OR REPLACE INTO research_analytics ({columns}) VALUES ({placeholders})",
                values
            )
            
    except Exception as e:
        print(f"❌ Error inserting research analytics: {e}")

def fetch_recent_analytics(limit: int = 100) -> pd.DataFrame:
    """
    Fetch recent research analytics for visualization.
    
    Args:
        limit: Number of rows to fetch
    
    Returns:
        DataFrame with analytics
    """
    query = """
        SELECT *
        FROM research_analytics
        ORDER BY timestamp DESC
        LIMIT ?
    """
    
    try:
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df
    except Exception as e:
        print(f"❌ Error fetching analytics: {e}")
        return pd.DataFrame()

# =========================
# SIGNAL STORE OPERATIONS (ENHANCED)
# =========================

def insert_signal(signal: Dict) -> None:
    """
    Insert a trading signal with research context.
    
    Args:
        signal: Signal dictionary with research context
    """
    if not signal:
        return
    
    # Add research context if available
    research_context = signal.get("research_context", {})
    if research_context and not isinstance(research_context, str):
        signal["research_context"] = json.dumps(research_context)
    
    # Add analytics summary
    analytics_summary = signal.get("analytics_summary", {})
    if analytics_summary and not isinstance(analytics_summary, str):
        signal["analytics_summary"] = json.dumps(analytics_summary)
    
    # Ensure all required fields
    required_fields = [
        "signal_id", "timestamp", "feature_version", "model_version",
        "signal_type", "confidence", "status", "created_at"
    ]
    
    for field in required_fields:
        if field not in signal:
            print(f"⚠️ Missing required field in signal: {field}")
            signal[field] = ""
    
    try:
        with get_connection() as conn:
            columns = ", ".join(signal.keys())
            placeholders = ", ".join(["?"] * len(signal))
            values = list(signal.values())
            
            conn.execute(
                f"INSERT INTO signals ({columns}) VALUES ({placeholders})",
                values
            )
            
            log_system_health(
                "signals",
                "INSERTED",
                f"Signal {signal['signal_id'][:8]} inserted: {signal['signal_type']} (conf: {signal['confidence']})",
                json.dumps({
                    "signal_id": signal["signal_id"],
                    "type": signal["signal_type"],
                    "confidence": signal["confidence"]
                })
            )
            
    except Exception as e:
        print(f"❌ Error inserting signal: {e}")
        log_system_health(
            "signals",
            "INSERT_FAILED",
            f"Failed to insert signal: {str(e)}"
        )
        raise

def signal_exists(feature_timestamp: str) -> bool:
    """
    Check if a signal already exists for given timestamp.
    
    Args:
        feature_timestamp: Feature timestamp
    
    Returns:
        True if signal exists
    """
    query = """
        SELECT 1 FROM signals WHERE timestamp = ? LIMIT 1
    """
    
    try:
        with get_connection() as conn:
            cur = conn.execute(query, (feature_timestamp,))
            return cur.fetchone() is not None
    except Exception as e:
        print(f"❌ Error checking signal existence: {e}")
        return False

def validate_new_signals(confidence_threshold: float = 0.2):
    """
    Promote NEW signals to VALIDATED if confidence is sufficient.
    
    Args:
        confidence_threshold: Minimum confidence for validation
    """
    query = """
        UPDATE signals
        SET status = 'VALIDATED'
        WHERE status = 'NEW'
          AND confidence >= ?
    """
    
    try:
        with get_connection() as conn:
            result = conn.execute(query, (confidence_threshold,))
            updated_count = result.rowcount
            
            if updated_count > 0:
                log_system_health(
                    "signals",
                    "VALIDATED",
                    f"Validated {updated_count} signals (conf >= {confidence_threshold})"
                )
                
    except Exception as e:
        print(f"❌ Error validating signals: {e}")
        log_system_health(
            "signals",
            "VALIDATION_FAILED",
            f"Signal validation failed: {str(e)}"
        )

def expire_old_signals():
    """
    Expire VALIDATED signals whose expiry_time has passed.
    """
    query = """
        UPDATE signals
        SET status = 'EXPIRED'
        WHERE status = 'VALIDATED'
          AND expiry_time < ?
    """
    
    now = datetime.utcnow().isoformat()
    
    try:
        with get_connection() as conn:
            result = conn.execute(query, (now,))
            expired_count = result.rowcount
            
            if expired_count > 0:
                log_system_health(
                    "signals",
                    "EXPIRED",
                    f"Expired {expired_count} signals"
                )
                
    except Exception as e:
        print(f"❌ Error expiring signals: {e}")

def update_signal_status(signal_id: str, new_status: str, 
                        pnl: Optional[float] = None,
                        exit_time: Optional[str] = None,
                        exit_price: Optional[float] = None) -> None:
    """
    Update signal lifecycle status with trade results.
    
    Args:
        signal_id: Signal ID
        new_status: New status
        pnl: Profit/Loss if applicable
        exit_time: Exit timestamp
        exit_price: Exit price
    """
    # Build update query
    updates = ["status = ?"]
    params = [new_status]
    
    if pnl is not None:
        updates.append("pnl = ?")
        params.append(pnl)
    
    if exit_time:
        updates.append("exit_time = ?")
        params.append(exit_time)
    
    if exit_price is not None:
        updates.append("exit_price = ?")
        params.append(exit_price)
    
    # Calculate trade duration if exit_time is provided
    if exit_time:
        updates.append("""
            trade_duration_seconds = (
                CAST((julianday(?) - julianday(created_at)) * 86400 AS INTEGER)
            )
        """)
        params.append(exit_time)
    
    query = f"""
        UPDATE signals
        SET {', '.join(updates)}
        WHERE signal_id = ?
    """
    
    params.append(signal_id)
    
    try:
        with get_connection() as conn:
            conn.execute(query, params)
            
            log_system_health(
                "signals",
                "UPDATED",
                f"Signal {signal_id[:8]} updated to {new_status}"
            )
            
    except Exception as e:
        print(f"❌ Error updating signal: {e}")
        log_system_health(
            "signals",
            "UPDATE_FAILED",
            f"Failed to update signal {signal_id}: {str(e)}"
        )

def fetch_active_signals(limit: int = 10) -> pd.DataFrame:
    """Fetch active signals for display."""
    query = """
        SELECT *
        FROM signals
        WHERE status IN ('NEW','VALIDATED')
        ORDER BY created_at DESC
        LIMIT ?
    """
    
    try:
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df
    except Exception as e:
        print(f"❌ Error fetching active signals: {e}")
        return pd.DataFrame()

def fetch_signal_performance(days: int = 30) -> Dict:
    """
    Fetch signal performance metrics.
    
    Args:
        days: Number of days to analyze
    
    Returns:
        Dictionary with performance metrics
    """
    cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    query = """
        SELECT 
            signal_type,
            status,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            AVG(pnl) as avg_pnl,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losers
        FROM signals
        WHERE created_at >= ?
        GROUP BY signal_type, status
    """
    
    try:
        with get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(cutoff_date,))
            
            if df.empty:
                return {}
            
            # Calculate overall metrics
            total_signals = df['count'].sum()
            profitable_signals = df[df['signal_type'] == 'BUY']['winners'].sum() + \
                                df[df['signal_type'] == 'SELL']['winners'].sum()
            
            performance = {
                "total_signals": int(total_signals),
                "profitable_signals": int(profitable_signals),
                "win_rate": round(profitable_signals / total_signals * 100, 1) if total_signals > 0 else 0,
                "breakdown": df.to_dict('records'),
                "period_days": days
            }
            
            return performance
            
    except Exception as e:
        print(f"❌ Error fetching performance: {e}")
        return {}

# =========================
# MODEL REGISTRY OPERATIONS
# =========================

def register_model(model_info: Dict) -> None:
    """
    Register trained ML model metadata.
    
    Args:
        model_info: Model information dictionary
    """
    required_keys = [
        "model_version", "feature_version", "algorithm",
        "trained_on_start", "trained_on_end", "metrics_json"
    ]
    
    for key in required_keys:
        if key not in model_info:
            raise ValueError(f"Missing required key: {key}")
    
    # Set created_at if not provided
    if "created_at" not in model_info:
        model_info["created_at"] = datetime.utcnow().isoformat()
    
    try:
        with get_connection() as conn:
            # Deactivate other models for this feature version
            conn.execute("""
                UPDATE model_registry
                SET is_active = 0
                WHERE feature_version = ?
            """, (model_info["feature_version"],))
            
            # Insert new model as active
            columns = ", ".join(model_info.keys())
            placeholders = ", ".join(["?"] * len(model_info))
            values = list(model_info.values())
            
            conn.execute(
                f"INSERT INTO model_registry ({columns}) VALUES ({placeholders})",
                values
            )
            
            log_system_health(
                "models",
                "REGISTERED",
                f"Model {model_info['model_version']} registered"
            )
            
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        raise

def fetch_active_model(feature_version: str = FEATURE_VERSION) -> Optional[Dict]:
    """
    Fetch the active model for a given feature version.
    
    Args:
        feature_version: Feature version
    
    Returns:
        Model info dictionary or None
    """
    query = """
        SELECT *
        FROM model_registry
        WHERE feature_version = ?
          AND is_active = 1
        ORDER BY created_at DESC
        LIMIT 1
    """
    
    try:
        with get_connection() as conn:
            row = conn.execute(query, (feature_version,)).fetchone()
            return dict(row) if row else None
    except Exception as e:
        print(f"❌ Error fetching active model: {e}")
        return None

# =========================
# SYSTEM HEALTH LOGGING (PUBLIC)
# =========================

def fetch_recent_health_logs(limit: int = 50) -> pd.DataFrame:
    """
    Fetch recent system health logs.
    
    Args:
        limit: Number of logs to fetch
    
    Returns:
        DataFrame with health logs
    """
    query = """
        SELECT *
        FROM system_health
        ORDER BY timestamp DESC
        LIMIT ?
    """
    
    try:
        with get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(limit,))
    except Exception as e:
        print(f"❌ Error fetching health logs: {e}")
        return pd.DataFrame()

# =========================
# FEATURE HISTORY TRACKING
# =========================

def track_feature_changes(feature_df: pd.DataFrame) -> None:
    """
    Track significant feature changes for monitoring.
    
    Args:
        feature_df: DataFrame with features
    """
    if feature_df.empty or len(feature_df) < 2:
        return
    
    # Only track important features
    important_features = [
        "oi_velocity", "net_gamma", "trap_probability",
        "divergence_score", "wall_strength", "put_call_ratio"
    ]
    
    current = feature_df.iloc[-1]
    previous = feature_df.iloc[-2] if len(feature_df) > 1 else current
    
    changes = []
    for feature in important_features:
        if feature in current and feature in previous:
            old_val = previous[feature]
            new_val = current[feature]
            
            if old_val is not None and new_val is not None and old_val != 0:
                change_pct = ((new_val - old_val) / abs(old_val)) * 100
                
                # Only track significant changes (>10%)
                if abs(change_pct) > 10:
                    changes.append({
                        "timestamp": current["timestamp"],
                        "feature_name": feature,
                        "old_value": old_val,
                        "new_value": new_val,
                        "change_pct": change_pct,
                        "created_at": datetime.utcnow().isoformat()
                    })
    
    if changes:
        try:
            with get_connection() as conn:
                for change in changes:
                    conn.execute("""
                        INSERT INTO feature_history 
                        (timestamp, feature_name, old_value, new_value, change_pct, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        change["timestamp"], change["feature_name"],
                        change["old_value"], change["new_value"],
                        change["change_pct"], change["created_at"]
                    ))
        except Exception as e:
            print(f"❌ Error tracking feature changes: {e}")

# =========================
# DATABASE MAINTENANCE
# =========================

def cleanup_old_data(days_to_keep: int = 30) -> Dict:
    """
    Clean up old data from database.
    
    Args:
        days_to_keep: Number of days of data to keep
    
    Returns:
        Dictionary with cleanup statistics
    """
    cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
    
    cleanup_stats = {}
    
    try:
        with get_connection() as conn:
            # Clean old market features
            result = conn.execute(
                "DELETE FROM market_features WHERE timestamp < ?",
                (cutoff_date,)
            )
            cleanup_stats["market_features"] = result.rowcount
            
            # Clean old signals
            result = conn.execute(
                "DELETE FROM signals WHERE created_at < ? AND status = 'EXPIRED'",
                (cutoff_date,)
            )
            cleanup_stats["signals"] = result.rowcount
            
            # Clean old system health logs
            result = conn.execute(
                "DELETE FROM system_health WHERE timestamp < ?",
                (cutoff_date,)
            )
            cleanup_stats["system_health"] = result.rowcount
            
            # Clean old research analytics
            result = conn.execute(
                "DELETE FROM research_analytics WHERE timestamp < ?",
                (cutoff_date,)
            )
            cleanup_stats["research_analytics"] = result.rowcount
            
            # Clean old feature history
            result = conn.execute(
                "DELETE FROM feature_history WHERE timestamp < ?",
                (cutoff_date,)
            )
            cleanup_stats["feature_history"] = result.rowcount
            
            # Vacuum database
            conn.execute("VACUUM")
            
            log_system_health(
                "database",
                "CLEANED",
                f"Cleaned up {sum(cleanup_stats.values())} old records",
                json.dumps(cleanup_stats)
            )
            
            return cleanup_stats
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        log_system_health(
            "database",
            "CLEANUP_FAILED",
            f"Cleanup failed: {str(e)}"
        )
        return {}
# =========================
# DATABASE INFO & STATS
# =========================

def get_database_stats() -> Dict:
    """Get database statistics."""
    stats = {}
    
    try:
        with get_connection() as conn:
            # Table counts
            tables = [
                "market_features", "signals", "model_registry",
                "system_health", "research_analytics", "feature_history"
            ]
            
            for table in tables:
                try:
                    cur = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                    row = cur.fetchone()
                    stats[f"{table}_count"] = row["count"] if row else 0
                except:
                    stats[f"{table}_count"] = 0
            
            # Latest feature timestamp
            try:
                cur = conn.execute("SELECT MAX(timestamp) as latest FROM market_features")
                row = cur.fetchone()
                stats["latest_feature"] = row["latest"] if row else None
            except:
                stats["latest_feature"] = None
            
            # Feature version
            stats["feature_version"] = get_current_feature_version(conn) or "unknown"
            
            # Database size
            if DB_PATH.exists():
                db_size = DB_PATH.stat().st_size
                stats["database_size_mb"] = round(db_size / (1024 * 1024), 2)
            else:
                stats["database_size_mb"] = 0
            
            return stats
            
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")
        return {}


# =========================
# INITIALIZATION
# =========================

def initialize_storage():
    """
    Initialize storage system.
    """
    print("Initializing storage system...")
    initialize_database()
    
    # Check and run migrations if needed
    current_version = get_database_stats().get("feature_version", "v1.0")
    if current_version != FEATURE_VERSION:
        print(f"Migrating from {current_version} to {FEATURE_VERSION}")
        migrate_database(FEATURE_VERSION)
    
    print("✓ Storage system initialized")

# Run initialization when module is imported
initialize_storage()